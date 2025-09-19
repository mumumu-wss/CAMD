import random
from functools import partial
from turtle import update
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from util.pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_relative
from util.misc import LayerNorm
from models_vit import Block, CrossBlock, PatchEmbed


class PermuteBN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        x = x.permute(0, 2, 1)  # N, L, C -> N, C, L
        x = x.float()
        x = self.bn(x)
        x = x.permute(0, 2, 1)  # N, C, L -> N, L, C

        return x


class MultiHeadAttentionTopK(nn.Module):
    def __init__(self, embed_dim, num_heads, k):
        super(MultiHeadAttentionTopK, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.k = k
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, local_feat, global_feat):
        batch_size = local_feat.shape[0]
        query = self.query(local_feat)
        key = self.key(global_feat)
        value = self.value(global_feat)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        topk_values, topk_indices = torch.topk(attn_scores, self.k,
                                               dim=-1)
        batch_indices = torch.arange(batch_size).view(-1, 1, 1, 1).expand(-1, self.num_heads, topk_indices.size(2),
                                                                          self.k)
        head_indices = torch.arange(self.num_heads).view(1, -1, 1, 1).expand(batch_size, -1, topk_indices.size(2),
                                                                             self.k)
        topk_global_feat = value[
            batch_indices, head_indices, topk_indices]

        topk_values = topk_values.softmax(dim=-1)
        attended_global_feat = torch.sum(topk_values.unsqueeze(-1) * topk_global_feat,
                                         dim=-2)

        attended_global_feat = attended_global_feat.transpose(1, 2).contiguous().view(batch_size, -1,
                                                                                      self.num_heads * self.head_dim)
        output = self.fc_out(attended_global_feat)
        return output, topk_values


class CAMDViT(nn.Module):
    """  SiameseIM with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=LayerNorm, norm_pix_loss=False, args=None, SelectTopK =2):
        super().__init__()
        self.norm_pix_loss = norm_pix_loss
        self.args = args
        decoder_embed_dim = args.decoder_embed_dim
        # --------------------------------------------------------------------------
        # cross-scale attention
        SelectTopK = self.args.select_top_k
        self.multihead_attention_topk = MultiHeadAttentionTopK(embed_dim, num_heads, SelectTopK)

        # --------------------------------------------------------------------------
        # encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if args.use_abs_pos_emb:
            if hasattr(self, 'cls_token'):
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding
            else:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        dpr = [x.item() for x in torch.linspace(0, args.drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  drop_path=dpr[i], init_values=args.init_values)
            for i in range(depth)])
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # decoder specifics
        if args.loss_type in ['mae']:
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            if hasattr(self, 'cls_token'):
                self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                                      requires_grad=False)  # fixed sin-cos embedding
            else:
                self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
                                                      requires_grad=False)

            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)])

            self.decoder_norm = norm_layer(decoder_embed_dim)
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        elif args.loss_type in ['CAMD', ]:
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            if hasattr(self, 'cls_token'):
                self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                                      requires_grad=False)  # fixed sin-cos embedding
            else:
                self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
                                                      requires_grad=False)

            if args.projector_depth > 0:
                self.projector_decoder_blocks = nn.ModuleList([
                    Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,
                          norm_layer=norm_layer if args.use_proj_ln else PermuteBN)
                    for i in range(args.projector_depth)])

            self.predictor_decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,
                      norm_layer=norm_layer if args.use_pred_ln else PermuteBN)
                for i in range(args.predictor_depth)])

            self.decoder_pred = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=True)  # decoder to patch
            if args.online_ln:
                self.student_norm = LayerNorm(decoder_embed_dim)
                for p in self.student_norm.parameters():
                    p.requires_grad = False
            else:
                self.student_norm = nn.Identity()
        # --------------------------------------------------------------------------

        # ---------------------------------------------------------------------------
        # decoder pos embed change dim
        if self.args.loss_type in ['CAMD', ]:
            self.decoder_pos_mlp = nn.Linear(decoder_embed_dim * 2, decoder_embed_dim)
        # ---------------------------------------------------------------------------

        self.initialize_weights()

        # build momentum branch
        if self.args.loss_type in ['CAMD', ]:
            self.build_momentum_target(img_size, patch_size, in_chans, embed_dim, num_heads,
                                       mlp_ratio, norm_layer, depth, decoder_embed_dim, decoder_num_heads)

        # stop grad for patch embedding
        if (not args.train_patch_embed):
            self.patch_embed.proj.weight.requires_grad = False
            self.patch_embed.proj.bias.requires_grad = False

    def build_momentum_target(self, img_size, patch_size, in_chans, embed_dim, num_heads,
                              mlp_ratio, norm_layer, depth, decoder_embed_dim, decoder_num_heads):
        # --------------------------------------------------------------------------
        # momentum encoder specifics
        self.mm_patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        if hasattr(self, 'cls_token'):
            self.mm_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.mm_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  init_values=self.args.init_values)
            for i in range(depth)])

        # load weight
        self.mm_patch_embed.load_state_dict(self.patch_embed.state_dict())
        for p in self.mm_patch_embed.parameters():
            p.requires_grad = False

        self.mm_cls_token.data.copy_(self.cls_token.data)
        self.mm_cls_token.requires_grad = False

        self.mm_blocks.load_state_dict(self.blocks.state_dict())
        for p in self.mm_blocks.parameters():
            p.requires_grad = False
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # momentum decoder specifics
        self.mm_decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mm_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if self.args.projector_depth > 0:
            self.mm_projector_decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,
                      norm_layer=norm_layer if self.args.use_proj_ln else PermuteBN)
                for i in range(self.args.projector_depth)])

        # load weight
        self.mm_decoder_embed.load_state_dict(self.decoder_embed.state_dict())
        for p in self.mm_decoder_embed.parameters():
            p.requires_grad = False

        self.mm_mask_token.data.copy_(self.mask_token.data)
        self.mm_mask_token.requires_grad = False

        if self.args.projector_depth > 0:
            self.mm_projector_decoder_blocks.load_state_dict(self.projector_decoder_blocks.state_dict())
            for p in self.mm_projector_decoder_blocks.parameters():
                p.requires_grad = False
        # ---------------------------------------------------------------------------

        if self.args.loss_type in ['CAMD', ]:
            self.teacher_norm = LayerNorm(decoder_embed_dim, elementwise_affine=False)
            for p in self.teacher_norm.parameters():
                p.requires_grad = False

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.args.use_abs_pos_emb:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5),
                                                cls_token=hasattr(self, 'cls_token'))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if hasattr(self, 'decoder_pos_embed'):
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches ** .5),
                                                        cls_token=hasattr(self, 'cls_token'))
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        if hasattr(self, 'cls_token'):
            torch.nn.init.normal_(self.cls_token, std=.02)
        if hasattr(self, 'mask_token'):
            torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask, ids_restore

    @torch.cuda.amp.autocast(enabled=False)
    def mm_update(self, mm):
        for param_q, param_k in zip(self.patch_embed.parameters(), self.mm_patch_embed.parameters()):
            param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        for param_q, param_k in zip(self.blocks.parameters(), self.mm_blocks.parameters()):
            param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        if hasattr(self, 'mm_cls_token'):
            self.mm_cls_token.data = self.mm_cls_token.data * mm + self.cls_token.data * (1. - mm)
        if hasattr(self, 'mm_norm'):
            for param_q, param_k in zip(self.norm.parameters(), self.mm_norm.parameters()):
                param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        if hasattr(self, 'mm_projector'):
            for param_q, param_k in zip(self.projector.parameters(), self.mm_projector.parameters()):
                param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        if hasattr(self, 'mm_decoder_embed'):
            for param_q, param_k in zip(self.decoder_embed.parameters(), self.mm_decoder_embed.parameters()):
                param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        if hasattr(self, 'mm_mask_token'):
            self.mm_mask_token.data = self.mm_mask_token.data * mm + self.mask_token.data * (1. - mm)
        if hasattr(self, 'mm_decoder_blocks'):
            for param_q, param_k in zip(self.decoder_blocks.parameters(), self.mm_decoder_blocks.parameters()):
                param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        if hasattr(self, 'mm_projector_decoder_blocks'):
            for param_q, param_k in zip(self.projector_decoder_blocks.parameters(),
                                        self.mm_projector_decoder_blocks.parameters()):
                param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        if hasattr(self, 'mm_decoder_norm'):
            for param_q, param_k in zip(self.decoder_norm.parameters(), self.mm_decoder_norm.parameters()):
                param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        if hasattr(self, 'mm_decoder_pred'):
            for param_q, param_k in zip(self.decoder_pred.parameters(), self.mm_decoder_pred.parameters()):
                param_k.data = param_k.data * mm + param_q.data * (1. - mm)

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        # masking: length -> length * mask_ratio
        mask, ids_restore = self.random_masking(x, mask_ratio)
        x = x[~mask.bool()].view(x.shape[0], -1, x.shape[-1])

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = x_

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_mae(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def forward(self, *args, **kwargs):
        if self.args.loss_type in ['CAMD', ]:
            return self.forward_CAMD(*args, **kwargs)
        else:
            return self.forward_mae(*args, **kwargs)

    def AGMD(self, local_features, global_features):
        layers_to_distill = self.args.dis_lay
        total_distill_loss = 0

        for idx in layers_to_distill:
            local_feat = local_features[idx]
            global_feat = global_features[idx]

            attended_global_feat, topk_values = self.multihead_attention_topk(local_feat, global_feat)

            global_avg_pool = attended_global_feat.mean(dim=1)
            local_avg_pool = local_feat.mean(dim=1)

            distill_loss = torch.nn.functional.mse_loss(local_avg_pool, global_avg_pool)
            total_distill_loss += distill_loss

        return total_distill_loss

    def forward_CAMD(self, x1, x2, rel_pos_21, mm, update_mm, mask=None):
        # forward online encoder
        if self.args.with_blockwise_mask:
            assert mask is not None, 'mask should not be None when mask_type is block'
            mask = mask.view(mask.shape[0], -1)
        else:
            assert False
            mask, ids_restore1 = self.random_masking(online_x1, self.args.mask_ratio)

        local_features = []
        global_features = []

        # get patch
        online_x1 = self.patch_embed(x1)  # torch.Size([256, 196, 768])

        # online Encoder
        online_x1 = online_x1 + self.pos_embed[:, 1:, :]  # torch.Size([256, 196, 768])
        online_x1 = online_x1[~mask.bool()].view(online_x1.shape[0], -1,
                                                 online_x1.shape[-1])  # torch.Size([256, 78, 768])
        # add cls token
        cls_tokens = self.cls_token.expand(online_x1.shape[0], -1, -1) + self.pos_embed[:, 0, :].unsqueeze(1)
        online_x1 = torch.cat((cls_tokens, online_x1), dim=1)  # # torch.Size([256, 79, 768])

        # forward online encoder
        for blk in self.blocks:
            online_x1 = blk(online_x1)  # torch.Size([256, 79, 768])
            local_features.append(online_x1[:, 1:, :])

        # forward online projector
        online_x1 = self.decoder_embed(online_x1)
        if self.args.projector_depth > 0:
            for blk in self.projector_decoder_blocks:
                online_x1 = blk(online_x1)

        # online decoder
        # calculate decoder pos embed
        cls_pos_embed = self.decoder_pos_embed[:, 0, :].unsqueeze(
            1)  # 提取 [CLS]token的位置嵌入       decoder_pos_embed[:, 0, :].shape:(batch_size, embed_dim) → (batch_size, 1, embed_dim)
        x1_vis_embed = self.decoder_pos_embed[:, 1:, :].repeat(online_x1.shape[0], 1, 1)[~mask.bool()].view(
            online_x1.shape[0], -1, self.decoder_pos_embed.shape[-1])  # torch.Size([256, 78, 768])
        # (batch_size, num_visible_tokens, embed_dim)
        x2_embed = get_2d_sincos_pos_embed_relative(*rel_pos_21, self.decoder_pos_embed.shape[-1],
                                                    int(self.num_patches ** .5))  # torch.Size([256, 196, 1536])
        x2_embed = self.decoder_pos_mlp(x2_embed)  # torch.Size([256, 196, 768])
        # append mask tokens to sequence
        cls_token = online_x1[:, 0, :].unsqueeze(1)
        x1_vis_tokens = online_x1[:, 1:, :]
        mask_tokens = self.mask_token.repeat(x2.shape[0], x2_embed.shape[1],
                                             1)  # [batch_size,,decoder_embed_dim]       # torch.Size([1, 1, 768])->torch.Size([256, 196, 768])
        x = torch.cat([cls_token + cls_pos_embed, x1_vis_tokens + x1_vis_embed, mask_tokens + x2_embed],
                      dim=1)  # torch.Size([256, 275, 768])

        # forward online decoder
        for blk in self.predictor_decoder_blocks:
            x = blk(x)

        # predictor projection
        x = self.decoder_pred(x)
        pred = x[:, -x2_embed.shape[1]:]

        # forward target encoder
        with torch.no_grad():
            if update_mm:
                self.mm_update(mm)
            target_x2 = self.mm_patch_embed(x2)  # torch.Size([256, 196, 768])
            mm_cls_tokens = self.mm_cls_token.expand(target_x2.shape[0], -1, -1)
            target_x2 = torch.cat((mm_cls_tokens, target_x2), dim=1)  # torch.Size([256, 197, 768])
            target_x2 = target_x2 + self.pos_embed  # torch.Size([256, 197, 768])

            # forward target encoder
            for blk in self.mm_blocks:
                target_x2 = blk(target_x2)  # torch.Size([256, 197, 768])
                global_features.append(target_x2[:, 1:, :])

            # forward target projector
            target_x2 = self.mm_decoder_embed(target_x2)
            if self.args.projector_depth > 0:
                for blk in self.mm_projector_decoder_blocks:
                    target_x2 = blk(target_x2)

            target = target_x2[:, 1:, :]

        distill_loss = self.AGMD(local_features, global_features)

        # compute loss
        outputs = {}
        with torch.cuda.amp.autocast(enabled=False):
            loss = self.compute_unigrad_loss(pred.float(), target.float())
            global_loss = self.Globalloss(pred.float(), target.float())
        outputs['loss_CAMD'] = loss.item()
        outputs['global_loss'] = global_loss.item()
        outputs['loss_ds'] = distill_loss.item()

        return loss, global_loss, distill_loss, outputs

    def compute_unigrad_loss(self, pred, target):
        pred = self.student_norm(pred)
        with torch.no_grad():
            target = self.teacher_norm(target)

        dense_pred = pred.reshape(-1, pred.shape[-1])
        dense_target = target.reshape(-1, target.shape[-1])

        # compute pos term
        pos_term = ((dense_pred - dense_target) ** 2).sum(-1).mean()
        # compute neg term
        correlation = (dense_target.T @ dense_target) / dense_target.shape[0]

        if dist.is_initialized():
            dist.all_reduce(correlation)
            correlation = correlation / dist.get_world_size()

        neg_term = torch.diagonal(dense_pred @ correlation @ dense_pred.T).mean()

        loss = (pos_term + self.args.neg_weight * neg_term) / pred.shape[-1]

        return loss

    def Globalloss(self, pred, target):
        # get average
        global_pred = pred.mean(dim=0)
        global_target = target.mean(dim=0)

        pos_term_global = ((global_pred - global_target) ** 2).sum(-1).mean()

        correlation_global = (global_target.T @ global_target) / global_target.shape[0]

        if dist.is_initialized():
            dist.all_reduce(correlation_global)
            correlation_global = correlation_global / dist.get_world_size()

        neg_term_global = torch.diagonal(global_pred @ correlation_global @ global_pred.T).mean()

        loss_global = (pos_term_global + self.args.neg_weight * neg_term_global) / global_pred.shape[-1]

        return loss_global


def CAMD_vit_base_patch16_dec512d8b(**kwargs):
    model = CAMDViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(LayerNorm, eps=1e-6), **kwargs)
    return model


def CAMD_vit_large_patch16_dec512d8b(**kwargs):
    model = CAMDViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(LayerNorm, eps=1e-6), **kwargs)
    return model


def CAMD_vit_huge_patch14_dec512d8b(**kwargs):
    model = CAMDViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
CAMD_vit_base_patch16 = CAMD_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
CAMD_vit_large_patch16 = CAMD_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
CAMD_vit_huge_patch14 = CAMD_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
