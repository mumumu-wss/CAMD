import torch

attr_threshold_InFace = torch.Tensor([0.5 for i in range(40)])

def accuracy(output, target, num,attr_threshold=attr_threshold_InFace,detail=False,nums=40):
    attr_threshold=torch.Tensor([0.5 for i in range(nums)]) #大小为40的一个张量，每个元素都是0.5
    predict_all = torch.gt(output, attr_threshold.cuda())   #讲output的值和阈值做比较
    target=target>0
    correct = target == predict_all
    correct_sum = correct.sum(dim=0)
    acc= correct_sum.float()/num
    if detail:
        return num,correct_sum.float(),acc.mean()

    return acc.mean()