# CAMD: Context-Aware Masked Distillation for General Self-Supervised Facial Representation Pre-Training

[Sensen Wang]()<sup>1</sup> &emsp; [Si Chen]()<sup>1</sup> &emsp; [Da-Han Wang]()<sup>1</sup> &emsp;
[Yang Hua]()<sup>2</sup> &emsp; [Yan Yan]()<sup>3</sup>


<sup>1</sup>Fujian Key Laboratory of Pattern Recognition and Image Understanding, School of Computer and Information Engineering, Xiamen University of Technology <br>
<sup>2</sup>Institute of Electronics, Communications and Information Technology, Queen’s University Belfast <br>
<sup>3</sup>School of Informatics, Xiamen University <br>

## Release🎉 

*  **2025-9.19**: All codes including data-preprocessing, pre-training, fine-tuning, and testing are released at [this page](https://github.com/mumumu-wss/CAMD/edit/main)

## Table of Contents

  - [🔧 Installation](#installation)
  - [⏳ CAMD Pre-training](#CAMD-pretraining)
    - [⚫ Pre-training Data](#pre-training-data)
      - [⬇️ Dataset Preparation](#pt-dataset-preparation)
      - [⬇️ Toolkit Preparation](#toolkit-preparation)
      - [📁 Folder Structure](#folder-structure)
    - [⚫ Pre-training Model](#pre-training-model)
      - [🚀 Model and Data Scaling](#model-and-data-scaling)
      - [💾 Pre-training/Resume from Checkpoint](#resume-for-pretraining)
    - [🤗 Pre-trained Checkpoints](#pre-trained-model)
      - [📥 Download Manually](#download-manually)
      - [💻 Download Script](#download-script)
  - [⚡ Fine-tuning CAMD Pre-trained ViTs for Downstream Tasks](#CAMD-finetuning)
    - [⚫ Face Parsing and Face Alignment](#fpafa)
      - [⬇️ Dataset Preparation](#dfd-dataset-preparation)
      - [⚡ Fine-tuning](#dfd-finetuning)
        - [✨ Fine-tuning with different dataset structure](#finetuning-different-dataset) 
      - [📊 Cross-Datasets Evaluation](#dfd-testing)
    - [⚫ Facial Attribute Recognition](#far)
      - [⬇️ Dataset Preparation](#far-dataset-preparation)
      - [⚡ Fine-tuning](#far-finetuning)
      - [📊 Cross-Datasets Evaluation](#diff-testing)
    - [⚫ Head Pose Estimation](#HPE)
      - [⬇️ Dataset Preparation](#fas-dataset-preparation)
      - [⚡ Fine-tuning and Evaluation](#fas-finetuning)
  - [Citation](#citing-CAMD)

---

# 🔧 Installation 

<a id="installation"></a>
Git clone this repository, creating a conda environment, and activate it via the following command: 

```bash
git clone https://github.com/mumumu-wss/CAMD.git
cd CAMD/
conda env create -f ./environment.yml
```

---

#  🚀 CAMD Pre-training

<a id="fsfm-pretraining"></a>
The implementation of pre-training CAMD ViT models from unlabeled facial images.

<a id="download-script"></a>

## ⚫ Pre-training Data 

<a id="pre-training-data"></a>

<details>
<a id="pt-dataset-preparation"></a>
<summary style="font-size: 20px; font-weight: bold;">⬇️ Dataset Preparation</summary>

For paper implementation, we have pre-trained our model on the following datasets. Download these datasets optionally and refer to [Folder Structure](#folder-structure).

- [VGGFace2](https://github.com/ox-vgg/vgg_face2) _for main experiments (raw data: images)_ 
</details>
<details>
<a id="folder-structure"></a>
<summary style="font-size: 20px; font-weight: bold;">📁 Folder Structure</summary>


> You need to modify the path corresponding to the file in `run_CAMD_PRETRAIN.sh`

The following is the **default Folder Structure**. The paths in each directory are described in the comments. 

```bash
datasets/
├── pretrain/
│   ├── VGG-Face2/    # VGGFace2
│   │   ├── train/    # download data
│   │   ├── test/    # download data
```

</details>

## ⚫ Pre-training Model

<a id="pre-training-model"></a>

<details>
<a id="pre-training-from-scratch"></a>
<summary style="font-size: 20px; font-weight: bold;">🔄 Pre-training from Scratch</summary>

`cd CAMD` and run the script `sh run_CAMD_pretrain.sh` to pre-train the model.
</details>


<details>
<a id="model-and-data-scaling"></a>
<summary style="font-size: 20px; font-weight: bold;">🚀 Model and Data Scaling</summary>


- **Model Scaling.** To pre-train ViT-Small, ViT-Base, ViT-Large, or ViT-Huge, set `--model` to one of:

  ```
  --model [CAMD_vit_base_patch16, CAMD_vit_large_patch16, CAMD_vit_huge_patch14 (with --patch_size 14)]
  ```

  </details>

## 🤗 Pre-trained Checkpoints

<a id="pre-trained-model"></a>

<details>
<a id="download-manually"></a>
<summary style="font-size: 20px; font-weight: bold;">📥 Download Manually</summary>


We provide the model weights.

coming soon.
</details>
---

#  ⚡ Fine-tuning CAMD Pre-trained ViTs for Downstream Tasks

<a id="CAMD-finetuning"></a>
The implementation of fine-tuning pre-trained model on various downstream face security-related tasks.


## ⚫ Facial Attribute Recognition

<a id="far"></a>

<details style="margin-left: 20px;">
<a id="diff-dataset-preparation"></a>
<summary style="font-size: 20px; font-weight: bold;">⬇️ Dataset Preparation</summary>


We train and test on CelebA and LFWA respectively. Download these datasets and refer to [FAR Folder Structure](#far-folder-structure).

- [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [LFWA](https://drive.google.com/drive/folders/0B7EVK8r0v71pQ3NzdzRhVUhSams?resourcekey=0-Kpdd6Vctf-AdJYfS55VULA)
</details>
<details style="margin-left: 20px;">
<a id="diff-folder-structure"></a>
<summary style="font-size: 20px; font-weight: bold;">📁 FAR Folder Structure</summary>


The following is the **default Folder Structure** for unseen FAR. The paths in each directory are described in the comments. 

```bash
datasets/
├── downstream/
│   ├── CelebA/
│   │   ├── Anno/
│   │   ├── Eval/
│   │   ├── img_align_celeba/
│   ├── LFWA/
│   │   ├── lfw/
│   │   ├── lfw_attributes.txt/
```
</details>
<details style="margin-left: 20px;">
<a id="diff-finetuning"></a>
<summary style="font-size: 20px; font-weight: bold;">⚡ Fine-tuning</summary>

`cd CAMD` and run the script `sh run_CAMD_far_finetune.sh` to fine-tune the model:

</details>
## ⚫ Facial Expression Recognition  (FER)

<a id="fer"></a>

<details style="margin-left: 20px;">
<a id="fas-dataset-preparation"></a>
<summary style="font-size: 20px; font-weight: bold;">⬇️ Dataset Preparation</summary>

- We train and test on ferplus and RAF-DB respectively. Download these datasets and refer to [FER Folder Structure](#far-folder-structure).

  - [ferplus](https://github.com/microsoft/FERPlus)
  - [RAF-DB](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset)

  ```bash
  datasets/
  ├── downstream/
  │   ├── ferplus/
  │   │   ├── data/
  │   │   ├── fer2013.csv/
  │   │   ├── fer2013new.csv/
  │   ├── RAF-DB/
  │   │   ├── basic/
  │   │   ├── compound/
  ```

  </details>

<details style="margin-left: 20px;">
<a id="fas-finetuning"></a>
<summary style="font-size: 20px; font-weight: bold;">⚡ Fine-tuning and Evaluation</summary>

`cd CAMD` and run the script `sh run_CAMD_fer_finetune.sh` to fine-tune the model:
</details>
---

## ⚫ Face Parsing and Face Alignment (FP and FA)

<a id="fpandfa"></a>

For the implementation of this part of the downstream tasks, please refer to [FaRL](https://github.com/faceperceiver/farl?tab=readme-ov-file#setup-downstream-training).

## ⚫ Head Pose Estimation (HPE)

<a id="fpandfa"></a>

For the implementation of this part of the downstream tasks, please refer to [TokenHPE](https://github.com/zc2023/TokenHPE)

# Citation

<a id="citation"></a>

If our research helps your work, please consider giving us a star ⭐ or citing us:

```
@inproceedings{
}
```
