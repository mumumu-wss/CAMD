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
  - [⚡ Fine-tuning FSFM Pre-trained ViTs for Downstream Tasks](#fsfm-finetuning)
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
