# MICap: A Unified Model for Identity-aware Movie Descriptions


[![Paper](https://img.shields.io/badge/arXiv-2405.11483-B31.svg)](https://arxiv.org/abs/2405.11483)
[![Project Page](https://img.shields.io/badge/project-MICap-green)](https://katha-ai.github.io/projects/micap/)
<a href="https://huggingface.co/spaces/dnaveenr/iSPICE-Metric" target="_blank">
    <img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>


## Overview

This repository contains the official code for CVPR 2024 paper "MICap: A Unified Model for Identity-aware Movie Descriptions". 

![Teaser Image](assets/Teaser.jpeg)

## Installation

Run the following commands to clone the repository
```
git clone https://github.com/katha-ai/MovieIdentityCaptioner-CVPR2024.git
cd MovieIdentityCaptioner-CVPR2024
```

#### Environment

Install the required conda environment by running the following command:
`conda env create -f conda_env.yml`

#### Data

Create a micap_data folder and fill in the path to this folder at the `data_dir` flag in the config_base.yaml file. Now for all the folders below, except for the SPICE Jar file and the Checkpoints, place them in the micap_data folder and put there relative paths into the config file at their specified locations in the instructions column.

| Features | Instructions |
|----------|--------------|
|[Clip Features](https://iiithydstudents-my.sharepoint.com/:u:/g/personal/haran_raajesh_students_iiit_ac_in/Ea2Ul6ja-vxAu-Cewv_dtBMBOAVS8NrMl2VoZYqHCGzeNw?e=9x5BZT) | The unzipped folder path should be filled in for the `input_clip_dir` flag in the config_base.yaml file |
|[Face Features](https://iiithydstudents-my.sharepoint.com/:u:/g/personal/haran_raajesh_students_iiit_ac_in/ERgFYwVBIHtIm_JTBMMybFEBQMwLe1Mb488TIdAheJuSvQ?e=waoScp) | The unzipped folder path should be filled in for the `input_arc_face_dir` flag in the config_base.yaml file | 
|[I3D Features](https://iiithydstudents-my.sharepoint.com/:u:/g/personal/haran_raajesh_students_iiit_ac_in/ESyImmMNkkRMqff92aif5ZEBbdPibAVK4AlJNaRHwXzQWA?e=xg1So9) | The unzipped folder path should be filled in for the `input_fc_dir` flag in the config_base.yaml file |  
|[Face Clusters](https://iiithydstudents-my.sharepoint.com/:u:/g/personal/haran_raajesh_students_iiit_ac_in/ERLchwaL2w9Nr8rHQzQdfaYBaCI5mElOMyuTo16B0neZaA?e=L8ahJS) | The unzipped file path should be filled in for the `input_arc_face_clusters` flag in the config_base.yaml file | 
|[MICap Json](https://iiithydstudents-my.sharepoint.com/:u:/g/personal/haran_raajesh_students_iiit_ac_in/EdoD16HKm4NBlrHGSyj9pVgBPP19E3EmfMBieRuCFhrWOw?e=DBBVtz) | The unzipped file path should be filled in for the `input_json` flag in the config_base.yaml file | 
|[Bert Text Embeddings](https://iiithydstudents-my.sharepoint.com/:u:/g/personal/haran_raajesh_students_iiit_ac_in/EQklOX8PqAZKowufcViYlm4BV6Itsrs1sq6V-Fpb2-mb1w?e=pze2Yr) | The unzipped folder path (`fillin_data/bert_text_gender_embedding`) should be filled in for the `bert_embedding_dir` flag in the config_base.yaml file |
|[H5 label file](https://iiithydstudents-my.sharepoint.com/:u:/g/personal/haran_raajesh_students_iiit_ac_in/ER4Tq0P2SOtPsr5j9VFoPE0BhOx5lU2ZxZXmMRCYpe1Z7w?e=EfYYrn) | The unzipped file path (`LSMDC16_labels_fillin_new_augmented.h5`) should be filled in for the `input_label_h5` flag in the config_base.yaml file |
|[Tokenizer](https://iiithydstudents-my.sharepoint.com/:u:/g/personal/haran_raajesh_students_iiit_ac_in/EdsXVAXTza5OqiwILYwtWhsBdw4YQzSG1fVu22IM2Q8tHw?e=kxGwfj) | The unzipped folder path should be filled in for the `tokenizer_path` flag in the config_base.yaml file |
|[SPICE Jar file](https://iiithydstudents-my.sharepoint.com/:u:/g/personal/haran_raajesh_students_iiit_ac_in/Ee4LJhU0QlNIpIbcIeBeYScBKKVaDmLexWXQ_KC_kUCsGA?e=BX6uga) | The unzipped file path should be placed in the `iSPICE` directory |
|[Checkpoints](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/haran_raajesh_students_iiit_ac_in/EmV8Bw4MVL5KiXqnoEu7HRwBlYDO0Hy86uErIJBXYvtksw?e=mTiBLj) | Folder that contains the various checkpoints for full captioning and joint training full captioning (cider score) and fitb and joint training fitb (class accuracy)|






## Training

The `run_type` flag in the `config_base.yaml` file can be adjusted to determine the task (either `fitb`, `fc` only, or both) for training MICap.

Make sure the `overfit` and `checkpoint` flags are set to `False`. Also, ensure the path relative to the features from the data directory is correctly set in the `config_base.yaml` file.

Once the yaml file is set run the command:`python train_mod.py`

## Evaluation

To evaluate a pretrained model, set the `checkpoint` flag to `True` in the `config_base.yaml` file.

The `run_type` flag in the `config_base.yaml` file can be adjusted to specify the task for evaluation.

Once the yaml file is set run the command:`python train_mod.py`


## Citation

Please consider citing our paper if the project helps your research with the following BibTex:

```bibtex
@inproceedings{raajesh2024micap,
  title={MICap: A Unified Model for Identity-aware Movie Descriptions},
  author={Raajesh, Haran and Desanur, Naveen Reddy and Khan, Zeeshan and Tapaswi, Makarand},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14011--14021},
  year={2024}
}
```
