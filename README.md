# ScgMamba: Structure-Guided Cascaded Graph-Mamba for 3D Human Pose Estimation"

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

This is the official PyTorch implementation of paper "ScgMamba: Structure-Guided Cascaded Graph-Mamba for 3D Human Pose Estimation".

## Environment

The project is developed under the following environment:

- Python 3.8.20
- Pytorch 2.01
- CUDA 11.8

For installation of the project dependencies, please run:

```
conda create -n ScgMamba python=3.8.20
conda activate ScgMamba
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url 
https://download.pytorch.org/whl/cu118
cd models/selective_scan && pip install -e .
pip install -r requirements.txt
```

## Dataset

### Human3.6M

#### Preprocessing

1. Download the fine-tuned Stacked Hourglass detections of [MotionBERT](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md)'s preprocessed H3.6M data [here](https://1drv.ms/u/s!AvAdh0LSjEOlgU7BuUZcyafu8kzc?e=vobkjZ) and unzip it to 'data/motion3d', or direct download our processed data [here](https://drive.google.com/file/d/1WWoVAae7YKKKZpa1goO_7YcwVFNR528S/view?usp=sharing) and unzip it.
2. Slice the motion clips by running the following python code in `common/convert_h36m.py`:

```text
python convert_h36m.py
```

### MPI-INF-3DHP

#### Preprocessing

Please refer to [P-STMO](https://github.com/paTRICK-swk/P-STMO#mpi-inf-3dhp) for dataset setup. After preprocessing, the generated .npz files (`data_train_3dhp.npz` and `data_test_3dhp.npz`) should be located at `data/motion3d` directory.

## Training

After dataset preparation, you can train the model as follows:

### Human3.6M

You can train Human3.6M with the following command:

```
 python train.py --config <PATH-TO-CONFIG> --checkpoint <PATH-TO-CHECKPOINT>
```

where config files are located at `configs/h36m`. 

### MPI-INF-3DHP

You can train MPI-INF-3DHP with the following command:

```
python train_3dhp.py --config <PATH-TO-CONFIG> --new-checkpoint <PATH-TO-CHECKPOINT>
```

## Evaluation

| Method     | frames | Params | MACs   | Human3.6M weights                                            |
| ---------- | ------ | ------ | ------ | ------------------------------------------------------------ |
| ScgMamba-S | 243    | 0.52M  | 2.17G  | [download](https://drive.google.com/file/d/1AbWanhcxyAJR7inwrqD2VMPt3xiIY4E6/view?usp=sharing) |
| ScgMamba-B | 243    | 2.03M  | 8.40G  | [download](https://drive.google.com/file/d/17sOfccjyYg6HOo9MNOTjfweKf9jtltIy/view?usp=sharing) |
| ScgMamba-L | 243    | 4.05M  | 16.80G | [download](https://drive.google.com/file/d/1GJQbKubysDSQMUljnUhF15sUcJnO46EO/view?usp=sharing) |

After downloading the weight, you can evaluate Human3.6M models by:

```
python train.py --eval-only --checkpoint <CHECKPOINT-DIRECTORY> --checkpoint-file <CHECKPOINT-FILE-NAME> --config <PATH-TO-CONFIG>
```

## Demo

Our demo is a modified version of the one provided by [MotionAGFormer](https://github.com/taatiteam/MotionAGFormer) repository. First, you need to download YOLOv3 and HRNet pretrained models [here](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA?usp=sharing) and put it in the './demo/lib/checkpoint' directory. Next, download our base model checkpoint from [here](https://drive.google.com/file/d/1Iii5EwsFFm9_9lKBUPfN8bV5LmfkNUMP/view) and put it in the './checkpoint' directory. Then, you need to put your in-the-wild videos in the './demo/video' directory.

Run the command below:

```
python vis.py --video 
```


## Acknowledgement

Our code refers to the following repositories:

- [MotionBERT](https://github.com/Walter0807/MotionBERT)
- [P-STMO](https://github.com/paTRICK-swk/P-STMO)
- [PoseMamba](https://github.com/nankingjing/PoseMamba)
- [MotionAGFormer](https://github.com/taatiteam/MotionAGFormer)

We thank the authors for releasing their codes.

