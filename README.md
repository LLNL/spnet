# Swap Path Network for Robust Person Search Pre-training

Repository for paper, "Swap Path Network for Robust Person Search Pre-training". [arXiv](https://arxiv.org/abs/2412.05433) [IEEE](https://ieeexplore.ieee.org/document/10943876)
Implements pre-training and fine-tuning of the Swap Path Net (SPNet) for person search.
Contains code, configs, and instructions to reproduce all results from the paper.

## Setup

### Installation

We provide support for installation via docker (preferred) and conda.

#### Docker
We recommend using docker as the most reliable and maintainable environment. We provide an example Dockerfile, with package versions set to those installed in a working version of the image. Most package versions can be updated as needed, but a necessary dependency is pytorch-lightning=2.0.2.

To build the Dockerfile:

```
host$ docker build --no-cache -t osr:v1.0.0 -f Dockerfile .
```

Example docker run command:


```
host$ docker run -it --rm \
        --ulimit core=0 \
        --name=osr_$(date +%F_%H-%M-%S) \
        --gpus=all \
        --net=host \
        -v /dev/shm:/dev/shm \
        -v <COCO_PATH>:/datasets/coco \
        -v <PRW_PATH>:/datasets/prw \
        -v <CUHK_PATH>:/datasets/cuhk \
        -v $(pwd)/weights/hub/checkpoints:/weights/hub/checkpoints \
        -v $(pwd):/home/username \
        -w /home/username \
        osr:v1.0.0 bash -c \
                "chown -R $(id -u):$(id -g) /home/username;\
                 groupadd -g $(id -g) groupname;\
                 useradd -u $(id -u) -g $(id -g) -d /home/username username;\
                 chmod -R 777 /weights/hub/checkpoints;\
                 su username -s /bin/bash;"

container$ export PATH=${PATH}:/opt/conda/bin
```

You can also re-install in the container with:

```
container$ python setup.py install --user
```

We recommend removing spnet package installation within the docker image if you plan to do this.

#### Conda
The only crucial version dependency for a python package is pytorch-lightning=2.0.2. We provide an example conda installation in conda.yaml: 

```
(base)$ conda env create -f conda.yaml

(base)$ conda activate osr

(osr)$ python setup.py install --user
```

To get versions of pytorch with CUDA that work with your system, the most reliable option is to install pytorch from the instructions on pytorch.org. For our use case this was the command:

```
(osr)$ conda install pytorch==1.13.1 torchvision cudatoolkit=11.3 -c pytorch
```

### Preparing Datasets
Optionally install gdown python package for easy download of the datasets from google drive.
```
pip install --user gdown
```

#### [COCO](https://cocodataset.org/#home)
See download instructions: [https://cocodataset.org/#download]()

To setup the COCOPersons subset for use with our framework:

```
osr_prep_cocopersons --dataset_dir ${DATASET_DIR}/coco
```

To crop COCOPersons person images for use with SOLIDER:

```
osr_crop_cocopersons --dataset_dir ${DATASET_DIR}/coco
```

#### [PRW](https://github.com/liangzheng06/PRW-baseline)
```
cd $DATASET_DIR
gdown https://drive.google.com/uc?id=0B6tjyrV1YrHeYnlhNnhEYTh5MUU
unzip PRW-v16.04.20.zip -d prw
osr_prep_prw --dataset_dir ${DATASET_DIR}/prw
```

#### [CUHK-SYSU](https://github.com/ShuangLI59/person_search)
```
cd $DATASET_DIR 
gdown https://drive.google.com/uc?id=1z3LsFrJTUeEX3-XjSEJMOBrslxD2T5af 
tar -xzvf cuhk_sysu.tar.gz -C cuhk
osr_prep_cuhk --dataset_dir ${DATASET_DIR}/cuhk
```

## Configs

We include all configs needed to reproduce the results in the paper in the ./configs directory. A README.md is in ./configs detailing organization of the config files.

## Training

To train a model, run:

```
osr_run --trial_config=<CONFIG_PATH>
```

An example training config on CUHK-SYSU is given in:

```
./configs/baseline/baseline/cuhk_baseline.yaml
```

### Pre-training

To pre-train a model i.e. ensure its weights can be used for fine-tuning, make sure the following flags are set in the config (values are an example):

```
log_dir: '/remote_logging/pt_logging'
trial_name: 'coco.qc.cnt'
ckpt_interval: 5
```

An example pre-training config on COCOPersons is given in: 

```
./configs/baseline/pretrain/coco_qc_cnt.yaml
```

### Fine-tuning

To fine-tune from a previous run, use the following config flags (values correspond to example pretrain above):

```
pretrain_dir: '/remote_logging/pt_logging/coco.qc.cnt'
pretrain_epoch: 29
```

Note that the pretrain_epoch starts from 0, so e.g., 29 corresponds to 30 epochs of pre-training.

An example pre-training config on COCOPersons is given in: 

```
./configs/baseline/finetune/cuhk_qc_ft.yaml
```

### Resuming

To resume training after a crash or intentional halt, use:


```
 osr_run --trial_config=<CONFIG_PATH> --resume
```

The most recent checkpoint will be restored, and the model will pickup training mid-epoch, with model and optimizer states intact.

## Evaluation

To test an existing trained model, run:

```
osr_run --test_config=<CONFIG_PATH> --test
```

This will locate and load the final checkpoint for the trial by default.

## Comparison Checkpoints

We compare against existing model backbone weights from SOLIDER. The repo and checkpoint path used are linked below.

- [SOLIDER](https://github.com/tinyvision/SOLIDER)
    - [Swin-B](https://drive.google.com/file/d/1uh7tO34tMf73MJfFqyFEGx42UBktTbZU/view?usp=share_link_link)

## Code Attribution

We would like to acknowledge the authors of the following repos for code which was critical in production of this repo:

- [NAE](https://github.com/dichen-cd/NAE4PS)
- [SeqNet](https://github.com/serend1p1ty/SeqNet)
- [GFN](https://github.com/LukeJaffe/GFN)
- [SOLIDER](https://github.com/tinyvision/SOLIDER)
- [torchvision](https://github.com/pytorch/vision)
- [albumentations](https://github.com/albumentations-team/albumentations)
- [pytorch\_metric\_learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
- [timm](https://github.com/huggingface/pytorch-image-models)

## Citation

```
@INPROCEEDINGS{Jaffe_2025_WACV,
    author={Jaffe, Lucas and Zakhor, Avideh},
    booktitle={2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}, 
    title={Swap Path Network for Robust Person Search Pre-training}, 
    year={2025},
    pages={9291-9301},
    doi={10.1109/WACV61041.2025.00900}
}
```

## License

This code is distributed under the terms of the MIT license.

LLNL-CODE-2005623
