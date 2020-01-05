[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://github.com/Ha0Tang/C2GAN/blob/master/LICENSE.md)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
![Packagist](https://img.shields.io/badge/Pytorch-0.4.1-red.svg)
![Last Commit](https://img.shields.io/github/last-commit/Ha0Tang/C2GAN)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)]((https://github.com/Ha0Tang/C2GAN/graphs/commit-activity))
![Contributing](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)
![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)

# Cycle In Cycle Generative Adversarial Networks for Keypoint-Guided Image Generation

## C2GAN Framework

![Framework](./imgs/c2gan_framework.jpg)

### [Paper](https://arxiv.org/abs/1908.00999) | [Project](http://disi.unitn.it/~hao.tang/project/C2GAN.html)
Cycle In Cycle Generative Adversarial Networks for Keypoint-Guided Image Generation.<br>
[Hao Tang](http://disi.unitn.it/~hao.tang/)<sup>1</sup>, [Dan Xu](http://www.robots.ox.ac.uk/~danxu/)<sup>2</sup>, [Gaowen Liu](https://dblp.uni-trier.de/pers/hd/l/Liu:Gaowen)<sup>3</sup>, [Wei Wang](https://weiwangtrento.github.io/)<sup>4</sup>, [Nicu Sebe](https://scholar.google.com/citations?user=stFCYOAAAAAJ&hl=en)<sup>1</sup> and [Yan Yan](https://scholar.google.com/citations?user=zhi-j1wAAAAJ&hl=en)<sup>3</sup> </br>
<sup>1</sup>University of Trento, <sup>2</sup>University of Oxford, <sup>3</sup>Texas State University, <sup>4</sup>EPFL </br>
The repository offers the official implementation of our paper in PyTorch.

### [License](./LICENSE.md)

Copyright (C) 2019 University of Trento, Italy.

All rights reserved.
Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**)

The code is released for academic research use only. For commercial use, please contact [hao.tang@unitn.it](hao.tang@unitn.it).

## Installation

Clone this repo.
```bash
git clone https://github.com/Ha0Tang/C2GAN
cd C2GAN/
```

This code requires PyTorch 0.4.1+ and python 3.6.9+. Please install dependencies by
```bash
pip install -r requirements.txt (for pip users)
```
or 

```bash
./scripts/conda_deps.sh (for Conda users)
```

To reproduce the results reported in the paper, you would need an NVIDIA TITAN Xp GPUs.

## Dataset Preparation
For your convenience we provide download scripts:
```
bash ./datasets/download_c2gan_dataset.sh RaFD_image_landmark
```
- `RaFD_image_landmark`: 3.0 GB
Prepare the datasets like in [this folder](./datasets/Radboud) aftre the download has finished. Please cite their paper if you use the data.

## C2GAN Training/Testing
- Download a dataset using the previous script (e.g., Radboud).
- To view training results and loss plots, run `python -m visdom.server` and click the URL [http://localhost:8097](http://localhost:8097).
- Train a model:
```
bash ./train_c2gan.sh
```
- To see more intermediate results, check out `./checkpoints/Radboud_c2gan/web/index.html`.
- Test the model:
```
bash ./test_c2gan.sh
```
- The test results will be saved to a html file here: `./results/Radboud_c2gan/latest_test/index.html`.

## Generating Images Using Pretrained Model
- You need download a pretrained model (e.g., Radboud) with the following script:
```
bash ./scripts/download_c2gan_model.sh Radboud
```
- The pretrained model is saved at `./checkpoints/{name}_pretrained/latest_net_G.pth`. 
- Then generate the result using
```
python test.py --dataroot ./datasets/Radboud --name Radboud_pretrained --model c2gan --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned --norm batch --gpu_ids 0 --batch 16;
```
The results will be saved at `./results/`. Use `--results_dir {directory_path_to_save_result}` to specify the results directory.

- For your own experiments, you might want to specify --netG, --norm, --no_dropout to match the generator architecture of the trained model.

## Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{tang2019cycleincycle,
  title={Cycle In Cycle Generative Adversarial Networks for Keypoint-Guided Image Generation},
  author={Tang, Hao and Xu, Dan and Liu, Gaowen and Wang, Wei and Sebe, Nicu and Yan, Yan},
  booktitle={ACM MM},
  year={2019}
}
```

## Acknowledgments
This source code is inspired by [Pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [GestureGAN](https://github.com/Ha0Tang/GestureGAN). 

## Contributions
If you have any questions/comments/bug reports, feel free to open a github issue or pull a request or e-mail to the author Hao Tang ([hao.tang@unitn.it](hao.tang@unitn.it)).





