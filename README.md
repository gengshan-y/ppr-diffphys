# diffphys

This repo contains the differentiable physics simulation module in "PPR: Physically Plausible Reconstruction from Monocular Videos". 
It performs motion imitation given a target trajectory by optimizing control reference, PD gains, body mass, global se3, and initial velocity. For instructions on combining differentiable rendering with physics, please take a look at [PPR](https://github.com/gengshan-y/ppr).

**[[Project page]](https://gengshan-y.github.io/ppr/)**

## Installation

### 
Create a clean conda environment (skip if you have installed the [lab4d environment](https://lab4d-org.github.io/lab4d/get_started/))
```
mamba create -n ppr-diffphys python=3.9
```

Install [pytorch](https://pytorch.org/get-started/locally/). Replace `mamba` with `conda` if mamba is not installed
```
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install cudatoolkit with a version matching pytorch. Skip if cudatoolkit is previously installed.
```
mamba install -c conda-forge cudatoolkit==11.7
```

Then install dependencies:
```
pip install -r requirements.txt
pip install urdfpy==0.0.22 --no-deps
```
Prepend `CUDA_HOME=/path-to-cuda-root/` if `CUDA_HOME` is not found.

## Motion Imitation on Mocap Data

To get results on Mocap data derived from [motion_imitation](https://github.com/erwincoumans/motion_imitation), execute
```
bash run.sh
```
The results will be stored in the following directory: `logdir/mi-xx-0/`.

Visualization at 0 iteration (left to right: target, simulated, control reference)

https://github.com/gengshan-y/ppr-diffphys/assets/13134872/3f64c55b-0e29-4f92-9a10-207ae983beed

Visualization at 100 iteration (left to right: target, simulated, control reference)

https://github.com/gengshan-y/ppr-diffphys/assets/13134872/23dc6199-e9a5-4aa4-9926-5f535f8acd94

To generate additional visualizations over iterations, execute:
```
python render_intermediate.py --testdir logdir/mi-pace-0/ --data_class sim
```
https://github.com/gengshan-y/ppr-diffphys/assets/13134872/c492bc79-f8a4-4d3c-b722-4e8e2690202d




## DiffRen+DiffSim
Implemented at [lab4d@ppr](https://github.com/lab4d-org/lab4d/tree/ppr). Please see [PPR](https://github.com/gengshan-y/ppr) for instructions.

## Citation

If you find this repository useful for your research, please cite the following work.
```
@inproceedings{yang2023ppr,
	title={Physically Plausible Reconstruction from Monocular Videos},
	author={Yang, Gengshan
	and Yang, Shuo
	and Zhang, John Z.
	and Manchester, Zachary
	and Ramanan, Deva},
	booktitle = {ICCV},
	year={2023},
}
```

## Acknowledgement
- The differentiable physics simulation (e.g., kinematics, dynamics, contact) uses [Warp](https://github.com/NVIDIA/warp) internally. 
- The laikago robot URDF file is taken from [TDS](https://github.com/erwincoumans/tiny-differentiable-simulator), the quadruped URDF file is converted and modified from [Mode-Adaptive Neural Networks for Quadruped Motion Control](https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_2018#mode-adaptive-neural-networks-for-quadruped-motion-control), and the human URDF file is converted from [Mujoco](https://mujoco.org/).
