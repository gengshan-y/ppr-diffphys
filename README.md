# diffphys

Differentiable physics simulation module of PPR. 

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

Install cudatoolkit with a version matching pytorch. Skip if cudatoolkit is installed
```
mamba install -c conda-forge cudatoolkit==11.7
```

Then install dependencies:
```
pip install -r requirements.txt
pip install urdfpy==0.0.22 --no-deps
```
You might need to prepend `CUDA_HOME=/path-to-cuda-root/`.

# Demo


## Install
`pip install open3d`


```
env_name=lab4d
cd /home/gengshay/miniconda3/envs/$env_name/
cd etc/conda/
mkdir -p activate.d
echo "export LD_LIBRARY_PATH=/home/gengshay/miniconda3/envs/lab4d/lib/:$LD_LIBRARY_PATH" >> env_vars.sh
chmod +x env_vars.sh
cd ../deactivate.d/
echo "unset LD_LIBRARY_PATH" >> env_vars.sh
chmod +x env_vars.sh
```

(0) use larger skeleton to avoid blowing up
(1) new urdf parser to deal with sperical joints => no reduncant links => 2x time steps
(2) smaller feet mass / normal gravity / use ke=20 to avoid bouncing artefact / limit_ke, shape_mu
