# vidyn
# install tds
issue fix here: https://github.com/erwincoumans/tiny-differentiable-simulator/issues/63
export PYTHONPATH="${PYTHONPATH}:~/code/tiny-differentiable-simulator/build/python"

# cmds
```
python diffmpm.py
bash scripts/save_fgbg.sh
```

TDS
```
CUDA_VISIBLE_DEVICES=1 python main.py --logname shiba-haru-1002-tds --pre_skel laikago --seqname shiba-haru-1013
```

Taichi
```
CUDA_VISIBLE_DEVICES=1 python main.py --logname shiba-haru-1002-tds --pre_skel laikago --seqname shiba-haru-1001 --backend taichi
```

Warp
```
python main.py --logname shiba-haru-1002-tds --pre_skel laikago --seqname mi-sidesteps --backend warp --total_iters 1000 --learning_rate 1e-3
```

# changes
(0) use larger skeleton to avoid blowing up
(1) new urdf parser to deal with sperical joints => no reduncant links => 2x time steps
(2) smaller feet mass / normal gravity / use ke=20 to avoid bouncing artefact / limit_ke, shape_mu
