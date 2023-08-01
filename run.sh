set -x
export DISPLAY=:99.0
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
sleep 3
set +x
exec "$@"

#ft
#CUDA_VISIBLE_DEVICES=1 python main.py --logname      mi-pace-mod-vel-toe-ft-1 --pre_skel laikago --seqname mi-pace --backend warp --total_iters 1000 --model_path logdir/mi-pace-svel-toe-1/params_5000.pth

# rollout
#CUDA_VISIBLE_DEVICES=1 python main.py --logname      mi-pace-mod-vel-toe-ft-1 --pre_skel laikago --rollout --seqname mi-pace --backend warp --total_iters 1000 --model_path logdir/mi-pace-svel-toe-1/params_5000.pth
#CUDA_VISIBLE_DEVICES=1 python main.py --logname      mi-trot-mod-vel-toe-ft-1 --pre_skel laikago --rollout --seqname mi-trot --backend warp --total_iters 1000 --model_path logdir/mi-trot-svel-1/params_5000.pth
#CUDA_VISIBLE_DEVICES=0 python main.py --logname      mi-spin-match-hardth-ft-1 --pre_skel laikago --rollout --seqname mi-spin --backend warp --total_iters 1000 --model_path logdir/mi-spin-svel-toe-1/params_1000.pth
#CUDA_VISIBLE_DEVICES=0 python main.py --logname mi-sidesteps-match-hardth-ft-1 --pre_skel laikago --rollout --seqname mi-sidesteps --backend warp --total_iters 5000 --model_path logdir/mi-sidesteps-svel-1/params_10500.pth
#CUDA_VISIBLE_DEVICES=0 python main.py --logname mi-sidesteps-match-hardth-ft-1 --pre_skel laikago --rollout --seqname mi-sidesteps --backend warp --total_iters 5000
#CUDA_VISIBLE_DEVICES=0 python main.py --logname      mi-turn-match-hardth-ft-1 --pre_skel laikago --rollout --seqname mi-turn --backend warp --total_iters 1000 --model_path logdir/mi-turn-svel-1/params_25000.pth

### training
CUDA_VISIBLE_DEVICES=0 python main.py --logname spin --pre_skel laikago --seqname mi-spin --backend warp
CUDA_VISIBLE_DEVICES=0 python main.py --logname trot --pre_skel laikago --seqname mi-trot --backend warp
CUDA_VISIBLE_DEVICES=0 python main.py --logname pace --pre_skel laikago --seqname mi-pace --backend warp
CUDA_VISIBLE_DEVICES=0 python main.py --logname sidesteps --pre_skel laikago --seqname mi-sidesteps --backend warp
CUDA_VISIBLE_DEVICES=0 python main.py --logname turn --pre_skel laikago --seqname mi-turn --backend warp

#CUDA_VISIBLE_DEVICES=0 python main.py --logname mi-trot-e10w8-1 --pre_skel laikago --seqname mi-trot --backend warp --total_iters 300
#CUDA_VISIBLE_DEVICES=0 python main.py --logname mi-pace-e100w1-dvel-1 --pre_skel laikago --seqname mi-pace --backend warp --total_iters 300
#CUDA_VISIBLE_DEVICES=0 python main.py --logname mi-spin-e100w4-6 --pre_skel laikago --seqname mi-spin --backend warp --total_iters 300
#CUDA_VISIBLE_DEVICES=0 python main.py --logname mi-turn-e100w4-1 --pre_skel laikago --seqname mi-turn --backend warp --total_iters 30000
#CUDA_VISIBLE_DEVICES=0 python main.py --logname mi-trot-e1000-wdw1-1 --pre_skel laikago --seqname mi-trot --backend warp --total_iters 300
#CUDA_VISIBLE_DEVICES=0 python main.py --logname mi-pace-e1000-wdw1-1 --pre_skel laikago --seqname mi-pace --backend warp --total_iters 300
#CUDA_VISIBLE_DEVICES=0 python main.py --logname mi-spin-e1000-wdw1-1 --pre_skel laikago --seqname mi-spin --backend warp --total_iters 300
