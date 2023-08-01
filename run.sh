set -x
export DISPLAY=:99.0
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
sleep 3
set +x
exec "$@"

### training
CUDA_VISIBLE_DEVICES=0 python main.py --logname spin --urdf_template laikago --seqname mi-spin 
CUDA_VISIBLE_DEVICES=0 python main.py --logname trot --urdf_template laikago --seqname mi-trot 
CUDA_VISIBLE_DEVICES=0 python main.py --logname pace --urdf_template laikago --seqname mi-pace 
CUDA_VISIBLE_DEVICES=0 python main.py --logname sidesteps --urdf_template laikago --seqname mi-sidesteps 
CUDA_VISIBLE_DEVICES=0 python main.py --logname turn --urdf_template laikago --seqname mi-turn 
