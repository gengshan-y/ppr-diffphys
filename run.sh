set -x
export DISPLAY=:99.0
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
sleep 3
set +x
exec "$@"

### training
rm -rf logdir/mi-*
CUDA_VISIBLE_DEVICES=0 python main.py --urdf_template laikago --seqname mi-trot --logname 0 
CUDA_VISIBLE_DEVICES=0 python main.py --urdf_template laikago --seqname mi-spin --logname 0 
CUDA_VISIBLE_DEVICES=0 python main.py --urdf_template laikago --seqname mi-pace --logname 0 
CUDA_VISIBLE_DEVICES=0 python main.py --urdf_template laikago --seqname mi-sidesteps --logname 0 
CUDA_VISIBLE_DEVICES=0 python main.py --urdf_template laikago --seqname mi-turn --logname 0 
