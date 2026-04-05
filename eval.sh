CUDA_VISIBLE_DEVICES=0 python train.py \
  --config checkpoint/Cascad/ID3/config.yaml \
  --evaluate checkpoint/Cascad/ID3/epoch_89.bin \
  --checkpoint checkpoint/Cascad/ID3/eval
# CUDA_VISIBLE_DEVICES=2 python train.py \
#   --config checkpoint/pose3d/PoseMamba_S/config.yaml \
#   --evaluate checkpoint/pose3d/PoseMamba_S/best_epoch.bin \
#   --checkpoint eval/checkpoint
# CUDA_VISIBLE_DEVICES=2 python train.py \
#   --config checkpoint/pose3d/PoseMamba_L/config.yaml \
#   --evaluate checkpoint/pose3d/PoseMamba_L/best_epoch.bin \
#   --checkpoint eval/checkpoint