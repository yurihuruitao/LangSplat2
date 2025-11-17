cd /home/ubuntu/langsplat/LangSplat/eval

python try.py \
  --json_folder /path/to/lerf_ovs/label/ramen \
  --feat_dir ../output \
  --ae_ckpt_dir ../autoencoder/ckpt \
  --pred_scene ramen_try_3 \
  --gt_scene ramen