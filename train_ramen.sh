for level in 3 2 1
do
    python train.py -s /home/ubuntu/langsplat/LangSplat/datasets/lerf_ovs/ramen_try -m output/ramen_try --start_checkpoint /home/ubuntu/langsplat/LangSplat/datasets/lerf_ovs/ramen_try/output/ramen_try/chkpnt30000.pth --feature_level ${level}
    # e.g. python train.py -s /home/ubuntu/langsplat/LangSplat/datasets/lerf_ovs/ramen -m output/ramen --start_checkpoint /home/ubuntu/langsplat/LangSplat/datasets/lerf_ovs/ramen/chkpnt30000.pth --feature_level 1
done