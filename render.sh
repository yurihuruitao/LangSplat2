casename="ramen_try"
for level in 1 2 3
do
    # render rgb
    python render.py -m output/${casename}_${level}
    # render language features
    python render.py -m output/${casename}_${level} --include_feature
    # e.g. python render.py -m output/sofa_3 --include_feature
done