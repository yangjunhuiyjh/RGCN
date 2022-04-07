export PYTHONPATH=./:../:$PYTHONPATH
cd $(dirname $0)/../
save_dir=../trained_models/lp_wn18

python3 experiment.py \
        --dataset=wn18 \
        --model=rgcn \
        --task=lp \
        --num_epoch=50 \
        --num_edge_types=36 \
        --norm_type=non-relation-degree \
        --l2param=5e-4 \
        --num_bases=2 \
        --hidden_dim=200 \
        --lr=1e-2 \
        --model_output_path=${save_dir} \
        --debug