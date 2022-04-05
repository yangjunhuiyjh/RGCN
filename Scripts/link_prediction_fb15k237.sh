export PYTHONPATH=./:../:$PYTHONPATH
cd $(dirname $0)/../
save_dir=../trained_models/lp_fb15k237

python3 experiment.py \
        --dataset=fb15k237 \
        --model=rgcn \
        --task=lp \
        --num_epoch=2 \
        --num_edge_types=237 \
        --norm_type=non-relation-degree \
        --l2param=5e-4 \
        --num_blocks=100 \
        --hidden_dim=500 \
        --lr=1e-2 \
        --model_output_path=${save_dir} \
        --debug