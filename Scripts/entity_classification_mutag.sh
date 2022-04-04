export PYTHONPATH=./:../:$PYTHONPATH
cd $(dirname $0)/../
save_dir=../trained_models/ec_mutag

python3 experiment.py \
        --dataset=mutag \
        --model=rgcn \
        --task=ec \
        --num_epoch=50 \
        --num_edge_types=46 \
        --norm_type=relation-degree \
        --l2param=5e-4 \
        --num_bases=30 \
        --hidden_dim=16 \
        --out_dim=2 \
        --lr=1e-2 \
        --model_output_path=${save_dir} \
        --debug