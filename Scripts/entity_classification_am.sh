export PYTHONPATH=./:../:$PYTHONPATH
cd $(dirname $0)/../
save_dir=../trained_models/ec_am

python3 experiment.py \
        --dataset=am \
        --model=rgcn \
        --task=ec \
        --num_epoch=50 \
        --num_edge_types=266 \
        --norm_type=relation-degree \
        --l2param=5e-4 \
        --num_bases=40 \
        --hidden_dim=10 \
        --out_dim=11 \
        --lr=1e-2 \
        --model_output_path=${save_dir} \
        --debug