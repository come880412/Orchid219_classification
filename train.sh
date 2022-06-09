# convnext_xlarge
python train.py --split_type random \
             --model convnext \
             --save_model ./checkpoints \
             --save_name convnext_xlarge_orchid219 \
             --load_pretrain '' \
             --weight_decay 0.05 \
             --n_epochs 60 \
             --warmup_epochs 7 \
             --mixup 0.4 \
             --optimizer adamw \
             --lr 1e-4 \
             --scheduler linearwarmup
             --smooth_factor 0.1 \
             --weight_decay 0.05 \
             --batch_size 24

# beit_large
python train.py --train_type fine_tune \
             --n_epochs 50 \
             --warmup_epochs 6 \
             --optimizer adamw \
             --split_type random \
             --load_pretrain '' \
             --num_classes 219 \
             --model beit \
             --save_model ./checkpoints \
             --save_name beit_orchid219 \
             --smooth_factor 0.2 \
             --batch_size 1 \
             --mixup 0.0 \
             --lr 1e-4 \
             --weight_decay 0.05 \
             --gpu_id 1
             --seed 8863

# swin_large
python train.py --train_type fine_tune \
             --n_epochs 100 \
             --warmup_epochs 10 \
             --optimizer adamw \
             --split_type random \
             --num_classes 219 \
             --model beit \
             --save_model ./checkpoints \
             --save_name beit_orchid219 \
             --smooth_factor 0.1 \
             --batch_size 16 \
             --mixup 0.6 \
             --lr 1e-4 \
             --weight_decay 0.05 \
             --gpu_id 1