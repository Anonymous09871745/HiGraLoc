# Branch Training Commands

## Global Branch Only

```bash
python -m training.train_branches \
  --batch_size 64 \
  --coarse_embed_dim 256 \
  --shuffle \
  --base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
  --use_features class color position num \
  --no_pc_augment \
  --fixed_embedding \
  --epochs 32 \
  --learning_rate 0.0001 \
  --lr_scheduler step \
  --lr_step 5 \
  --lr_gamma 0.5 \
  --temperature 0.05 \
  --ranking_loss CCL \
  --num_of_hidden_layer 3 \
  --alpha 2 \
  --hungging_model t5-large \
  --folder_name exp_coarse_staged \
  --branch global \
  --epochs_per_branch 32 \
  --cpus 4
```

## Object Branch Only

```bash
python -m training.train_branches \
  --batch_size 64 \
  --coarse_embed_dim 256 \
  --shuffle \
  --base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
  --use_features class color position num \
  --no_pc_augment \
  --fixed_embedding \
  --epochs 32 \
  --learning_rate 0.0001 \
  --lr_scheduler step \
  --lr_step 5 \
  --lr_gamma 0.5 \
  --temperature 0.05 \
  --ranking_loss CCL \
  --num_of_hidden_layer 3 \
  --alpha 2 \
  --hungging_model t5-large \
  --folder_name exp_coarse_staged \
  --branch object \
  --epochs_per_branch 32 \
  --cpus 4
```

## Relation Branch Only

```bash
python -m training.train_branches \
  --batch_size 64 \
  --coarse_embed_dim 256 \
  --shuffle \
  --base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
  --use_features class color position num \
  --no_pc_augment \
  --fixed_embedding \
  --epochs 32 \
  --learning_rate 0.0001 \
  --lr_scheduler step \
  --lr_step 5 \
  --lr_gamma 0.5 \
  --temperature 0.05 \
  --ranking_loss CCL \
  --num_of_hidden_layer 3 \
  --alpha 2 \
  --hungging_model t5-large \
  --folder_name exp_coarse_staged \
  --branch relation \
  --epochs_per_branch 32 \
  --cpus 4
```

## Train All Branches Sequentially

```bash
python -m training.train_branches \
  --batch_size 64 \
  --coarse_embed_dim 256 \
  --shuffle \
  --base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
  --use_features class color position num \
  --no_pc_augment \
  --fixed_embedding \
  --epochs 32 \
  --learning_rate 0.0001 \
  --lr_scheduler step \
  --lr_step 5 \
  --lr_gamma 0.5 \
  --temperature 0.05 \
  --ranking_loss CCL \
  --num_of_hidden_layer 3 \
  --alpha 2 \
  --hungging_model t5-large \
  --folder_name exp_coarse_staged \
  --branch all \
  --epochs_per_branch 32 \
  --cpus 4
```

## Output Files

- **Checkpoints**: `./checkpoints/{branch}/`
  - `global_best.pth`, `object_best.pth`, `relation_best.pth` (best models)
  - `global_epoch{epoch}_acc{acc}.pth`, etc. (per-epoch models)

- **Training Logs**: `./checkpoints/{branch}/{branch}_training_log.txt`


cd /root/autodl-tmp/MNCL
## Inference stage - val set
python evaluation/pipeline_separate.py \
    --global_checkpoint checkpoints/global/global_epoch21_acc0.7841.pth \
    --object_checkpoint checkpoints/object/object_epoch15_acc0.8754.pth \
    --relation_checkpoint checkpoints/relation/relation_epoch14_acc0.8701.pth \
    --base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
    --batch_size 64 \
    --coarse_embed_dim 256 \
    --no_pc_augment \
    --fixed_embedding \
    --use_features class color position num \
    --hungging_model t5-large \
    --pointnet_path ./checkpoints/pointnet_acc0.86_lr1_p256.pth \
    --num_mentioned 6 \
    --object_size 28 \
    --inter_module_num_heads 4 \
    --inter_module_num_layers 1 \
    --intra_module_num_heads 4 \
    --intra_module_num_layers 1 \
    --num_of_hidden_layer 3 \
    --alpha 2 \
    --top_k 1 3 5 10
    
 ## Inference stage - test set
python -m evaluation.pipeline_separate \
    --global_checkpoint checkpoints/global/global_epoch21_acc0.7841.pth \
    --object_checkpoint checkpoints/object/object_epoch15_acc0.8754.pth \
    --relation_checkpoint checkpoints/relation/relation_epoch14_acc0.8701.pth \
    --base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
    --batch_size 64 \
    --coarse_embed_dim 256 \
    --no_pc_augment \
    --fixed_embedding \
    --use_features class color position num \
    --hungging_model t5-large \
    --pointnet_path ./checkpoints/pointnet_acc0.86_lr1_p256.pth \
    --num_mentioned 6 \
    --object_size 28 \
    --inter_module_num_heads 4 \
    --inter_module_num_layers 1 \
    --intra_module_num_heads 4 \
    --intra_module_num_layers 1 \
    --num_of_hidden_layer 3 \
    --alpha 2 \
    --top_k 1 3 5 10 \
    --use_test_set
    
    
参数搜索： 
1. evaluation/pipeline_weight_search.py - 权重搜索脚本
功能：

以 0.05 步长搜索 9261 种权重组合
支持 --use_test_set 在测试集上评估
以 Hit@1 为主要排序指标
输出 Top-20 最佳权重组合
保存 CSV 和 JSON 结果文件
运行命令：

python -m evaluation.pipeline_weight_search \
    --global_checkpoint checkpoints/global/global_epoch21_acc0.7841.pth \
    --object_checkpoint checkpoints/object/object_epoch15_acc0.8754.pth \
    --relation_checkpoint checkpoints/relation/relation_epoch14_acc0.8701.pth \
    --base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
    --batch_size 64 \
    --coarse_embed_dim 256 \
    --no_pc_augment \
    --fixed_embedding \
    --use_features class color position num \
    --hungging_model t5-large \
    --pointnet_path ./checkpoints/pointnet_acc0.86_lr1_p256.pth \
    --num_mentioned 6 \
    --object_size 28 \
    --inter_module_num_heads 4 \
    --inter_module_num_layers 1 \
    --intra_module_num_heads 4 \
    --intra_module_num_layers 1 \
    --num_of_hidden_layer 3 \
    --alpha 2 \
    --weight_step 0.05 \
    --use_test_set \
    --output_dir ./weight_search_results
    

 更新了 evaluation/pipeline_separate.py
新增功能：

添加了 --weight_global、--weight_object、--weight_relation 参数
支持自定义权重组合进行评估
运行命令示例：

python -m evaluation.pipeline_separate \
    --global_checkpoint checkpoints/global/global_epoch21_acc0.7841.pth \
    --object_checkpoint checkpoints/object/object_epoch15_acc0.8754.pth \
    --relation_checkpoint checkpoints/relation/relation_epoch14_acc0.8701.pth \
    --base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
    --batch_size 64 \
    --coarse_embed_dim 256 \
    --no_pc_augment \
    --fixed_embedding \
    --use_features class color position num \
    --hungging_model t5-large \
    --pointnet_path ./checkpoints/pointnet_acc0.86_lr1_p256.pth \
    --num_mentioned 6 \
    --object_size 28 \
    --inter_module_num_heads 4 \
    --inter_module_num_layers 1 \
    --intra_module_num_heads 4 \
    --intra_module_num_layers 1 \
    --num_of_hidden_layer 3 \
    --alpha 2 \
    --weight_global 0.30 \
    --weight_object 0.40 \
    --weight_relation 0.30 \
    --top_k 1 3 5 10 \
    --use_test_set
    
添加了在fine阶段进行评估的命令：
cd /root/autodl-tmp/MNCL

python -m evaluation.pipeline_three_branch_with_fine \
    --global_checkpoint ./checkpoints/global/global_epoch25_acc0.8036.pth \
    --object_checkpoint /root/autodl-tmp/object_epoch23_acc0.8720.pth \
    --relation_checkpoint /root/autodl-tmp/relation_epoch17_acc0.8585.pth \
    --path_fine ./checkpoints/k360_30-10_scG_pd10_pc4_spY_all/path_to_fine/fine_contN_epoch26_offset0.093_lr0.0003_obj-6-16_ecl0_eco0_p256_npa1_f-class-color-position-num.pth \
    --weight_global 0.3 \
    --weight_object 1.0 \
    --weight_relation 0.8 \
    --base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
    --use_test_set \
    --batch_size 32 \
    --coarse_embed_dim 256 \
    --no_pc_augment \
    --fixed_embedding \
    --use_features class color position num \
    --hungging_model t5-large \
    --pointnet_path ./checkpoints/pointnet_acc0.86_lr1_p256.pth \
    --num_mentioned 6 \
    --object_size 28 \
    --inter_module_num_heads 4 \
    --inter_module_num_layers 1 \
    --intra_module_num_heads 4 \
    --intra_module_num_layers 1 \
    --num_of_hidden_layer 3 \
    --alpha 2