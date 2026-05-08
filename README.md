# HiGraLoc

**HiGraLoc: Hierarchical Graph-Aware Text-to-Point-Cloud Localization**

This repository contains the anonymous code release for **HiGraLoc**, a text-to-point-cloud localization framework designed for reproducible research on **KITTI360Pose**.

Given a natural-language query, HiGraLoc retrieves the most relevant 3D point-cloud cell and optionally refines the final position with a fine localization module. The codebase is organized to help the community reproduce the main training, inference, and evaluation results described in the paper.

## Highlights

- Coarse text-to-cell retrieval baseline.
- Three-branch retrieval framework with **global**, **object**, and **relation** branches.
- Weighted branch fusion for improved retrieval.
- Coarse-to-fine localization evaluation.
- Weight search and visualization utilities for analysis.

## Anonymous Release Notice

This repository is prepared for anonymous public review/reproduction.

- No author names, affiliations, emails, homepages, or personal identifiers are included in this README.
- Large local artifacts are intentionally excluded, including datasets, checkpoints, logs, and generated results.
- Any paths shown below use repository-relative placeholders so they can be reused on different machines.

## Table of Contents

- [1. Repository Structure](#1-repository-structure)
- [2. Environment Setup](#2-environment-setup)
- [3. Dataset Preparation](#3-dataset-preparation)
- [4. Checkpoints and External Models](#4-checkpoints-and-external-models)
- [5. Training](#5-training)
- [6. Evaluation and Inference](#6-evaluation-and-inference)
- [7. Weight Search](#7-weight-search)
- [8. Visualization](#8-visualization)
- [9. Expected Outputs](#9-expected-outputs)
- [10. Reproducibility Tips](#10-reproducibility-tips)
- [11. Troubleshooting](#11-troubleshooting)
- [12. Citation](#12-citation)
- [13. Acknowledgements](#13-acknowledgements)

## 1. Repository Structure

```text
HiGraLoc/
├── README.md
├── requirements.txt
├── branch_training_commands.md
├── dataloading/                      # KITTI360Pose dataloaders
│   └── kitti360pose/
├── datapreparation/                  # dataset preparation utilities
│   └── kitti360pose/
├── evaluation/                       # evaluation and visualization pipelines
│   ├── coarse.py
│   ├── pipeline.py
│   ├── pipeline_separate.py
│   ├── pipeline_three_branch_with_fine.py
│   ├── pipeline_weight_search.py
│   └── utils.py
├── models/                           # model definitions
│   ├── branches/                     # global / object / relation branches
│   ├── global_level_innovation/
│   ├── pointcloud/
│   ├── cell_retrieval.py
│   ├── cross_matcher.py
│   ├── language_encoder.py
│   └── object_encoder.py
├── scripts/                          # benchmark and helper scripts
├── training/                         # training entry points
│   ├── coarse.py
│   ├── fine.py
│   ├── train_branches.py
│   ├── losses.py
│   └── utils.py
└── docs/                             # notes and supplementary docs
```

The following directories are expected locally but are not included in the anonymous repository:

```text
HiGraLoc/
├── data/
├── checkpoints/
├── results/
└── weight_search_results/
```

## 2. Environment Setup

### 2.1 Clone the repository

```bash
git clone <ANONYMOUS_REPOSITORY_URL>
cd HiGraLoc
```

### 2.2 Create the environment

The codebase was developed with Python 3.10.

```bash
conda create -n higraloc python=3.10 -y
conda activate higraloc
```

### 2.3 Install PyTorch

One tested setup uses **PyTorch 1.11** with **CUDA 11.3**:

```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
```

If your environment differs, install a compatible PyTorch version first, then install the remaining dependencies below.

### 2.4 Install dependencies

```bash
pip install -r requirements.txt
```

This project depends on `torch_geometric` and related packages. If installation fails, make sure the versions of the following packages match your local PyTorch/CUDA environment:

```text
torch-scatter
torch-sparse
torch-cluster
torch-spline-conv
torch_geometric
```

### 2.5 Verify basic command entry points

Before training or evaluation, it is useful to check that the main modules can be imported correctly:

```bash
python -m training.coarse --help
python -m training.fine --help
python -m training.train_branches --help
python -m evaluation.pipeline_separate --help
python -m evaluation.pipeline_three_branch_with_fine --help
```

## 3. Dataset Preparation

HiGraLoc uses the public **KITTI360Pose** benchmark.

### 3.1 Download the dataset

Please download KITTI360Pose from the official release page used by the Text2Pos benchmark:

- KITTI360Pose download: <https://cvg.cit.tum.de/webshare/g/text2pose/>
- Text2Pos paper: <https://arxiv.org/abs/2203.15125>

### 3.2 Organize the dataset

Place the processed data under the following directory:

```text
HiGraLoc/data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/
```

Expected structure:

```text
data/
└── KITTI360Pose/
    └── k360_30-10_scG_pd10_pc4_spY_all/
        ├── cells/
        ├── direction/
        ├── poses/
        ├── street_centers/
        └── visloc/
```

All commands in this README assume:

```bash
--base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/
```

### 3.3 Optional preprocessing scripts

If you need to inspect or regenerate parts of the processed data, see:

```text
datapreparation/kitti360pose/prepare.py
datapreparation/kitti360pose/prepare_images.py
datapreparation/kitti360pose/select.py
datapreparation/kitti360pose/utils.py
```

For most reproduction settings, directly using the released processed KITTI360Pose files is sufficient.

## 4. Checkpoints and External Models

This anonymous repository does **not** include large model files.

### 4.1 Text backbone

The training and evaluation commands use:

```bash
--hungging_model t5-large
```

You can either:

- place a local `t5-large/` folder under the repository root, or
- replace `t5-large` with your own local Hugging Face model path.

### 4.2 Point-cloud backbone and task checkpoints

Create a local `checkpoints/` directory with a layout similar to the following:

```text
checkpoints/
├── global/
│   └── global_epoch25_acc0.8036.pth
├── object/
│   └── object_epoch23_acc0.8720.pth
├── relation/
│   └── relation_epoch17_acc0.8585.pth
├── k360_30-10_scG_pd10_pc4_spY_all/
│   └── path_to_fine/
│       └── fine_contN_epoch26_offset0.093_lr0.0003_obj-6-16_ecl0_eco0_p256_npa1_f-class-color-position-num.pth
└── pointnet_acc0.86_lr1_p256.pth
```

These filenames are examples matching the commands below. You may replace them with your own trained checkpoints if needed.

## 5. Training

Adjust `--batch_size`, CPU workers, and CUDA settings to match your hardware.

### 5.1 Train the coarse retrieval baseline

```bash
python -m training.coarse \
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
  --folder_name exp_coarse
```

### 5.2 Train the fine localization model

```bash
python -m training.fine \
  --batch_size 32 \
  --fine_embed_dim 128 \
  --shuffle \
  --base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
  --use_features class color position num \
  --no_pc_augment \
  --fixed_embedding \
  --epochs 32 \
  --learning_rate 0.0003 \
  --hungging_model t5-large \
  --regressor_cell all \
  --pmc_prob 0.5 \
  --folder_name exp_fine
```

### 5.3 Train the three HiGraLoc branches separately

The three-branch setup is trained with `training.train_branches`.

#### Global branch

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

#### Object branch

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

#### Relation branch

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

### 5.4 Train all branches sequentially

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

### 5.5 Where training outputs are saved

Typical outputs include:

```text
checkpoints/global/
checkpoints/object/
checkpoints/relation/
checkpoints/<experiment_name>/
```

Training logs may also be written under `checkpoints/<branch>/`.

## 6. Evaluation and Inference

### 6.1 Evaluate the coarse retrieval baseline

Validation set:

```bash
python -m evaluation.coarse \
  --base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
  --use_features class color position num \
  --no_pc_augment \
  --hungging_model t5-large \
  --fixed_embedding \
  --path_coarse ./checkpoints/exp_coarse/COARSE_MODEL_NAME.pth
```

Test set:

```bash
python -m evaluation.coarse \
  --base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
  --use_features class color position num \
  --use_test_set \
  --no_pc_augment \
  --hungging_model t5-large \
  --fixed_embedding \
  --path_coarse ./checkpoints/exp_coarse/COARSE_MODEL_NAME.pth
```

### 6.2 Evaluate the original coarse-to-fine pipeline

```bash
python -m evaluation.pipeline \
  --base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
  --use_features class color position num \
  --use_test_set \
  --no_pc_augment \
  --no_pc_augment_fine \
  --hungging_model t5-large \
  --fixed_embedding \
  --path_coarse ./checkpoints/exp_coarse/COARSE_MODEL_NAME.pth \
  --path_fine ./checkpoints/exp_fine/FINE_MODEL_NAME.pth
```

### 6.3 Evaluate three-branch retrieval

Validation set:

```bash
python -m evaluation.pipeline_separate \
  --global_checkpoint ./checkpoints/global/global_epoch21_acc0.7841.pth \
  --object_checkpoint ./checkpoints/object/object_epoch15_acc0.8754.pth \
  --relation_checkpoint ./checkpoints/relation/relation_epoch14_acc0.8701.pth \
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
```

Test set:

```bash
python -m evaluation.pipeline_separate \
  --global_checkpoint ./checkpoints/global/global_epoch21_acc0.7841.pth \
  --object_checkpoint ./checkpoints/object/object_epoch15_acc0.8754.pth \
  --relation_checkpoint ./checkpoints/relation/relation_epoch14_acc0.8701.pth \
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
```

### 6.4 Evaluate weighted three-branch fusion

```bash
python -m evaluation.pipeline_separate \
  --global_checkpoint ./checkpoints/global/global_epoch21_acc0.7841.pth \
  --object_checkpoint ./checkpoints/object/object_epoch15_acc0.8754.pth \
  --relation_checkpoint ./checkpoints/relation/relation_epoch14_acc0.8701.pth \
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
```

### 6.5 Final HiGraLoc three-branch + fine evaluation

This is the final coarse-to-fine evaluation command corresponding to the HiGraLoc pipeline. Replace checkpoint filenames with the ones available in your environment.

```bash
python -m evaluation.pipeline_three_branch_with_fine \
  --global_checkpoint ./checkpoints/global/global_epoch25_acc0.8036.pth \
  --object_checkpoint ./checkpoints/object/object_epoch23_acc0.8720.pth \
  --relation_checkpoint ./checkpoints/relation/relation_epoch17_acc0.8585.pth \
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
```

## 7. Weight Search

To search for branch fusion weights:

```bash
python -m evaluation.pipeline_weight_search \
  --global_checkpoint ./checkpoints/global/global_epoch21_acc0.7841.pth \
  --object_checkpoint ./checkpoints/object/object_epoch15_acc0.8754.pth \
  --relation_checkpoint ./checkpoints/relation/relation_epoch14_acc0.8701.pth \
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
```

This script evaluates multiple branch-weight combinations and saves search results locally.

## 8. Visualization

Useful visualization utilities are provided under `evaluation/`:

```text
evaluation/visualize_three_branch_retrieval.py
evaluation/render_query_bev.py
evaluation/render_dataset_query_bev.py
evaluation/render_paper_retrieval_figure.py
evaluation/render_paper_retrieval_figure_grid.py
evaluation/render_paper_retrieval_figure_grid_top123_correct.py
```

Related tests/utilities include:

```text
evaluation/test_visualize_three_branch_retrieval.py
evaluation/test_render_query_bev.py
evaluation/test_render_dataset_query_bev.py
evaluation/test_render_paper_retrieval_figure.py
```

Generated visual outputs are typically saved under `results/`.

## 9. Expected Outputs

After running experiments, you may obtain files such as:

```text
checkpoints/                     # model weights and logs
results/                         # visualizations and JSON outputs
weight_search_results/           # weight search CSV / JSON outputs
data/KITTI360Pose/evaluation_logs/
```

These outputs are intentionally excluded from version control in the anonymous repository.

## 10. Reproducibility Tips

- Always run commands from the repository root.
- Keep dataset layout exactly consistent with the paths shown above.
- Use the same text backbone throughout training and evaluation.
- Make sure the PointNet checkpoint matches the branch checkpoints you are evaluating.
- When reproducing final results, use the same branch weights as the final command.
- If you retrain models from scratch, keep the experiment folder names organized so later evaluation commands are easy to update.

## 11. Troubleshooting

### 11.1 `ModuleNotFoundError`

Make sure you are in the repository root:

```bash
cd HiGraLoc
```

Then run the code with module mode:

```bash
python -m evaluation.pipeline_separate --help
```

### 11.2 CUDA / PyTorch Geometric installation issues

Ensure that your PyTorch, CUDA, and PyG package versions match each other. If needed, reinstall the PyG-related packages for your exact local environment.

### 11.3 Text encoder path issues

If `--hungging_model t5-large` cannot be resolved locally, replace it with a local path, for example:

```bash
--hungging_model ./t5-large
```

### 11.4 Missing checkpoint files

This repository does not include `.pth` files. Please place all required checkpoints under `checkpoints/` and update the command-line paths accordingly.

### 11.5 Missing dataset files

If loading fails, verify that the dataset was downloaded completely and that `cells/`, `direction/`, `poses/`, `street_centers/`, and `visloc/` are all present under the expected `base_path`.

## 12. Citation

If you find this repository useful, please cite the corresponding paper after the final bibliographic information is available.

A placeholder format is shown below:

```bibtex
@article{higraloc_anonymous,
  title={HiGraLoc: Hierarchical Graph-Aware Text-to-Point-Cloud Localization},
  author={Anonymous},
  journal={under review},
  year={2026}
}
```

## 13. Acknowledgements

This repository follows the public text-to-point-cloud localization benchmark setting and uses the public KITTI360Pose data protocol. We acknowledge the open-source ecosystems around PyTorch, PyTorch Geometric, and Hugging Face Transformers, as well as prior public resources that made this reproduction-oriented release possible.
