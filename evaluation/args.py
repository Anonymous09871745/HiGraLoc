import argparse
from argparse import ArgumentParser
import os
import os.path as osp


def create_base_parser():
    """创建基础参数解析器，不执行解析"""
    parser = argparse.ArgumentParser(description="Evaluation Pipeline for NegLoc")

    # Paths
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="K360")
    parser.add_argument("--base_path", type=str)
    parser.add_argument("--path_coarse", type=str, help="The path to the Cell-Retrieval model")
    parser.add_argument("--path_fine", type=str, help="The path to the Hints-to-Objects matching model")

    # Options
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--threshs", type=int, nargs="+", default=[5, 10, 15])  # Possibly over-write here when it becomes a list of tuples
    parser.add_argument("--use_features", nargs="+", default=["class", "color", "position", "num"])
    parser.add_argument("--use_test_set", action="store_true", help="Run test-set instead of the validation set.")
    parser.add_argument("--no_pc_augment", action="store_true")
    parser.add_argument("--no_pc_augment_fine", action="store_true")

    # Fine
    parser.add_argument("--fine_embed_dim", type=int, default=128)
    parser.add_argument("--fine_num_decoder_heads", type=int, default=4)
    parser.add_argument("--fine_num_decoder_layers", type=int, default=2)
    parser.add_argument("--pad_size", type=int, default=16)
    parser.add_argument("--num_mentioned", type=int, default=6)
    parser.add_argument("--describe_by", type=str, default="all")

    #Object-encoder / PointNet
    parser.add_argument("--coarse_embed_dim", type=int, default=256)
    parser.add_argument("--pointnet_layers", type=int, default=3)
    parser.add_argument("--pointnet_variation", type=int, default=0)
    parser.add_argument("--pointnet_numpoints", type=int, default=256)
    parser.add_argument("--pointnet_path", type=str, default="./checkpoints/pointnet_acc0.86_lr1_p256.pth")
    parser.add_argument("--pointnet_freeze", action="store_true")
    parser.add_argument("--pointnet_features", type=int, default=2)
    parser.add_argument("--class_embed", action="store_true")
    parser.add_argument("--color_embed", action="store_true")
    parser.add_argument("--object_size", type=int, default=28)
    parser.add_argument("--object_inter_module_num_heads", type=int, default=4)
    parser.add_argument("--object_inter_module_num_layers", type=int, default=2)

    # Language Encoder
    parser.add_argument("--hungging_model", type=str, help="hugging face model")
    parser.add_argument("--fixed_embedding", action="store_true")
    parser.add_argument("--inter_module_num_heads", type=int, default=4)
    parser.add_argument("--inter_module_num_layers", type=int, default=1)
    parser.add_argument("--intra_module_num_heads", type=int, default=4)
    parser.add_argument("--intra_module_num_layers", type=int, default=1)
    parser.add_argument("--fine_intra_module_num_heads", type=int, default=4)
    parser.add_argument("--fine_intra_module_num_layers", type=int, default=1)

    parser.add_argument("--ranking_loss", type=str, default="CCL")

    # MSG
    parser.add_argument("--num_of_hidden_layer", type=int, default=3)

    # CCL_alpha
    parser.add_argument("--alpha", type=int, default=2)
    
    # fine-localization
    parser.add_argument("--coarse_path_in_fine", type=str, default=None, help="calculate Object-level Confidence in fine-localization") 

    # confidence thresh
    parser.add_argument("--confidence_thresh", type=float, default=0.15) 

    return parser


def parse_arguments():
    """保持向后兼容的解析函数"""
    parser = create_base_parser()
    args = parser.parse_args()

    assert osp.isfile(args.path_coarse)
    if args.path_fine:
        assert osp.isfile(args.path_fine)

    return args

