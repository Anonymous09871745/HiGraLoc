"""
三分支独立模型模块

该模块包含三个独立的分支模型:
- GlobalBranch: 全局特征分支
- ObjectBranch: 对象级特征分支
- RelationBranch: 关系级特征分支

每个分支独立训练、独立保存权重。
评估时将三个分支的相似度相加得到最终结果。
"""

from .global_branch import GlobalBranch
from .object_branch import ObjectBranch
from .relation_branch import RelationBranch

__all__ = [
    'GlobalBranch',
    'ObjectBranch',
    'RelationBranch',
]
