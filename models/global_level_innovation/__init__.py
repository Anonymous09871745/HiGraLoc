"""
Global Branch Level Innovation Models

该模块包含 global 分支训练所需的所有模型文件，
核心创新点：谱图流形变换 (Spectral Manifold Transform) 替代 FFT。

使用方法:
    from models.global_level_innovation import GlobalBranch
"""

from .global_branch import GlobalBranch
from .bir_lstm import ConvBlock, SpectralManifoldBlock
from .object_encoder import ObjectEncoder
from .language_encoder import LanguageEncoder

__all__ = [
    'GlobalBranch',
    'ConvBlock',
    'SpectralManifoldBlock',
    'ObjectEncoder',
    'LanguageEncoder',
]
