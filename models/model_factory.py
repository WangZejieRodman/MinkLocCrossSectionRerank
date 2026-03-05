# models/model_factory.py
import torch.nn as nn

from models.minkloc import MinkLoc
from misc.utils import ModelParams
from models.layers.pooling_wrapper import PoolingWrapper
from models.minkbev import MinkBEVBackbone
from models.slice_branch import SliceSequenceBranch  # 引入新增的序列提取分支


def model_factory(model_params: ModelParams):
    """
    模型工厂（支持 BEV 和 Cross-Section Dual-Stream）
    """
    if model_params.model == 'MinkLocBEV':
        in_channels = getattr(model_params, 'in_channels', 32)
        print(f"Model Factory: Initializing MinkLocBEV...")
        print(f"  Input Channels (Z-layers): {in_channels}")
        print(f"  Feature Size (Backbone Out): {model_params.feature_size}")

        backbone = MinkBEVBackbone(in_channels=in_channels,
                                   out_channels=model_params.feature_size,
                                   dimension=2)

        pooling = PoolingWrapper(pool_method=model_params.pooling,
                                 in_dim=model_params.feature_size,
                                 output_dim=model_params.output_dim)

        model = MinkLoc(backbone=backbone, pooling=pooling,
                        normalize_embeddings=model_params.normalize_embeddings)

    elif model_params.model == 'MinkLocCross':
        in_channels = getattr(model_params, 'in_channels', 64)
        print(f"Model Factory: Initializing MinkLocCross (Dual-Stream)...")
        print(f"  Input Channels (S-slices): {in_channels}")
        print(f"  Feature Size (Backbone Out): {model_params.feature_size}")

        # 粗流 Backbone
        backbone = MinkBEVBackbone(in_channels=in_channels,
                                   out_channels=model_params.feature_size,
                                   dimension=2)

        # 粗流 Pooling
        pooling = PoolingWrapper(pool_method=model_params.pooling,
                                 in_dim=model_params.feature_size,
                                 output_dim=model_params.output_dim)

        # =========================================================
        # 精流 Slice Sequence Branch
        # =========================================================
        # 可通过配置文件修改 slice_feature_dim，默认为 32 维
        slice_feature_dim = getattr(model_params, 'slice_feature_dim', 32)
        slice_branch = SliceSequenceBranch(num_slices=in_channels, feature_dim=slice_feature_dim)

        # 将两条流同时传入模型顶层
        model = MinkLoc(backbone=backbone, pooling=pooling,
                        normalize_embeddings=model_params.normalize_embeddings,
                        slice_branch=slice_branch)

    elif model_params.model == 'MinkLoc':
        raise NotImplementedError(
            "MinkLoc (3D) has been removed from this codebase."
        )
    else:
        raise NotImplementedError(f'Model not implemented: {model_params.model}')

    return model