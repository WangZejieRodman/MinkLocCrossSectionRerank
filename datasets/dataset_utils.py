# datasets/dataset_utils.py
import numpy as np
from typing import List
import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from sklearn.neighbors import KDTree

from datasets.base_datasets import EvaluationTuple, TrainingDataset
from datasets.augmentation import TrainSetTransform
from datasets.pointnetvlad.pnv_train import PNVTrainingDataset
from datasets.pointnetvlad.pnv_train import TrainTransform as PNVTrainTransform
from datasets.samplers import BatchSampler
from misc.utils import TrainingParams
from datasets.base_datasets import PointCloudLoader
from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader


def get_pointcloud_loader(dataset_type) -> PointCloudLoader:
    return PNVPointCloudLoader()


def make_datasets(params: TrainingParams, validation: bool = True):
    datasets = {}
    train_set_transform = TrainSetTransform(params.set_aug_mode) if params.set_aug_mode > 0 else None
    train_transform = PNVTrainTransform(params.aug_mode) if params.aug_mode > 0 else None

    datasets['train'] = PNVTrainingDataset(params.dataset_folder, params.train_file,
                                           transform=train_transform, set_transform=train_set_transform)
    if validation:
        datasets['val'] = PNVTrainingDataset(params.dataset_folder, params.val_file)

    return datasets


def make_collate_fn(dataset: TrainingDataset, quantizer, batch_split_size=None):
    def collate_fn(data_list):
        # 解包 ( (pc, centerline), label )
        clouds = [e[0][0] for e in data_list]
        centerlines = [e[0][1] for e in data_list]
        labels = [e[1] for e in data_list]

        if dataset.set_transform is not None:
            lens = [len(cloud) for cloud in clouds]
            clouds = torch.cat(clouds, dim=0)
            # 同时将全局点云和中心线列表传进去做同步随机旋转翻转
            clouds, centerlines = dataset.set_transform((clouds, centerlines))
            clouds = clouds.split(lens)

        # Compute positives and negatives mask
        positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in
                          labels]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)

        # =========================================================
        # 横截面 特征输入 (同时传入 pc 和 centerline)
        # =========================================================
        batch_coords = []
        batch_feats = []

        for pc, cl in zip(clouds, centerlines):
            # 调用量化器，这里要求量化器支持传入 (pc, centerline)
            coords, features = quantizer(pc, cl)
            batch_coords.append(coords)
            batch_feats.append(features)

        if batch_split_size is None or batch_split_size == 0:
            coords = ME.utils.batched_coordinates(batch_coords)
            feats = torch.cat(batch_feats, dim=0)
            batch = {'coords': coords, 'features': feats}
        else:
            batch = []
            for i in range(0, len(batch_coords), batch_split_size):
                temp_coords = batch_coords[i:i + batch_split_size]
                temp_feats = batch_feats[i:i + batch_split_size]
                c = ME.utils.batched_coordinates(temp_coords)
                f = torch.cat(temp_feats, dim=0)
                minibatch = {'coords': c, 'features': f}
                batch.append(minibatch)

        return batch, positives_mask, negatives_mask

    return collate_fn


def make_dataloaders(params: TrainingParams, validation=True):
    datasets = make_datasets(params, validation=validation)
    dataloders = {}
    train_sampler = BatchSampler(datasets['train'], batch_size=params.batch_size,
                                 batch_size_limit=params.batch_size_limit,
                                 batch_expansion_rate=params.batch_expansion_rate)

    quantizer = params.model_params.quantizer
    train_collate_fn = make_collate_fn(datasets['train'], quantizer, params.batch_split_size)
    dataloders['train'] = DataLoader(datasets['train'], batch_sampler=train_sampler,
                                     collate_fn=train_collate_fn, num_workers=params.num_workers,
                                     pin_memory=True)
    if validation and 'val' in datasets:
        val_collate_fn = make_collate_fn(datasets['val'], quantizer, params.batch_split_size)
        val_sampler = BatchSampler(datasets['val'], batch_size=params.val_batch_size)
        dataloders['val'] = DataLoader(datasets['val'], batch_sampler=val_sampler, collate_fn=val_collate_fn,
                                       num_workers=params.num_workers, pin_memory=True)

    return dataloders


def filter_query_elements(query_set: List[EvaluationTuple], map_set: List[EvaluationTuple], dist_threshold: float) -> \
List[EvaluationTuple]:
    map_pos = np.zeros((len(map_set), 2), dtype=np.float32)
    for ndx, e in enumerate(map_set):
        map_pos[ndx] = e.position
    kdtree = KDTree(map_pos)
    filtered_query_set = []
    count_ignored = 0
    for ndx, e in enumerate(query_set):
        position = e.position.reshape(1, -1)
        nn = kdtree.query_radius(position, dist_threshold, count_only=True)[0]
        if nn > 0:
            filtered_query_set.append(e)
        else:
            count_ignored += 1
    print(
        f"{count_ignored} query elements ignored - not having corresponding map element within {dist_threshold} [m] radius")
    return filtered_query_set


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e