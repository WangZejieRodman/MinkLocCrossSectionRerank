# misc/utils.py
import os
import configparser
import time
import numpy as np


class ModelParams:
    def __init__(self, model_params_path):
        config = configparser.ConfigParser()
        config.read(model_params_path)
        params = config['MODEL']

        self.model_params_path = model_params_path
        self.model = params.get('model')
        self.output_dim = params.getint('output_dim', 256)

        self.coordinates = params.get('coordinates', 'bev')

        # =====================================================
        # 分支 1: 原有的 BEV 解析逻辑
        # =====================================================
        if self.coordinates == 'bev':
            if 'coords_range' in params:
                self.coords_range = [float(e) for e in params['coords_range'].split(',')]
            else:
                self.coords_range = [-10., -10, -4, 10, 10, 8]

            if 'div_n' in params:
                self.div_n = [int(e) for e in params['div_n'].split(',')]
            else:
                self.div_n = [256, 256, 32]

            self.in_channels = params.getint('in_channels', self.div_n[2])

            from datasets.quantization import BEVQuantizer
            self.quantizer = BEVQuantizer(coords_range=self.coords_range, div_n=self.div_n)

        # =====================================================
        # 分支 2: 新增的 Cross-Section 解析逻辑
        # =====================================================
        elif self.coordinates == 'cross':
            if 'wz_range' in params:
                self.wz_range = [float(e) for e in params['wz_range'].split(',')]
            else:
                self.wz_range = [-10.0, -4.0, 10.0, 8.0]

            if 's_range' in params:
                self.s_range = [float(e) for e in params['s_range'].split(',')]
            else:
                self.s_range = [-12.0, 12.0]

            if 'div_n' in params:
                self.div_n = [int(e) for e in params['div_n'].split(',')]
            else:
                self.div_n = [256, 32]

            self.s_thickness = params.getfloat('s_thickness', 0.375)
            self.in_channels = params.getint('in_channels', 64)

            from datasets.cross_section_quantization import CrossSectionQuantizer
            self.quantizer = CrossSectionQuantizer(
                wz_range=self.wz_range,
                div_n=self.div_n,
                s_range=self.s_range,
                s_thickness=self.s_thickness
            )
        else:
            raise NotImplementedError(f'Unsupported coordinates: {self.coordinates}')

        # 通用参数
        self.normalize_embeddings = params.getboolean('normalize_embeddings', False)
        self.feature_size = params.getint('feature_size', 256)
        self.pooling = params.get('pooling', 'GeM')

    def print(self):
        print('Model parameters:')
        print(f'  model: {self.model}')
        print(f'  coordinates: {self.coordinates}')
        if self.coordinates == 'bev':
            print(f'  coords_range: {self.coords_range}')
        elif self.coordinates == 'cross':
            print(f'  wz_range: {self.wz_range}')
            print(f'  s_range: {self.s_range}')
            print(f'  s_thickness: {self.s_thickness}')
        print(f'  div_n: {self.div_n}')
        print(f'  in_channels: {self.in_channels}')
        print(f'  feature_size: {self.feature_size}')
        print(f'  output_dim: {self.output_dim}')
        print(f'  pooling: {self.pooling}')
        print(f'  normalize_embeddings: {self.normalize_embeddings}')
        print('')


def get_datetime():
    return time.strftime("%Y%m%d_%H%M")


class TrainingParams:
    def __init__(self, params_path: str, model_params_path: str, debug: bool = False):
        assert os.path.exists(params_path), 'Cannot find configuration file: {}'.format(params_path)
        assert os.path.exists(model_params_path), 'Cannot find model-specific configuration file: {}'.format(
            model_params_path)
        self.params_path = params_path
        self.model_params_path = model_params_path
        self.debug = debug

        config = configparser.ConfigParser()
        config.read(self.params_path)

        params = config['DEFAULT']
        self.dataset_folder = params.get('dataset_folder')

        params = config['TRAIN']
        self.save_freq = params.getint('save_freq', 0)
        self.num_workers = params.getint('num_workers', 0)

        self.batch_size = params.getint('batch_size', 64)
        self.batch_split_size = params.getint('batch_split_size', None)

        self.batch_expansion_th = params.getfloat('batch_expansion_th', None)
        if self.batch_expansion_th is not None:
            assert 0. < self.batch_expansion_th < 1.
            self.batch_size_limit = params.getint('batch_size_limit', 256)
            self.batch_expansion_rate = params.getfloat('batch_expansion_rate', 1.5)
        else:
            self.batch_size_limit = self.batch_size
            self.batch_expansion_rate = None

        self.val_batch_size = params.getint('val_batch_size', self.batch_size_limit)

        self.lr = params.getfloat('lr', 1e-3)
        self.epochs = params.getint('epochs', 20)
        self.optimizer = params.get('optimizer', 'Adam')
        self.scheduler = params.get('scheduler', 'MultiStepLR')
        if self.scheduler is not None:
            if self.scheduler == 'CosineAnnealingLR':
                self.min_lr = params.getfloat('min_lr')
            elif self.scheduler == 'MultiStepLR':
                if 'scheduler_milestones' in params:
                    scheduler_milestones = params.get('scheduler_milestones')
                    self.scheduler_milestones = [int(e) for e in scheduler_milestones.split(',')]
                else:
                    self.scheduler_milestones = [self.epochs + 1]
            else:
                raise NotImplementedError('Unsupported LR scheduler: {}'.format(self.scheduler))

        self.weight_decay = params.getfloat('weight_decay', None)
        self.loss = params.get('loss').lower()
        if 'contrastive' in self.loss:
            self.pos_margin = params.getfloat('pos_margin', 0.2)
            self.neg_margin = params.getfloat('neg_margin', 0.65)
        elif 'triplet' in self.loss:
            self.margin = params.getfloat('margin', 0.4)
        elif self.loss == 'truncatedsmoothap':
            self.positives_per_query = params.getint("positives_per_query", 4)
            self.tau1 = params.getfloat('tau1', 0.01)
            self.margin = params.getfloat('margin', None)

        self.similarity = params.get('similarity', 'euclidean')
        assert self.similarity in ['cosine', 'euclidean']

        self.aug_mode = params.getint('aug_mode', 1)
        self.set_aug_mode = params.getint('set_aug_mode', 1)
        self.train_file = params.get('train_file')
        self.val_file = params.get('val_file', None)
        self.test_file = params.get('test_file', None)

        self.model_params = ModelParams(self.model_params_path)
        self._check_params()

    def _check_params(self):
        assert os.path.exists(self.dataset_folder), 'Cannot access dataset: {}'.format(self.dataset_folder)

    def print(self):
        print('Parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e != 'model_params':
                print('{}: {}'.format(e, param_dict[e]))

        self.model_params.print()
        print('')