# training/train_cyd_cross.py
import os
import sys
import torch

# 将项目根目录加入环境变量，以便正确导入模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from training.trainer import do_train
from misc.utils import TrainingParams

def train_cyd_cross():
    class Args:
        def __init__(self):
            # 1. 训练参数配置文件 (刚刚新建的)
            self.config = '../config/config_cyd_cross.txt'
            # 2. 模型结构配置文件 (之前新建的横截面模型配置)
            self.model_config = '../models/minkloc_cross.txt'
            # 3. 调试模式
            self.debug = False

    args = Args()

    # 路径检查
    if not os.path.exists(args.config):
        print(f"错误: 找不到配置文件: {os.path.abspath(args.config)}")
        return
    if not os.path.exists(args.model_config):
        print(f"错误: 找不到模型配置文件: {os.path.abspath(args.model_config)}")
        return

    print('=' * 60)
    print('启动 CYD 横截面 (Cross-Section) 网络训练')
    print('=' * 60)
    print(f'Training Config : {args.config}')
    print(f'Model Config    : {args.model_config}')
    print('')

    # 解析参数
    params = TrainingParams(args.config, args.model_config, debug=args.debug)
    params.print()

    # 将当前工作目录切换到项目根目录，以防某些相对路径报错
    os.chdir(project_root)

    # 开始执行训练循环 (复用原项目逻辑)
    model, model_pathname = do_train(params, skip_final_eval=True)

    print('\n' + '=' * 60)
    print(f'训练结束！')
    print(f'最终模型权重已保存在: {model_pathname}_final.pth')
    print('=' * 60)

if __name__ == '__main__':
    train_cyd_cross()