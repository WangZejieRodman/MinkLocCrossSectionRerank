import os
import sys
import pickle
import numpy as np
import torch
import tqdm
from sklearn.neighbors import KDTree
import MinkowskiEngine as ME

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.model_factory import model_factory
from misc.utils import TrainingParams
from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader


# ========================================================
# 极速 Numpy 版 DTW 距离计算 (支持序列不等长对齐)
# ========================================================
def compute_dtw_distance(seq1, seq2):
    """
    计算两个特征序列的 DTW 距离
    :param seq1: (N, D) numpy array
    :param seq2: (M, D) numpy array
    """
    N, M = seq1.shape[0], seq2.shape[0]
    # 如果有任意一个序列为空，返回无穷大
    if N == 0 or M == 0:
        return float('inf')

    # 计算 pairwise 欧氏距离矩阵
    # 利用 numpy 广播机制快速计算
    dist_mat = np.linalg.norm(seq1[:, None, :] - seq2[None, :, :], axis=2)

    # 动态规划矩阵
    dtw_matrix = np.full((N + 1, M + 1), float('inf'))
    dtw_matrix[0, 0] = 0.0

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = dist_mat[i - 1, j - 1]
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],  # 插入
                                          dtw_matrix[i, j - 1],  # 删除
                                          dtw_matrix[i - 1, j - 1])  # 匹配

    return dtw_matrix[N, M]


def filter_empty_slices(seq):
    """
    剔除全 0 的空切片，避免空洞干扰 DTW 对齐
    :param seq: (64, D) numpy array
    """
    norms = np.linalg.norm(seq, axis=1)
    # 保留 L2 范数大于 1e-5 的有效切片
    return seq[norms > 1e-5]


def evaluate_cyd(model, device, params: TrainingParams, log=False, show_progress=False):
    dataset_cyd_path = os.path.join(project_root, 'datasets', 'cyd')
    database_path = os.path.join(dataset_cyd_path, 'cyd_evaluation_database_109_111.pickle')
    query_path = os.path.join(dataset_cyd_path, 'cyd_evaluation_query_112_113.pickle')

    if not os.path.exists(database_path) or not os.path.exists(query_path):
        print("❌ 错误: 找不到测试字典文件。")
        return None

    with open(database_path, 'rb') as f:
        database_sets = pickle.load(f)
    with open(query_path, 'rb') as f:
        query_sets = pickle.load(f)

    model.eval()

    # ==========================================
    # 提取 Database 的 双流 特征
    # ==========================================
    db_global_list = []
    db_seq_list = []
    database_to_session_map = []

    print("Computing database embeddings (Dual-Stream)...")
    for i, db_set in enumerate(tqdm.tqdm(database_sets, disable=not show_progress)):
        if len(db_set) > 0:
            glob_emb, seq_emb = get_latent_vectors(model, db_set, device, params)
            db_global_list.append(glob_emb)
            db_seq_list.append(seq_emb)
            for local_idx in range(len(glob_emb)):
                database_to_session_map.append((i, local_idx))

    # 构建全局粗流索引
    database_global_output = np.vstack(db_global_list)
    # 展平序列特征列表，方便后续精检索
    database_seq_output = np.concatenate(db_seq_list, axis=0)

    database_nbrs = KDTree(database_global_output)

    # ==========================================
    # 提取 Query 的 双流 特征
    # ==========================================
    query_global_list = []
    query_seq_list = []
    print("Computing query embeddings (Dual-Stream)...")
    for query_set in tqdm.tqdm(query_sets, disable=not show_progress):
        if len(query_set) > 0:
            glob_emb, seq_emb = get_latent_vectors(model, query_set, device, params)
            query_global_list.append(glob_emb)
            query_seq_list.append(seq_emb)
        else:
            query_global_list.append(np.array([]).reshape(0, 256))
            query_seq_list.append(np.array([]).reshape(0, 64, 32))

    num_neighbors = 25
    recall_coarse = np.zeros(num_neighbors)
    recall_fine = np.zeros(num_neighbors)
    one_percent_recall_list = []
    num_evaluated = 0
    threshold = max(int(round(len(database_global_output) / 100.0)), 1)

    print("\nStarting Coarse-to-Fine Dual-Stream Retrieval...")
    for j, query_set in enumerate(tqdm.tqdm(query_sets, desc="Retrieving")):
        if len(query_set) == 0: continue

        q_globals = query_global_list[j]
        q_seqs = query_seq_list[j]

        for query_idx in range(len(q_globals)):
            query_details = query_set[query_idx]
            if 'positives' not in query_details: continue

            # 获取真实 Ground Truth
            true_neighbors_global = []
            for db_session_idx in query_details['positives']:
                for local_idx in query_details['positives'][db_session_idx]:
                    for global_idx, (sess_idx, loc_idx) in enumerate(database_to_session_map):
                        if sess_idx == db_session_idx and loc_idx == local_idx:
                            true_neighbors_global.append(global_idx)
                            break

            if len(true_neighbors_global) == 0: continue
            num_evaluated += 1

            # ==========================================
            # 阶段一：粗流 (Coarse) - KD-Tree 极速召回 Top-25
            # ==========================================
            distances, indices = database_nbrs.query(np.array([q_globals[query_idx]]), k=num_neighbors)
            coarse_candidates = indices[0]

            # 统计粗流 Recall (作为 Baseline 对比)
            for k in range(len(coarse_candidates)):
                if coarse_candidates[k] in true_neighbors_global:
                    recall_coarse[k:] += 1
                    break

            top_1_percent = coarse_candidates[0:threshold]
            one_percent_recall_list.append(
                1.0 if len(set(top_1_percent).intersection(set(true_neighbors_global))) > 0 else 0.0)

            # ==========================================
            # 阶段二：精流 (Fine) - DTW 序列重排 (Re-ranking)
            # ==========================================
            q_seq_valid = filter_empty_slices(q_seqs[query_idx])

            fine_scores = []
            for candidate_idx in coarse_candidates:
                c_seq_valid = filter_empty_slices(database_seq_output[candidate_idx])

                # 正向 DTW
                dtw_fwd = compute_dtw_distance(q_seq_valid, c_seq_valid)
                # 逆向 DTW (应对双向行驶)
                c_seq_rev = c_seq_valid[::-1]
                dtw_bwd = compute_dtw_distance(q_seq_valid, c_seq_rev)

                # 取最小值作为该候选帧的最终相似度距离
                min_dtw = min(dtw_fwd, dtw_bwd)
                fine_scores.append(min_dtw)

            # 根据 DTW 距离对 Top-25 重新升序排序
            re_ranked_indices = [x for _, x in sorted(zip(fine_scores, coarse_candidates))]

            # 统计精流重排后的 Recall
            for k in range(len(re_ranked_indices)):
                if re_ranked_indices[k] in true_neighbors_global:
                    recall_fine[k:] += 1
                    break

    ave_recall_coarse = (recall_coarse / float(num_evaluated)) * 100.0
    ave_recall_fine = (recall_fine / float(num_evaluated)) * 100.0
    ave_one_percent_recall = np.mean(one_percent_recall_list) * 100.0

    return {
        'ave_one_percent_recall': ave_one_percent_recall,
        'ave_recall_coarse': ave_recall_coarse,
        'ave_recall_fine': ave_recall_fine,
        'num_evaluated': num_evaluated
    }


def get_latent_vectors(model, point_cloud_set, device, params):
    pc_loader = PNVPointCloudLoader()
    model.eval()

    global_embeddings = None
    sequence_embeddings = None

    for i, elem_ndx in enumerate(point_cloud_set):
        pc_file_path = os.path.join(params.dataset_folder, point_cloud_set[elem_ndx]["query"])
        pc, cl = pc_loader(pc_file_path)
        pc = torch.tensor(pc, dtype=torch.float)

        glob_emb, seq_emb = compute_embedding(model, pc, cl, device, params)

        if global_embeddings is None:
            global_embeddings = np.zeros((len(point_cloud_set), glob_emb.shape[1]), dtype=glob_emb.dtype)
            sequence_embeddings = np.zeros((len(point_cloud_set), seq_emb.shape[1], seq_emb.shape[2]),
                                           dtype=seq_emb.dtype)

        global_embeddings[i] = glob_emb
        sequence_embeddings[i] = seq_emb

    return global_embeddings, sequence_embeddings


def compute_embedding(model, pc, cl, device, params):
    coords, feats = params.model_params.quantizer(pc, cl)
    with torch.no_grad():
        bcoords = ME.utils.batched_coordinates([coords])
        batch = {'coords': bcoords.to(device), 'features': feats.to(device)}

        y = model(batch)

        glob_emb = y['global'].detach().cpu().numpy()

        # 兼容处理：检查模型是否输出了序列特征
        if 'sequence' in y:
            seq_emb = y['sequence'].detach().cpu().numpy()
        else:
            # 如果没有开启精流，返回空的序列特征占位
            seq_emb = np.zeros((1, 64, 32), dtype=np.float32)

    return glob_emb, seq_emb


if __name__ == "__main__":
    config_file = '../config/config_cyd_cross.txt'
    model_config_file = '../models/minkloc_cross.txt'

    # 【注意】这里请替换为你刚才跑出来的 epoch_x.pth 权重文件路径
    # 即使才跑了 1-2 个 epoch 也可以测试一下验证逻辑
    weights_file = '/home/wzj/pan1/MinkLocCrossSection_clean/weights/model_MinkLocCross_20260304_0224_final.pth'

    params = TrainingParams(config_file, model_config_file, debug=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model_factory(params.model_params)

    # 加载权重
    try:
        model.load_state_dict(torch.load(weights_file, map_location=device))
        print(f"✅ 成功加载模型权重: {weights_file}")
    except Exception as e:
        print(f"⚠️ 警告: 加载权重失败，可能是路径不对。使用随机初始化权重进行测试。({e})")

    model.to(device)

    print("\n" + "=" * 60)
    print("开始评估 CYD 数据集 (单帧级粗精双流)")
    print("=" * 60)

    stats = evaluate_cyd(model, device, params, show_progress=True)

    if stats is not None:
        print("\n" + "=" * 60)
        print(f"评估完成！总评估 Query 数量: {stats['num_evaluated']}")
        print(f"Top 1% Recall: {stats['ave_one_percent_recall']:.2f}%")
        print("-" * 60)
        print("【第一阶段：粗流 (KD-Tree Baseline)】")
        print(f"  Recall@1:  {stats['ave_recall_coarse'][0]:.2f}%")
        print(f"  Recall@5:  {stats['ave_recall_coarse'][4]:.2f}%")
        print(f"  Recall@10: {stats['ave_recall_coarse'][9]:.2f}%")
        print("-" * 60)
        print("【第二阶段：精流 (DTW Re-ranking 逆序双向增强)】")
        print(f"  Recall@1:  {stats['ave_recall_fine'][0]:.2f}%")
        print(f"  Recall@5:  {stats['ave_recall_fine'][4]:.2f}%")
        print(f"  Recall@10: {stats['ave_recall_fine'][9]:.2f}%")
        print("=" * 60)