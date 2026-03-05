import os
import sys
import torch
import numpy as np
import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from eval.evaluate_cyd import evaluate_cyd, compute_embedding
from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader
from datasets.rotation_utils import rotate_point_cloud_z
from misc.utils import TrainingParams
from models.model_factory import model_factory


def evaluate_cyd_with_rotation(model, device, params, rotation_angles):
    import pickle
    from sklearn.neighbors import KDTree

    dataset_cyd_path = os.path.join(project_root, 'datasets', 'cyd')
    database_path = os.path.join(dataset_cyd_path, 'cyd_evaluation_database_109_111.pickle')
    query_path = os.path.join(dataset_cyd_path, 'cyd_evaluation_query_112_113.pickle')

    with open(database_path, 'rb') as f:
        database_sets = pickle.load(f)
    with open(query_path, 'rb') as f:
        query_sets = pickle.load(f)

    # 数据库不旋转
    database_embeddings = []
    print("\nComputing database embeddings (0°)...")
    for db_set in tqdm.tqdm(database_sets):
        database_embeddings.append(
            get_latent_vectors_rot(model, db_set, device, params, 0) if len(db_set) > 0 else np.array([]).reshape(0,
                                                                                                                  256))

    all_database_embeddings = []
    database_to_session_map = []
    for i, db_emb in enumerate(database_embeddings):
        if len(db_emb) > 0:
            all_database_embeddings.append(db_emb)
            for local_idx in range(len(db_emb)): database_to_session_map.append((i, local_idx))

    database_output = np.vstack(all_database_embeddings)
    database_nbrs = KDTree(database_output)

    all_stats = {}
    for angle in rotation_angles:
        print(f"\n[{angle}°] Computing query embeddings...")
        query_embeddings = []
        for query_set in tqdm.tqdm(query_sets):
            query_embeddings.append(
                get_latent_vectors_rot(model, query_set, device, params, angle) if len(query_set) > 0 else np.array(
                    []).reshape(0, 256))

        recall = np.zeros(25)
        num_evaluated = 0
        for j, query_set in enumerate(query_sets):
            if len(query_set) == 0: continue
            queries_output = query_embeddings[j]
            for query_idx in range(len(queries_output)):
                query_details = query_set[query_idx]
                if 'positives' not in query_details: continue

                true_neighbors_global = []
                for db_session_idx in query_details['positives']:
                    for local_idx in query_details['positives'][db_session_idx]:
                        for global_idx, (sess_idx, loc_idx) in enumerate(database_to_session_map):
                            if sess_idx == db_session_idx and loc_idx == local_idx:
                                true_neighbors_global.append(global_idx)

                if len(true_neighbors_global) == 0: continue
                num_evaluated += 1
                distances, indices = database_nbrs.query(np.array([queries_output[query_idx]]), k=25)
                for k in range(len(indices[0])):
                    if indices[0][k] in true_neighbors_global:
                        recall[k:] += 1
                        break

        ave_recall = (recall / float(num_evaluated)) * 100.0
        all_stats[angle] = {'ave_recall': ave_recall}
        print(f"  --> Recall@1 for {angle}°: {ave_recall[0]:.2f}%")

    return all_stats


def get_latent_vectors_rot(model, point_cloud_set, device, params, rotation_angle):
    pc_loader = PNVPointCloudLoader()
    model.eval()
    embeddings = None
    for i, elem_ndx in enumerate(point_cloud_set):
        pc_file_path = os.path.join(params.dataset_folder, point_cloud_set[elem_ndx]["query"])
        pc, cl = pc_loader(pc_file_path)

        # 如果测试旋转，不仅旋转点云，对应的中心线也要同步旋转！
        if rotation_angle != 0:
            pc = rotate_point_cloud_z(pc, rotation_angle)
            cl[:, :3] = rotate_point_cloud_z(cl[:, :3], rotation_angle)

        pc = torch.tensor(pc, dtype=torch.float)
        embedding = compute_embedding(model, pc, cl, device, params)

        if embeddings is None: embeddings = np.zeros((len(point_cloud_set), embedding.shape[1]), dtype=embedding.dtype)
        embeddings[i] = embedding
    return embeddings


if __name__ == "__main__":
    config_file = '../config/config_cyd_cross.txt'
    model_config_file = '../models/minkloc_cross.txt'
    weights_file = '/home/wzj/pan1/MinkLocCrossSection/weights/model_MinkLocCross_20260301_1708_final.pth'

    params = TrainingParams(config_file, model_config_file, debug=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model_factory(params.model_params)
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.to(device)

    rotation_angles = [0, 5, 10, 15, 30, 45, 90, 180]
    print("=" * 60)
    print("开始旋转鲁棒性测试 (Cross-Section 特性)")
    print("=" * 60)

    all_stats = evaluate_cyd_with_rotation(model, device, params, rotation_angles)