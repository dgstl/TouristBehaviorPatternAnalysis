"""
Group activity path pattern recognition
"""

import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import geopandas as gpd


def calculate_distance(p1, p2, point_type):
    """
    计算两点的欧氏距离
    Args:
        p1:
        p2:

    Returns:

    """
    if point_type == 'angle':
        return abs(p1 - p2)
    else:
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_angles(coordinates):
    """
    计算每条轨迹的前进方向角
    Args:
        coordinates:

    Returns:

    """
    angles = []
    for i in range(len(coordinates) - 1):
        x_i, y_i = coordinates[i][0], coordinates[i][1]
        x_next, y_next = coordinates[i + 1][0], coordinates[i + 1][1]
        dx = x_next - x_i
        dy = y_next - y_i

        if dx > 0:
            angle = np.arctan2(dy, dx)
        elif dx <= 0 and dy >= 0:
            angle = np.arctan2(dy, dx) + np.pi
        else:
            angle = np.arctan2(dy, dx) - np.pi

        angles.append(angle)

    return angles


def hausdorff_one_way(A, B, input_type):
    distances = []
    for a in A:
        min_distance = min(calculate_distance(a, b, input_type) for b in B)
        distances.append((a, min_distance))
    return distances


def spatial_distance(A, B, input_type='points'):
    # Compute A -> B and B -> A
    distances_A_to_B = hausdorff_one_way(A, B, input_type)
    distances_B_to_A = hausdorff_one_way(B, A, input_type)

    dist_a, dist_b = 0, 0

    # print("A to B matching points and distances:")
    for point, dist in distances_A_to_B:
        dist_a += dist
        # print(f"Point {point} has closest distance {dist}")

    # print("\nB to A matching points and distances:")
    for point, dist in distances_B_to_A:
        # print(f"Point {point} has closest distance {dist}")
        dist_b += dist

    final_dist = 0.5 * (dist_a / len(distances_A_to_B)) + 0.5 * (dist_b / len(distances_B_to_A))

    return final_dist


def custom_distance(A, B):
    # 要不要把角度和空间距离都转换为(0,1)之间的数值，归一化
    spatial_dist = spatial_distance(A, B)
    # Calculate the angles
    A_angle = calculate_angles(A)
    B_angle = calculate_angles(B)
    angel_dist = spatial_distance(A_angle, B_angle, 'angle')

    # traj_dist = 0.5 * spatial_dist + 0.5 * angel_dist
    # print(traj_dist)

    return spatial_dist, angel_dist


def scale_matrix(matrix):
    # 将输入转换为NumPy数组，以便可以使用NumPy的功能
    matrix = np.array(matrix)

    # 找到矩阵中的最大值
    max_value = np.max(matrix)

    # 如果最大值大于0，则缩放矩阵
    if max_value > 0:
        scaled_matrix = matrix / max_value
    else:
        # 如果最大值为0或负数，缩放后的矩阵将全为0
        scaled_matrix = matrix

    return scaled_matrix


def save_distance_matrix(data, file):
    user_trajectories = {}
    for k, v in data.items():
        trjs = []
        for point in v:
            trjs.append(point[1:])
        user_trajectories[k] = trjs

    user_ids = list(user_trajectories.keys())

    # 初始化距离矩阵
    num_users = len(user_ids)
    spatial_distance_matrix = np.zeros((num_users, num_users))
    angel_distance_matrix = np.zeros((num_users, num_users))
    print('number of users:', num_users)

    # 计算距离矩阵的上三角部分（不包括对角线）
    for i in range(num_users):
        print(i, ' of ', num_users, ' users.')
        for j in range(i, num_users):  # 只计算上三角部分
            spatial_distance_matrix[i][j], angel_distance_matrix[i][j] = (
                custom_distance(user_trajectories[user_ids[i]], user_trajectories[user_ids[j]]))

    distance_matrix = 0.5 * scale_matrix(spatial_distance_matrix) + 0.5 * scale_matrix(angel_distance_matrix)

    # 通过转置上三角部分来填充下三角部分
    distance_matrix += distance_matrix.T - np.diag(distance_matrix.diagonal())

    # 对角线设为0（如果是自己与自己比较）
    np.fill_diagonal(distance_matrix, 0)

    np.save(file, distance_matrix)


def perform_clustering(users, m_file, method='ward', cut_pos=0.2):
    if '10' in m_file:
        cluster_out_path = os.path.join(os.path.dirname(m_file), '10_route_cluster.csv')
        dendrogram_out_path = os.path.join(os.path.dirname(m_file), '10_route_cluster.jpg')
    else:
        cluster_out_path = os.path.join(os.path.dirname(m_file), '14_route_cluster.csv')
        dendrogram_out_path = os.path.join(os.path.dirname(m_file), '14_route_cluster.jpg')

    """Perform hierarchical clustering using a custom distance metric."""
    # Convert dictionary data to list of arrays
    distance_matrix = np.load(m_file)

    # 打印行列数
    print("行数:", distance_matrix.shape[0])
    print("列数:", distance_matrix.shape[1])

    Z = linkage(distance_matrix, method)

    # 绘制树状图
    plt.figure(figsize=(9, 7))  # 5.3
    dendrogram(Z, orientation='top', distance_sort='descending', show_leaf_counts=True)

    # 假设我们选择最大高度的某个百分比作为阈值
    max_d = cut_pos * np.max(Z[:, 2])  # 例如，取最大高度的70%

    # 在树状图上绘制分割线
    plt.hlines(y=max_d, xmin=0, xmax=plt.gca().get_xlim()[1], colors='r', linestyles='--')

    # 隐藏x轴的ticks标签
    plt.xticks([], [])

    # Set the yticks with the specified font
    plt.yticks(fontsize=18, fontproperties=FontProperties(family='Times New Roman'))

    # 显示图表
    plt.ylabel('Distance', fontsize=20, fontname='Times New Roman')
    plt.tight_layout()
    plt.savefig(dendrogram_out_path, dpi=100)

    # 根据阈值 max_d 来确定簇的数量
    clusters = fcluster(Z, max_d, criterion='distance')

    # 打印出分类的数量
    num_clusters = len(set(clusters))  # set()用于获取唯一集群编号的集合，其长度即为集群的数量
    print(f"Number of clusters: {num_clusters}")

    cluster_results = []
    for i, cluster_id in enumerate(clusters):
        cluster_results.append((users[i], cluster_id))

    dtype = [('user', 'U10'), ('id', int)]  # 'U10' 表示最大长度为10的Unicode字符串
    structured_array = np.array(cluster_results, dtype=dtype)

    np.savetxt(cluster_out_path, structured_array, fmt=['%s', '%d'], delimiter=',', newline='\n')

    return cluster_results


if __name__ == "__main__":
    """ process datasets """
    base_dir = 'tourists_traces'
    with open('camera_params.json', 'r') as f:
        cameras = json.load(f)

    date_str = '20240818'
    for cam, vals in cameras.items():
        if cam != 'guierlu04-2':
            continue
        for i in range(0, 2):
            filtered_geo_trace = os.path.join(base_dir, date_str, cam, vals['geotraces_filtered'][i])
            with open(filtered_geo_trace, 'r') as f:
                tourist_traj = json.load(f)

            # Step1: 计算所有过滤后有效轨迹之间的相似度
            print(len(tourist_traj.keys()))
            # keys_to_extract = list(tourist_traj.keys())[:10]  # 获取字典的前10个键
            # selected_data = {key: tourist_traj[key] for key in keys_to_extract}
            # save_distance_matrix(tourist_traj, os.path.join(base_dir, date_str, cam, vals['similar_matrix'][i]))
            # distance_matrix = np.load(os.path.join(base_dir, date_str, cam, vals['similar_matrix'][i]))
            # print(distance_matrix.shape)

            # Step2: 根据轨迹的相似度矩阵，实现路径模式聚类与划分
            # Perform clustering
            all_users = list(tourist_traj.keys())
            user_class = perform_clustering(all_users, os.path.join(base_dir, date_str, cam, vals['similar_matrix'][i]),
                               cut_pos=0.2)
            # wygc05-am:0.15        ;wygc05-pm:0.15;
            # guierlu04-2-am:0.2   ;guierlu04-2-pm:0.2;
            # guierlu06-1-am：0.1  ;guierlu06-1-pm：0.2
            # daqiongding-am:0.08  ;daqiongding-pm:0.1

            # Step3: 统计各类线路特征
            # 将GeoJSON字符串转换为GeoDataFrame
            # gdf = gpd.read_file(os.path.join(base_dir, date_str, cam, vals['geojson'][i]))
            # gdf['coordinates'] = gdf['geometry'].apply(lambda x: '; '.join(map(str, x.coords)))
            # df_geo = gdf.drop('geometry', axis=1)
            # print(len(df_geo))
            #
            # cps = '10_route_cluster.csv' if i == 0 else '14_route_cluster.csv'
            # df_cls = pd.read_csv(os.path.join(base_dir, date_str, cam, cps), header=None, names=['user_id', 'route_cls'])
            #
            # df_geo['user_id'] = df_geo['user_id'].astype(str)
            # df_cls['user_id'] = df_cls['user_id'].astype(str)
            # merged_df = pd.merge(df_geo, df_cls, on='user_id', how='inner')
            # merged_df['angle'] = ''
            # for index, row in merged_df.iterrows():
            #     angle = row['direction']
            #     dir_str = ''
            #     if 45 <= angle < 135:
            #         dir_str = 'E'
            #     elif 135 <= angle < 225:
            #         dir_str = 'S'
            #     elif 225 <= angle < 315:
            #         dir_str = 'W'
            #     else:
            #         dir_str = 'N'
            #
            #     merged_df.at[index, 'angle'] = dir_str
            #
            # grouped = merged_df.groupby('route_cls').agg({
            #     'route_cls': 'size',  # 计算每个唯一值的出现次数
            #     'total_distance': 'mean',  # 计算'Value'列的平均值
            #     'average_speed': 'mean',  # 计算'OtherColumn'列的平均值
            #     'direction': 'mean',
            #     'angle': lambda x: x.mode()[0] if not x.empty else None  # 使用 lambda 函数来获取众数
            # })
            #
            # merged_df.to_csv('all_user_attrs.csv')
            #
            # print(grouped)
