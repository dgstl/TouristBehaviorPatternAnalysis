"""
Recognition of tourists' spatial and temporal stay patterns
"""

import csv
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import st_clustering as stc
from matplotlib.font_manager import FontProperties
from osgeo import gdal

from TouristBehaviorPatternAnalysis.personal_trace_plot import calculate_total_distance
from TouristBehaviorPatternAnalysis.plot_util import stay_time_pattern


def plot_3d_raster(data, labels, tif_path, out_path):
    # 加载TIFF影像
    dataset = gdal.Open(tif_path)
    if dataset is None:
        print("TIFF file could not be opened")
        return

    # 获取影像的地理变换参数和投影信息
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    # 将TIFF数据转换为数组
    red_band = dataset.GetRasterBand(1).ReadAsArray().astype(np.float64)
    green_band = dataset.GetRasterBand(2).ReadAsArray().astype(np.float64)
    blue_band = dataset.GetRasterBand(3).ReadAsArray().astype(np.float64)

    # 归一化波段数据到0-1范围
    red_band /= 255.0
    green_band /= 255.0
    blue_band /= 255.0

    tiff_data = np.dstack((red_band, green_band, blue_band))

    y_bottom = geotransform[3]
    y_top = geotransform[3] + tiff_data.shape[0] * geotransform[5]

    # 创建3D散点图
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图
    colors = plt.cm.tab20(np.linspace(0, 1, len(set(labels))))
    for k, col in zip(set(labels), colors):
        edgecolor = 'k' if k == -1 else col
        facecolor = 'none' if k == -1 else col
        if k > -1:
            class_member_mask = (labels == k)
        xyz = data[class_member_mask]
        # scatter = ax.scatter(xyz[:, 1], xyz[:, 2], xyz[:, 0], 'o', c=facecolor, edgecolor=edgecolor,
        #                      s=15, label=f'Class {k}' if k != -1 else 'Outliers')
        scatter = ax.scatter(xyz[:, 1], xyz[:, 2], xyz[:, 0], 'o', c=facecolor, edgecolor=edgecolor,
                             s=15, label=f'Class {k}')
        # y_bottom-(xyz[:, 2]-y_top)
        scatter.set_zorder(3)

    # 绘制TIFF影像作为底图
    # 假设tif_data的坐标与你的数据集坐标对齐，这里仅演示如何绘制
    # x = np.linspace(geotransform[0], geotransform[0] + tiff_data.shape[1] * geotransform[1], tiff_data.shape[1])
    # y = np.linspace(geotransform[3], geotransform[3] + tiff_data.shape[0] * geotransform[5], tiff_data.shape[0])
    # X, Y = np.meshgrid(x, y)
    # ax.scatter(X.flatten(), Y.flatten(), np.zeros_like(X.flatten()), c=tiff_data.reshape(-1, 3), cmap=None, zorder=0,
    #            alpha=0.05)

    X_plane, Y_plane = np.meshgrid(
        np.linspace(geotransform[0], geotransform[0] + tiff_data.shape[1] * geotransform[1], tiff_data.shape[1]),
        np.linspace(geotransform[3], geotransform[3] + tiff_data.shape[0] * geotransform[5], tiff_data.shape[0]))
    Z_plane = np.zeros_like(X_plane)

    # 显示图像作为平面，确保它在其他对象的后面
    ax.plot_surface(X_plane, Y_plane, Z_plane, rstride=2, cstride=2, facecolors=tiff_data, shade=False, zorder=1,
                    alpha=0.2)

    ax.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Time')

    font_prop = FontProperties(family='Times New Roman', style='normal', size=20, weight='bold')
    ax.set_title('{} spatiotemporal stay points'.format(len(set(labels)) - 1), fontproperties=font_prop, y=0.995)

    ax.view_init(elev=30, azim=-80)

    # 对于x轴
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    # 对于y轴
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%d'))

    plt.tight_layout(pad=0.2, h_pad=None, w_pad=None, rect=None)
    plt.savefig(out_path, dpi=300)


def st_cluster(data, space_radius=2, time_radius=1, min_pts=10, pic_path='', out=''):
    data[:, 0] /= 60

    # we can choose:
    # ST_DBSCAN, ST_Agglomerative, ST_OPTICS, ST_SpectralClustering, ST_AffinityPropagation, ST_HDBSCAN
    st_dbscan = stc.ST_DBSCAN(eps1=space_radius, eps2=time_radius, min_samples=min_pts)

    # st_dbscan.fit(data)
    st_dbscan.st_fit(data[:, :3])
    # st_dbscan.st_fit_frame_split(data, frame_size=10)
    labels = st_dbscan.labels

    print(np.unique(labels))
    print(len(labels))
    print(len(data))

    point_cls = []
    for index in range(0, len(data)):
        if labels[index] > -1:
            # print(data[index][1], data[index][2], labels[index])
            point_cls.append(
                (data[index][1], data[index][2], data[index][0], data[index][3], labels[index]))  # x,y,t,duration,class

    return point_cls
    # plot_3d(data, st_dbscan.labels, pic_path, out)
    # plot_3d_raster(data, st_dbscan.labels, pic_path, out)


def identify_stay_points(traj_points, time_window=10, speed_threshold=0.7):
    stay_points = []
    current_stay = []

    for i in range(1, len(traj_points)):
        # 计算当前点和前一个点之间的时间差（假设时间单位为秒）
        time_diff = traj_points[i][0] - traj_points[i - 1][0]

        # 计算距离
        distance = math.sqrt(
            (traj_points[i][1] - traj_points[i - 1][1]) ** 2 + (traj_points[i][2] - traj_points[i - 1][2]) ** 2)

        # 计算速度
        speed = distance / time_diff if time_diff > 0 else 0

        # 检查速度是否小于阈值
        if speed <= speed_threshold:
            current_stay.append(traj_points[i])
        else:
            # 如果当前停留点集非空，且时间窗口满足条件，则添加到停留点集
            if len(current_stay) > 0 and (traj_points[i][0] - current_stay[0][0]) >= time_window:
                mid_stay = current_stay[len(current_stay) // 2]
                mid_stay.append(traj_points[i][0] - current_stay[0][0])
                stay_points.append(mid_stay)
            # 重置当前停留点集
            current_stay = []

    # 检查最后一组点是否为停留点
    if len(current_stay) > 0 and (traj_points[-1][0] - current_stay[0][0]) >= time_window:
        mid_stay = current_stay[(len(current_stay) - 1) // 2]
        mid_stay.append(traj_points[-1][0] - current_stay[0][0])
        stay_points.append(mid_stay)

    return stay_points


def detect_stay(trace_path):
    # 用户停留识别
    # 计算每个时刻的用户运动速度
    # 连续10s的运动速度都小于0.5m/s则判断这个时段为停留
    with open(trace_path, 'r') as f:
        traces = json.load(f)

    all_stay = []
    for uid, trace in traces.items():
        stay_points = identify_stay_points(trace)
        all_stay.extend(stay_points)

    print('轨迹数：', len(traces), '停留数：', len(all_stay))
    return np.array(all_stay)


if __name__ == "__main__":
    base_dir = 'tourists_traces\\20240818'
    with open('camera_params.json', 'r') as f:
        cameras = json.load(f)

    # 时空聚类，产生时空聚集点
    for cam, vals in cameras.items():
        print(cam)
        # if cam != 'guierlu06-1':
        #     continue
        for i in range(0, len(vals['geotraces'])):
            trace = os.path.join(base_dir, cam, vals['geotraces'][i])
            stay_points = detect_stay(trace)

            out_path = os.path.join(base_dir, cam, vals['stay_clust'][i])
            sp_point_cls = st_cluster(stay_points, pic_path=vals['raster'])

            # df = pd.DataFrame(columns=['x', 'y', 't', 'duration', 'cls'], data=sp_point_cls)
            # df.to_csv(out_path)

            last_three_columns = [row[-3:] for row in sp_point_cls]
            stay_time_pattern(last_three_columns, os.path.join(base_dir, cam, vals['stay_pic_path'][i]), 90)
