"""
Conversion of pixel coordinates to geographic coordinates method
"""
import csv
import json

import numpy as np
import pandas as pd


class CoorTransfer:
    def __init__(self, camera):
        self.camera = camera
        self.H = self.read_homography_from_csv()
        self.M = self.read_Affine()
        self.print = False

    def get_geo_coors(self, p):
        # 计算校正后的图像中的坐标点 p_prime
        p_prime = self.transform_point(p)

        # 计算该点对应的地理坐标
        X, Y = self.image_to_geographic_coords(p_prime[0], p_prime[1])

        if self.print:
            print(f"校正后的坐标: {p_prime}")
            print(f"地理坐标: ({X}, {Y})")

        return X, Y

    def transform_point(self, p):
        """
        将原始图像中的点 p 通过单应性矩阵 H 转换到校正后的图像中的点。
        p: 原始图像中的点坐标 [x, y]
        H: 单应性矩阵
        """
        # 将点 p 扩展为齐次坐标形式 [x, y, 1]
        p_homogeneous = np.append(p, 1)

        # 应用单应性矩阵变换
        p_prime_homogeneous = np.dot(self.H, p_homogeneous)

        # 从齐次坐标转换为笛卡尔坐标
        # 注意：这里使用的是 np.linalg.inv(H) 来对 H 进行逆变换
        # 在实际应用中，你应该确保 H 是可逆的，否则你需要解决其他问题
        # p_prime = np.dot(np.linalg.inv(H), p_prime_homogeneous)[:-1]

        # 或者，如果不需要逆变换，可以直接除以齐次坐标的最后一个元素进行归一化
        p_prime = p_prime_homogeneous[:-1] / p_prime_homogeneous[-1]

        return p_prime

    def image_to_geographic_coords(self, x, y):
        """
        实现仿射变换，对于图像中的一个点 (x, y)，计算对应的地理坐标 (X, Y)
        仿射变换矩阵 M 由6个参数构成：tfw文件中的各参数的含义如下
        A 【X方向上的象素分辨素】 0.02
        D 【旋转系数】 0
        B 【旋转系统】 0
        E 【Y方向上的象素分辨率】 -0.02
        C 【栅格地图左上角象素中心X坐标】 438736.80798
        F 【栅格地图左上角象素中心Y坐标】 2471988.50468
        :param x:
        :param y:
        :param M:
        :return:
        """
        # 将点扩展为齐次坐标形式
        point = np.array([x, y, 1])
        # 应用仿射变换
        transformed_point = np.dot(self.M, point)
        # 直接使用变换后的坐标（不需要归一化）
        X = transformed_point[0]
        Y = transformed_point[1]
        return X, Y

    # 读取CSV文件并转换为NumPy数组
    def read_homography_from_csv(self):
        # 创建一个空列表来存储行数据
        homography_list = []

        # 打开CSV文件进行读取
        with open('spatial_reference/' + self.camera + '/homography_matrix.csv', 'r') as file:
            csv_reader = csv.reader(file)

            # 遍历CSV文件中的每一行
            for row in csv_reader:
                # 将每行数据转换为浮点数并添加到列表中
                homography_list.append([float(x) for x in row])

        # 将列表转换为NumPy数组
        H = np.array(homography_list)

        return H

    def read_Affine(self):
        # 读取仿射变换参数
        data = np.genfromtxt('spatial_reference/' + self.camera + '/affine_matrix.tfw', dtype=float)
        # 要转换成：M = np.array([[0.02, 0, 13217456.5231], [0, -0.02, 3751831.9221]])
        M = np.array([data[::2], data[1::2]])
        return M


def trace2geo(data, camera):
    """
    视频轨迹转为地理坐标
    :return:
    """
    trace_geo = {}

    trans = CoorTransfer(camera)
    for k, v in data.items():
        new_points = []
        for pp in v:
            t, x, y = pp[0], pp[1], pp[2]
            nx, ny = trans.get_geo_coors((x, y))
            new_points.append([t, nx, ny])
        trace_geo[k] = new_points
    return trace_geo


def point2trace(pth):
    """
    每帧坐标点数据转换为用户轨迹数据
    :param pth:
    :return:
    """
    pd_points = pd.read_csv(pth, sep=' ',
                            names=['frame', 'id', 'xl', 'yl', 'xr', 'yr'])
    if len(pd_points) > 0:
        plus = 10 - max(pd_points['frame']) % 10  # 补充数据到最后一帧为第150帧
        pd_points['frame'] = pd_points['frame'] + plus
        # pd_points['x'] += pd_points['w'] / 2
        # pd_points['y'] += pd_points['h']
        pd_points['x'] = (pd_points['xl'] + pd_points['xr']) / 2
        pd_points['y'] = pd_points['yr']

        points = pd_points[pd_points['frame'] % 5 == 0]

        traces = {}

        all_id = points['id'].unique()
        for id in all_id:
            id_points = points[points['id'] == id]

            for i, row in id_points.iterrows():
                if id not in traces.keys():
                    traces[id] = []
                traces[id].append([row['x'], row['y'], int(row['frame'] / 5),
                                   row['xl'], row['yl'], row['xr'], row['yr']])

        return traces
    else:
        return None

