"""
Convert each frame of user location data for multi-object tracking into trajectory data identified by personal ID
 (including image coordinate system and geographic coordinate system)
"""
import json
import math
import os.path

import pandas as pd

from TouristBehaviorPatternAnalysis.spatialMapping import trace2geo


def trace_filter(rec_1, rec_2):
    """
    轨迹后处理
    """
    s_rec1 = (rec_1[2] - rec_1[0]) * (rec_1[3] - rec_1[1])  #第一个bbox面积 = 长×宽
    s_rec2 = (rec_2[2] - rec_2[0]) * (rec_2[3] - rec_2[1])  #第二个bbox面积 = 长×宽
    sum_s = s_rec1 + s_rec2  #总面积
    left = max(rec_1[0], rec_2[0])  #交集左上角顶点横坐标
    right = min(rec_1[2], rec_2[2])  #交集右下角顶点横坐标
    bottom = max(rec_1[1], rec_2[1])  #交集左上角顶点纵坐标
    top = min(rec_1[3], rec_2[3])  #交集右下角顶点纵坐标

    if left >= right or top <= bottom:  #不存在交集的情况
        return 0
    else:
        inter = (right - left) * (top - bottom)  #求交集面积
        iou = (inter / (sum_s - inter)) * 1.0  #计算IOU
        print(iou)


def mot2trace(file_path, fps):
    """
    将MOT格式的数据转换为个人轨迹

    参数:
    MOT格式的每帧用户位置
    fps为视频的帧率，按照此帧率抽取每秒的用户轨迹
    """
    user_mots = pd.read_csv(file_path, header=None, sep=' ',
                            names=['frame', 'tid', 'left_x', 'left_y', 'right_x', 'right_y'])

    user_trace = {}
    for index, row in user_mots.iterrows():
        tid = str(row['tid'])
        if tid not in user_trace.keys():
            user_trace[tid] = []

        x = int((row['left_x'] + row['right_x']) / 2)
        y = int(row['right_y'])
        second = math.ceil(row['frame'] / fps)
        # second = row['frame'] / fps
        user_trace[tid].append((second, x, y))

    final_traces = {}
    for uid, utrace in user_trace.items():
        simple_trace = []
        times = []
        for trace in utrace:
            if trace[0] not in times:
                simple_trace.append(trace)
                times.append(trace[0])
        if len(simple_trace) >= 5:
            final_traces[uid] = simple_trace
    return final_traces


def write_traces(pixel_trace, geo_trace, file_path):
    print('存储轨迹...')
    dir_path = os.path.dirname(file_path)

    with open(os.path.join(dir_path, os.path.basename(file_path).split('.')[0] + '_trace.json'), 'w') as file:
        json.dump(pixel_trace, file, indent=4)
    file.close()

    with open(os.path.join(dir_path, os.path.basename(file_path).split('.')[0] + '_geotrace.json'), 'w') as file:
        json.dump(geo_trace, file, indent=4)

    file.close()


if __name__ == '__main__':
    base_path = 'tourists_traces\\20240818'
    camera_fps = {
        # 'guierlu04-2': 25,
        # 'guierlu06-1': 25,
        'wygc05': 25,
        # 'daqiongding': 25
    }
    for camera, fps in camera_fps.items():
        files = os.listdir(os.path.join(base_path, camera))
        for file in files:
            if '.txt' in file:
                print(file)
                mot_path = os.path.join(base_path, camera, file)
                point_trace = mot2trace(mot_path, fps)
                geo_tracks = trace2geo(point_trace, camera)
                write_traces(point_trace, geo_tracks, mot_path)
