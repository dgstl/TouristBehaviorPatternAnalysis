import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from boxmot import DeepOCSORT
from boxmot.utils.plots import colors

"""
Visitor target motion trajectory generation based on adaptive slicing
Use a format similar to MOT16 to store trace results
"""


def objTrack(in_file, out_path, slices, det=False, confidence=0.5, target_fps=5):
    """
    Multi-object tracking and recording of trajectories
    Args:
        in_file:
        out_path:
        slices:
        det:
        confidence:
    Returns:
    """
    # 加载目标检测模型和追踪器模型
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='weights/yolov8l.pt',
        confidence_threshold=confidence,
        device="cuda:0"
    )

    tracker = DeepOCSORT(  ## osnet_x1_0_market1501
        model_weights=Path('weights/osnet_x1_0_msmt17.pt'),  #  osnet_x0_25_msmt17  osnet_ain_x1_0_msmt17_MD
        device='cuda:0',
        det_thresh=confidence,  # 检测框的置信度阈值
        min_hits=3,  # 指定一个对象被跟踪的最小次数，卡尔曼滤波每更新一次，hit_streak就加一
        iou_threshold=0.3,  # 检测框和轨迹之间IOU的最小阈值
        # w_association_emb=0.2,
        w_association_emb=0.5,  # 关联嵌入权重
        fp16=False
    )

    inPath = Path(in_file)

    vid = cv2.VideoCapture(str(inPath.absolute()))
    v_height, v_width = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = vid.get(cv2.CAP_PROP_FPS)
    skip = fps / target_fps

    # 记录跟踪目标轨迹
    pred_tracks = []

    if det:
        # 视频文件写对象
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # 视频信息设置
        writer = cv2.VideoWriter(out_path + "/" + inPath.parent.name + '_' + str(inPath.name),
                                 fourcc,
                                 target_fps,
                                 (int(vid.get(3)), int(vid.get(4))),
                                 True)

    thickness = 2
    fontscale = 0.4
    cnt = 0  # 记录当前是第几帧

    # 处理每一帧
    while True:
        print(cnt)
        # 读取帧：ret是bool,表示是否成功捕获帧，im是捕获的帧
        ret, im = vid.read()

        # 若未捕获帧，则退出循环
        if not ret:
            break
        cnt += 1

        # 若满足跳帧的条件则该帧不处理
        if not (cnt % skip == 0):
            continue

        result = get_sliced_prediction(
            im,
            detection_model,
            slice_height=int((v_height // (slices[1]+1)) * 1.2),
            slice_width=int((v_width // (slices[0]+1)) * 1.2),
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=2
        )

        # filtered_objects = [obj for obj in result.object_prediction_list if obj.category.id == 0]
        all_objects = result.object_prediction_list

        # 检测到的目标数量
        num_predictions = len(all_objects)

        # 存储目标检测结果和跟踪所需的信息，每个目标用 6 个值来表示
        dets = np.zeros([num_predictions, 6], dtype=np.float32)
        # 遍历 object_prediction_list 中的每个目标检测结果，并同时获取其索引和值
        for ind, object_prediction in enumerate(all_objects):
            # 将当前目标检测结果的边界框坐标（左上角和右下角）转换为 [x_min, y_min, x_max, y_max] 的形式
            dets[ind, :4] = np.array(object_prediction.bbox.to_xyxy(), dtype=np.float32)
            # 这一行将当前目标检测结果的置信度分数存储在 dets 数组的当前行的第五个位置
            dets[ind, 4] = object_prediction.score.value
            # 这一行将当前目标检测结果的类别标签存储在 dets 数组的当前行的第六个位置
            dets[ind, 5] = object_prediction.category.id

        tracks = tracker.update(dets, im)  # --> (x, y, x, y, id, conf, cls, ind)

        if tracks.shape[0] != 0:
            xyxys = tracks[:, 0:4].astype('int')  # float64 to int
            ids = tracks[:, 4].astype('int')  # float64 to int
            # confs = tracks[:, 5].round(decimals=2)
            clss = tracks[:, 6].astype('int')  # float64 to int
            # inds = tracks[:, 7].astype('int')  # float64 to int

            # print bboxes with their associated id, cls and conf
            for xyxy, id, cls in zip(xyxys, ids, clss):  # , confs, clss
                if cls == 0:
                    pred_tracks.append([int(cnt - 1), int(id), xyxy[0], xyxy[1], xyxy[2], xyxy[3]])

                    if det:
                        im = cv2.rectangle(
                            im,
                            (xyxy[0], xyxy[1]),
                            (xyxy[2], xyxy[3]),
                            colors(id, True),
                            thickness
                        )
                        cv2.putText(
                            im,
                            f'id: {id}, c: {cls}',  # conf: {conf}, c: {cls},
                            (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontscale,
                            colors(id, True),
                            thickness
                        )

        if det:
            writer.write(im)

    # 轨迹保存
    np_pred_tracks = np.array(pred_tracks)
    np.savetxt(os.path.join(out_path, inPath.parent.name + '_' + str(inPath.name) + ".txt"), np_pred_tracks,
               fmt='%d')  # 将数组中数据写入到data.txt文件

    if det:
        writer.release()
    vid.release()


def get_all_mots():
    root_path = 'D:\\OriginVideos\\tourists_behavior\\20240818'
    out_path = 'tourists_traces\\20240818'

    with open('slice_eval\slice_config.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 跳帧跟踪即可
    for camera, slice in data.items():
        directory = os.path.join(root_path, camera)
        # 读取目录下的所有文件，并存储在列表中
        file_list = [os.path.join(directory, file) for file in os.listdir(directory) if
                     os.path.isfile(os.path.join(directory, file)) and not file.startswith('.')]

        run_log = []
        for file in file_list:
            print(file)
            start_time = time.time()
            save_path = os.path.join(out_path, camera)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            objTrack(file, save_path, slice, True, confidence=0.2, target_fps=5)
            end_time = time.time()
            runtime = end_time - start_time
            run_log.append(file + ' runs ' + str(runtime / 60) + 'minutes')

        np.savetxt(os.path.join(out_path, camera + '_log.txt'), np.array(run_log), fmt='%s')


if __name__ == '__main__':
    get_all_mots()
    # objTrack('D:\\OriginVideos\\tourists_behavior\\eval\\10_0.mp4', "tourists_traces\\2024-07-28\\eval",
    #          [614, 368], True, confidence=0.2, target_fps=5)
