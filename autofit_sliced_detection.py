import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from boxmot.utils.plots import colors
from TouristBehaviorPatternAnalysis.precision_assess import calculate_precision
from TouristBehaviorPatternAnalysis.precision_assess import calculate_metrics

"""
Different slicing protocols were generated, and the optimal slicing protocol was determined 
based on the evaluation of object detection accuracy.
"""


def detect_print(img, bboxes, scores, fname):
    cnt = 0
    for xyxy, conf in zip(bboxes, scores):  # , confs, clss
        img = cv2.rectangle(
            img,
            (int(xyxy[0]), int(xyxy[1])),
            (int(xyxy[2]), int(xyxy[3])),
            colors(cnt, True),
            thickness=2
        )
        # cv2.putText(
        #     img,
        #     f'id: {cnt}, c: {conf}',  # conf: {conf}, c: {cls},
        #     (int(xyxy[0]), int(xyxy[1]) - 10),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.4,
        #     colors(cnt, True),
        #     thickness=2
        # )
        cnt += 1
    # fname = in_file.split('.')[0] + '_' + str(wh[0]) + '_' + str(wh[1]) + '.png'
    cv2.imwrite(fname, img)


def obj_detect(in_file, confidence=0.2, wh=(), slices=[], det_write=False):
    """
    基于切片的目标检测
    Args:
        in_file:
        slices:
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

    # 读取帧：ret是bool,表示是否成功捕获帧，im是捕获的帧
    im = cv2.imread(in_file)

    result = get_sliced_prediction(
        im,
        detection_model,
        slice_height=slices[1],
        slice_width=slices[0],
        overlap_height_ratio=slices[3],
        overlap_width_ratio=slices[2],
        auto_slice_resolution=False,
        # slice_export_prefix='s-',
        # slice_dir='./weights',
        verbose=1
    )

    filtered_objects = [obj for obj in result.object_prediction_list if obj.category.id == 0]
    # all_objects = result.object_prediction_list
    bboxes, scores = [], []
    for predict_obj in filtered_objects:
        bbox = predict_obj.bbox
        xyxys = bbox.to_xyxy()
        xyxys.append(0)
        bboxes.append(xyxys)
        scores.append(round(predict_obj.score.value, 2))

    # 检测到的目标数量
    num_predictions = len(filtered_objects)
    print('预测检测框数量：', num_predictions)

    if det_write:
        fname = in_file.split('.')[0] + '_' + str(wh[0]) + '_' + str(wh[1]) + '.png'
        detect_print(im, bboxes, scores, fname)

    return bboxes


def calculate_splits(image_path, horizontal_splits, vertical_splits):
    """
    计算切片方案
    Args:
        image_path:
        horizontal_splits:
        vertical_splits:

    Returns:

    """
    image = Image.open(image_path)
    width, height = image.size

    # 计算每个分割框的宽度和高度
    split_width = width // (vertical_splits + 1)
    split_height = height // (horizontal_splits + 1)

    return split_width, split_height


def read_gt_data(fp):
    """
    读取地面真实数据
    Args:
        fp:

    Returns:

    """
    with open(fp, 'r', encoding='utf-8') as file:
        data = json.load(file)['shapes']
        results = []
        for i in range(0, len(data)):
            points = data[i]['points']
            lx, ly = points[0][0], points[0][1]
            rx, ry = points[2][0], points[2][1]
            results.append([lx, ly, rx, ry, 0])

        return results


def main(base_path, img_dir, out_path):
    # 指定图片路径
    file_list = os.listdir(os.path.join(base_path, img_dir))

    image_files = []
    for f in file_list:
        if f.endswith('.png') or f.endswith('.jpg'):
            image_files.append(os.path.join(base_path, img_dir, f))

    # 指定横线和竖线的分割数
    horizontal_splits = range(0, 6)  # 横线分割数
    vertical_splits = range(0, 6)  # 竖线分割数

    slices = {}
    acc_results_full = {}
    acc_results_val = {}

    for i in horizontal_splits:
        for j in vertical_splits:
            # if i < 5 or j < 5:
            #     continue
            accuracy = {}

            for image_path in image_files:
                # 计算分割框的宽度和高度
                split_width, split_height = calculate_splits(image_path, i, j)

                if j == 0:
                    if i == 0:  # 水平、垂直都切分0片
                        slices[(i, j)] = [split_width, split_height, 0, 0]
                    else:  # 垂直不切分，水平切分
                        # slices[(i, j)] = [split_width, split_height, 0, 0.2]
                        slices[(i, j)] = [int(split_width), int(split_height * 1.2), 0, 0.2]
                else:
                    if i == 0:  # 水平不切分，垂直切分
                        # slices[(i, j)] = [split_width, split_height, 0.2, 0]
                        slices[(i, j)] = [int(split_width * 1.2), int(split_height * 1.2), 0.2, 0]
                    else:  # 水平和垂直都切分
                        slices[(i, j)] = [int(split_width * 1.2), int(split_height * 1.2), 0.2, 0.2]
                        # slices[(i, j)] = [split_width, split_height, 0.2, 0.2]

                print(i, j, "分割框宽度和高度分别为:", split_width, split_height)

                # 检测个人目标位置
                detected_boxes = obj_detect(image_path, 0.2, (i, j), slices[(i, j)], False)

                filename, extension = os.path.splitext(image_path)
                ground_truth_boxes = read_gt_data(filename + '.json')
                print('真实检测框数量：', len(ground_truth_boxes))
                # detect_print(cv2.imread(image_path), ground_truth_boxes, [1] * len(ground_truth_boxes),
                #              image_path.split('.')[0] + '_gt.png')

                # TP, acc, matched_boxes, unmatched_boxes = (
                #     calculate_metrics(ground_truth_boxes, detected_boxes, 0, 0.5))
                TP, acc, matched_boxes, unmatched_boxes = calculate_precision(detected_boxes, ground_truth_boxes)
                # detect_print(cv2.imread(image_path), matched_boxes, [1] * len(matched_boxes),
                #              image_path.split('.')[0] + '_' + str(i) + '_' + str(j) + '_matched.png')
                # detect_print(cv2.imread(image_path), unmatched_boxes, [1] * len(unmatched_boxes),
                #              image_path.split('.')[0] + '_' + str(i) + '_' + str(j) + '_unmatched.png')

                accuracy[image_path] = acc
                print(image_path, '成功匹配的检测框数量', TP, '精度：', acc)

            acc_results_full[str(i) + '-' + str(j)] = accuracy
            acc_results_val[str(i) + '-' + str(j)] = sum(accuracy.values()) / len(accuracy)

    # 使用 json.dump() 将字典写入到文件
    with open(os.path.join(out_path, img_dir + '_slices_precision_full.json'), 'w') as file:
        json.dump(acc_results_full, file)

    with open(os.path.join(out_path, img_dir + '_slices_precision_val.json'), 'w') as file:
        json.dump(acc_results_val, file)


def result_plot():
    base_path = 'slice_eval/accs/'
    files = os.listdir(base_path)

    for f in files:
        if 'val' in f:
            with open(os.path.join(base_path, f), 'r') as file:
                data = json.load(file)

            x_values = set()
            y_values = set()
            for key in data.keys():
                x, y = key.split('-')
                x_values.add(x)
                y_values.add(y)

            x_unique = sorted(list(x_values))
            y_unique = sorted(list(y_values))

            heatmap_data = np.zeros((len(y_unique), len(x_unique)))

            for key, value in data.items():
                x, y = key.split('-')
                x_index = x_unique.index(x)
                y_index = y_unique.index(y)
                heatmap_data[x_index, y_index] = value

            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(heatmap_data, annot=True, cmap='coolwarm',
                             xticklabels=x_unique, yticklabels=y_unique, annot_kws={'fontsize': 14})

            plt.xlabel('X')
            plt.ylabel('Y')

            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_label_position('top')

            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            plt.tight_layout()
            file_name = f.split('_')[0] + '.jpg'
            print(file_name)
            plt.savefig(os.path.join(base_path, file_name), dpi=300)


if __name__ == '__main__':
    # base_path = 'slice_eval/imgs'
    # img_dirs = os.listdir(base_path)
    # for img_dir in img_dirs:
    #     main(base_path, img_dir, 'slice_eval/accs')

    result_plot()
    # main('slice_eval/imgs', '无忧广场05', 'slice_eval/test')
