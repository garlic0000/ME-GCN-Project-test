import os
import glob
import csv
from pathlib import Path

from tqdm import tqdm
import numpy as np
import cv2
import yaml

from feature_tools import calculate_roi_freature_list


def min_max_normalize(data):
    """
    Min-Max线性归一化
    将输入映射到(0, 1)
    """
    min_val = np.min(data)
    max_val = np.max(data)
    epsilon = 1e-8  # 防止除零
    normalized_data = (data - min_val) / (max_val - min_val + epsilon)  # 避免除零
    return normalized_data


def sigmoid_normalize(data):
    """
    Sigmoid归一化
    Sigmoid函数将输入映射到(0, 1)区间
    """
    return 1 / (1 + np.exp(-data))


def tanh_normalize(data):
    """
    Tanh归一化
    Tanh函数将输入映射到(-1, 1)区间，但可以通过简单的变换将其映射到(0, 1)
    """
    return (np.tanh(data) + 1) / 2


def get_flow_x_y(frame1, frame2):
    # 使用 TV-L1 算法创建光流计算器
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    # 计算光流
    flow = tvl1.calc(frame1, frame2, None)
    # 将光流的x和y分量分开
    flow_x, flow_y = flow[..., 0], flow[..., 1]
    flow_x = flow_x.astype(np.float32)
    flow_y = flow_y.astype(np.float32)
    flow_x_y = np.stack((flow_x, flow_y), axis=-1)
    # Min-Max线性归一化
    flow_x_y = min_max_normalize(flow_x_y) - 0.5
    return flow_x_y


def get_original_flow(frame1, frame2):
    """
    获取原始光流
    """
    # 使用 TV-L1 算法创建光流计算器
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    # 计算光流
    flow = tvl1.calc(frame1, frame2, None)
    # # 将光流的x和y分量分开
    # flow_x, flow_y = flow[..., 0], flow[..., 1]
    # flow_x = flow_x.astype(np.float32)
    # flow_y = flow_y.astype(np.float32)
    # flow_x_y = np.stack((flow_x, flow_y), axis=-1)
    return flow


# def scale_flow_to_range(flow):
#     """
#     将原始光流的幅度缩放到 [0, 0.5] 范围，并保持方向信息。
#
#     Args:
#         flow: np.ndarray, shape of flow should be (-1, 2)
#
#     Returns:
#         scaled_flow: 缩放后的光流向量
#     """
#     assert flow.dtype == np.float32, "Element type of optical flow should be float32"
#
#     # Step 1: 计算每个光流向量的幅度
#     magnitudes = np.linalg.norm(flow, axis=1, keepdims=True)  # 幅度 (n, 1)
#     magnitudes = np.clip(magnitudes, 1e-6, None)  # 避免出现零值，避免除零错误
#
#     # Step 2: 对光流方向进行归一化
#     normalized_flow = flow / magnitudes  # 方向单位向量 (n, 2)
#
#     # Step 3: 对幅度进行缩放，将其映射到 [0, 0.5]
#     max_magnitude = np.max(magnitudes)
#     min_magnitude = np.min(magnitudes)
#
#     if max_magnitude > min_magnitude:
#         # 将幅度缩放到 [0, 1]，然后再缩放到 [0, 0.5]
#         scaled_magnitudes = (magnitudes - min_magnitude) / (max_magnitude - min_magnitude)
#         scaled_magnitudes = scaled_magnitudes * 0.5
#     else:
#         # 如果 max_magnitude 和 min_magnitude 相等，设为 0
#         scaled_magnitudes = np.zeros_like(magnitudes)
#
#     # # Step 4: 恢复光流，按缩放后的幅度恢复光流向量
#     scaled_flow = normalized_flow * scaled_magnitudes  # 方向 * 缩放后的幅度
#     return scaled_flow


def normalize_flow_to_range(flow):
    """
    将光流的 x 和 y 分量归一化到 [0, 1]，然后平移到 [-0.5, 0.5] 范围。

    Args:
        flow: 光流数据, np.ndarray, shape of flow should be (H, W, 2)

    Returns:
        flow_x_y_normalized: 归一化并平移后的光流 x 和 y 分量
    """
    # Step 1: 获取光流的 x 和 y 分量
    flow_x = flow[..., 0]  # 光流的 x 分量
    flow_y = flow[..., 1]  # 光流的 y 分量

    # Step 2: 计算每个像素的光流模，并获取模的最大值
    magnitudes = np.linalg.norm(flow, axis=2)  # 计算光流的模 (H, W)
    max_magnitude = np.max(magnitudes)  # 取光流模的最大值

    if max_magnitude == 0:
        raise ValueError("Max magnitude of the flow is zero, normalization not possible.")

    # Step 3: 将光流的 x 和 y 分量归一化到 [0, 255]
    flow_x_normalized_255 = ((flow_x + max_magnitude) / (2 * max_magnitude)) * np.float32(255)
    flow_y_normalized_255 = ((flow_y + max_magnitude) / (2 * max_magnitude)) * np.float32(255)
    # 可以保存x y 灰度图像 暂时不保存

    # Step 4: 进一步归一化到 [0, 1]，再平移到 [-0.5, 0.5]
    flow_x_y_normalized_255 = np.stack((flow_x_normalized_255, flow_y_normalized_255), axis=2)
    flow_x_y_normalized = (flow_x_y_normalized_255 / np.float32(255)) - 0.5  # 归一化并平移

    return flow_x_y_normalized



def get_croped_count(root_path):
    count = 0
    for sub_item in Path(root_path).iterdir():
        if sub_item.is_dir():
            for type_item in sub_item.iterdir():
                if type_item.is_dir():
                    # 光流会少一张图片
                    # 也有可能是 - opt_step
                    count += len(glob.glob(os.path.join(
                        str(type_item), "*.jpg")))
    return count


def optflow_feature(opt):
    cropped_root_path = opt["cropped_root_path"]
    feature_root_path = opt["feature_root_path"]
    landmark_root_path = opt["cropped_root_path"]
    # anno_csv_path = opt["anno_csv_path"]
    print(f'dataset: {opt["dataset"]}')
    sum_count = get_croped_count(cropped_root_path)
    print("cropped count = ", sum_count)

    opt_step = 1  # int(get_micro_expression_average_len(anno_csv_path) // 2)
    print(f"opt_step: {opt_step}")

    # for debug use
    # short_video_list = []
    with tqdm(total=sum_count) as tq:
        for sub_item in Path(cropped_root_path).iterdir():
            if not sub_item.is_dir():
                continue
            for type_item in sub_item.iterdir():
                if not type_item.is_dir():
                    continue
                image_list = sorted(glob.glob(os.path.join(str(type_item), "*.jpg")))
                csv_landmark_path = os.path.join(
                    landmark_root_path,
                    sub_item.name, type_item.name, "landmarks.csv")
                if not os.path.exists(csv_landmark_path):
                    print("\n")
                    print(f"{csv_landmark_path} does not exist")
                    continue
                prev_frame = cv2.imread(image_list[0], cv2.IMREAD_GRAYSCALE)
                with open(csv_landmark_path, 'r') as f:
                    roi_feature_list_sequence = []  # feature in whole video
                    csv_r = list(csv.reader(f))
                    for index, row in enumerate(csv_r):
                        if index < opt_step:
                            # 用于测试
                            # print("index < opt_step")
                            # print(index, opt_step, row)
                            continue
                        frame = cv2.imread(image_list[index], cv2.IMREAD_GRAYSCALE)
                        # flow_x_y = get_flow_x_y(prev_frame, frame)
                        # 获取原始光流
                        flow = get_original_flow(prev_frame, frame)
                        # 进行缩放
                        flow_x_y = normalize_flow_to_range(flow)
                        prev_frame = frame
                        landmarks = np.array(
                            [(int(row[index]), int(row[index + 68]))
                             for index in range(int(len(row) // 2))])
                        try:
                            # radius=5 从面部关键点半径为5的区域提取感兴趣区域ROI
                            roi_feature_list = calculate_roi_freature_list(
                                flow_x_y, landmarks, radius=5)
                            roi_feature_list_sequence.append(
                                np.stack(roi_feature_list, axis=0))
                            tq.update()
                        except Exception as exp:
                            roi_feature_list_sequence = []
                            print("roi_feature_list 有问题")
                            print(f"{sub_item.name}  {type_item.name}")
                            # 打印异常信息
                            print(str(exp))
                            break
                    if len(roi_feature_list_sequence) > 0:
                        new_type_dir_path = os.path.join(
                            feature_root_path, sub_item.name)
                        if not os.path.exists(new_type_dir_path):
                            os.makedirs(new_type_dir_path)
                        np.save(os.path.join(
                            new_type_dir_path, f"{type_item.name}.npy"),
                            np.stack(roi_feature_list_sequence, axis=0))


if __name__ == "__main__":
    # with open("/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/config.yaml", encoding="UTF-8") as f:
    with open("/kaggle/input/optflow-yaml/config_optflow_cv2.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
    optflow_feature(opt)
