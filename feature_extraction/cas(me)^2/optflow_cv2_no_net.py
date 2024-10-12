import os
import cv2
import numpy as np
import glob
import yaml
from pathlib import Path
from tqdm import tqdm


def calculate_tvl1_optical_flow(frame1, frame2):
    # 使用 TV-L1 算法创建光流计算器
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    # 计算光流
    flow = tvl1.calc(frame1, frame2, None)

    return flow


# def save_flow_to_image(flow, output_path, frame_index):
#     # 将光流的x和y分量分开
#     flow_x, flow_y = flow[..., 0], flow[..., 1]
#
#     # 归一化到 0~255 之间
#     flow_x_normalized = cv2.normalize(flow_x, None, 0, 255, cv2.NORM_MINMAX)
#     flow_y_normalized = cv2.normalize(flow_y, None, 0, 255, cv2.NORM_MINMAX)
#
#     # 转换为8位图像
#     flow_x_img = flow_x_normalized.astype(np.uint8)
#     flow_y_img = flow_y_normalized.astype(np.uint8)
#
#     # 保存为图像文件
#     cv2.imwrite(os.path.join(output_path, f"flow_x_{frame_index:05d}.jpg"), flow_x_img)
#     cv2.imwrite(os.path.join(output_path, f"flow_y_{frame_index:05d}.jpg"), flow_y_img)

# def save_flow_to_npy(flow, output_path, frame_index):
#     # 将光流的x和y分量分开
#     flow_x, flow_y = flow[..., 0], flow[..., 1]
#
#     # 保留原始的光流数据为 float32 格式
#     # 这样存储数据太大 9655张图片提取的光流npy为5.77G
#     flow_x = flow_x.astype(np.float32)
#     flow_y = flow_y.astype(np.float32)
#
#     # 将 flow_x 和 flow_y 数据合并为一个 NumPy 数组
#     # axis = -1 是在stack中添加的新维度为最后一个维度
#     # (H, W) ----> (H, W, 2)  新维度为2
#     flow_x_y = np.stack((flow_x, flow_y), axis=-1)
#     flow_x_y = flow_x_y / np.float32(255) - 0.5
#     # 保存为 NumPy `.npy` 文件
#     np.save(os.path.join(output_path, f"flow_{frame_index:05d}.npy"), flow_x_y)

def save_flow_to_npz(flow, output_path, frame_index):
    # 将光流的x和y分量分开
    flow_x, flow_y = flow[..., 0], flow[..., 1]

    # 保留原始的光流数据为 float32 格式
    # 这样存储数据太大 9655张图片提取的光流npy为5.77G
    flow_x = flow_x.astype(np.float32)
    flow_y = flow_y.astype(np.float32)

    # 将 flow_x 和 flow_y 数据合并为一个 NumPy 数组
    # axis = -1 是在stack中添加的新维度为最后一个维度
    # (H, W) ----> (H, W, 2)  新维度为2
    flow_x_y = np.stack((flow_x, flow_y), axis=-1)
    flow_x_y = flow_x_y / np.float32(255) - 0.5
    # 保存为压缩的 NumPy `.npz` 文件
    np.savez_compressed(os.path.join(output_path, f"flow_{frame_index:05d}.npz"), flow_x_y=flow_x_y)


def process_optical_flow_for_dir(input_dir, output_dir, opt_step=1):
    # 排序
    # 读取裁剪的图片
    image_list = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
    if len(image_list) < 2:
        print(f"Not enough images in {input_dir} to compute optical flow")
        return
    frame_index = 0
    prev_frame = cv2.imread(image_list[0], cv2.IMREAD_GRAYSCALE)

    for i in range(opt_step, len(image_list), opt_step):
        frame = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)
        flow = calculate_tvl1_optical_flow(prev_frame, frame)
        save_flow_to_npz(flow, output_dir, frame_index)
        frame_index += 1
        prev_frame = frame


def get_dir_count(root_path):
    """
    计算需要处理的已裁剪的图片数
    """
    count = 0
    for sub_item in Path(root_path).iterdir():
        if sub_item.is_dir():
            for type_item in sub_item.iterdir():
                if type_item.is_dir():
                    if len(glob.glob(
                            os.path.join(str(type_item), "*.jpg"))) > 0:
                        count += 1
    return count


def optflow(opt):
    cropped_root_path = opt["cropped_root_path"]
    optflow_root_path = opt["optflow_root_path"]

    if not os.path.exists(cropped_root_path):
        print(f"path {cropped_root_path} does not exist")
        exit(1)
    if not os.path.exists(optflow_root_path):
        os.makedirs(optflow_root_path)

    dir_count = get_dir_count(cropped_root_path)
    print("flow count = ", dir_count)

    opt_step = 1

    with tqdm(total=dir_count) as tq:
        for sub_item in Path(cropped_root_path).iterdir():
            if sub_item.is_dir():
                for type_item in sub_item.iterdir():
                    if type_item.is_dir():
                        new_sub_dir_path = os.path.join(optflow_root_path, sub_item.name, type_item.name)
                        if not os.path.exists(new_sub_dir_path):
                            os.makedirs(new_sub_dir_path)
                        print(f"Processing optical flow for {type_item}")
                        process_optical_flow_for_dir(str(type_item), new_sub_dir_path, opt_step)
                        tq.update()


if __name__ == "__main__":
    # with open("/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/config.yaml", encoding="UTF-8") as f:
    with open("/kaggle/input/optflow-yaml/config_optflow_cv2.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
    optflow(opt)
