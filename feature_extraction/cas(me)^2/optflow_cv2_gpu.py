import os
import cv2
import numpy as np
import glob
import yaml
from pathlib import Path
from tqdm import tqdm


# 使用 cv2.cuda 版本来计算光流
def calculate_tvl1_optical_flow_gpu(frame1, frame2):
    # 使用 GPU 版的 TVL1 光流
    tvl1 = cv2.cuda_OpticalFlowDual_TVL1.create()

    # 将图像上传到 GPU
    gpu_frame1 = cv2.cuda_GpuMat()
    gpu_frame2 = cv2.cuda_GpuMat()
    gpu_frame1.upload(frame1)
    gpu_frame2.upload(frame2)

    # 计算光流
    gpu_flow = tvl1.calc(gpu_frame1, gpu_frame2, None)

    # 将光流从 GPU 下载到 CPU
    flow = gpu_flow.download()

    return flow


def save_flow_to_image(flow, output_path, frame_index):
    flow_x, flow_y = flow[..., 0], flow[..., 1]
    flow_x_normalized = cv2.normalize(flow_x, None, 0, 255, cv2.NORM_MINMAX)
    flow_y_normalized = cv2.normalize(flow_y, None, 0, 255, cv2.NORM_MINMAX)
    flow_x_img = flow_x_normalized.astype(np.uint8)
    flow_y_img = flow_y_normalized.astype(np.uint8)
    cv2.imwrite(os.path.join(output_path, f"flow_x_{frame_index:05d}.jpg"), flow_x_img)
    cv2.imwrite(os.path.join(output_path, f"flow_y_{frame_index:05d}.jpg"), flow_y_img)


def process_optical_flow_for_dir(input_dir, output_dir, opt_step=1):
    image_list = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
    if len(image_list) < 2:
        print(f"Not enough images in {input_dir} to compute optical flow")
        return

    frame_index = 0
    prev_frame = cv2.imread(image_list[0], cv2.IMREAD_GRAYSCALE)

    for i in range(opt_step, len(image_list), opt_step):
        frame = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)
        flow = calculate_tvl1_optical_flow_gpu(prev_frame, frame)
        save_flow_to_image(flow, output_dir, frame_index)
        frame_index += 1
        prev_frame = frame


def get_dir_count(root_path):
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
                new_sub_dir_path = os.path.join(optflow_root_path, sub_item.name)
                if not os.path.exists(new_sub_dir_path):
                    os.makedirs(new_sub_dir_path)
                for type_item in sub_item.iterdir():
                    if type_item.is_dir():
                        print(f"Processing optical flow for {type_item}")
                        process_optical_flow_for_dir(str(type_item), new_sub_dir_path, opt_step)
                        tq.update()


if __name__ == "__main__":
    with open("/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/config_test.yaml", encoding="UTF-8") as f:
        # with open("/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
    optflow(opt)
