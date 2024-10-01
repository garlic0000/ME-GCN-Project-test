import os
import shutil
import glob
from pathlib import Path
from tqdm import tqdm

import yaml
import pandas as pd


def get_img_sampling_count(anno_csv_path):
    """
    计算抽样图片数量
    是否应该计算注释文件中被标记的帧的数量？
    start_frame,apex_frame,end_frame
    理论上是注释文件中每一排*3==关键帧的数量
    但是有的帧没有检测出来 比如标记为0
    所以需要进一步对csv文件进行筛查
    """
    sum_count = 0
    for sub_item in Path(root_path).iterdir():
        if sub_item.is_dir():
            for type_item in sub_item.iterdir():
                if type_item.is_dir():
                    sum_count += 3
    return sum_count


def apex_sampling(opt):
    """
    视频帧抽样
    抽取关键帧 将关键帧放入另一个文件夹 用于下一步处理
    """
    try:
        # 原始数据路径 被分解为图片帧的视频
        # 原始数据路径是 图片
        original_root_path = opt["original_root_path"]
        # 保存采样的路径
        simpled_root_path = opt["simpled_root_path"]
        # 读取注释文件
        # original_anno_csv_path = opt["original_anno_csv_path"]
        anno_csv_path = opt["anno_csv_path"]
    except KeyError:
        # print(f"Dataset {dataset} does not need to be sampled")
        print("数据集 {} 不需要抽样".format(opt["dataset"]))
        print("feature_extraction/cas(me)^2/apex_sample.py 第35行")
        exit(1)

    if not os.path.exists(original_root_path):
        print(f"path {original_root_path} is not exist")
        exit(1)

    # sum_count = get_img_sampling_count(original_root_path)
    # print("img count = ", sum_count)

    if not os.path.exists(simpled_root_path):
        os.makedirs(simpled_root_path)
    df = pd.read_csv(anno_csv_path)
    print("打印列名")
    print(df.columns)
    # 使用抽样图片的数量做进度条
    with tqdm(total=sum_count) as tq:
        for sub_item in Path(original_root_path).iterdir():
            if not sub_item.is_dir():
                continue
            for type_item in sub_item.iterdir():
                if not sub_item.is_dir():
                    continue

                # 创建新的目录
                new_dir_path = os.path.join(simpled_root_path, sub_item.name,
                                            type_item.name)
                if not os.path.exists(new_dir_path):
                    os.makedirs(new_dir_path)
               
                img_path_list = glob.glob(
                    os.path.join(str(type_item), "*.jpg"))
                img_path_list.sort()

                label = df[(df['Filename']==type_item.name) & (df['Subject']==int(sub_item.name[3:5]))]
                apex = int(label['ApexFrame'])
                on = int(label['OnsetFrame'])
                off = int(label['OffsetFrame'])
                #import pdb;pdb.set_trace()
                if apex-on > off-apex:
                    sampling_index_list = [0, int((apex-on)/2), apex-on, off-on]
                else:
                    sampling_index_list = [0, apex-on, int((off-apex)/2), off-on]
                
                for i, sampling_index in enumerate(sampling_index_list):
                    shutil.copy(
                        img_path_list[sampling_index],
                        os.path.join(
                            new_dir_path, "img_" + str(i).zfill(5) + ".jpg"))
                    tq.update()

if __name__ == "__main__":

    with open("/kaggle/working/ME-GCN-Project/feature_extraction/config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
    apex_sampling(opt)
