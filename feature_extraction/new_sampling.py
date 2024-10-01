import os
import shutil
import glob
from pathlib import Path
from tqdm import tqdm

import yaml
import pandas as pd


def get_img_sampling_count(root_path, sampling_ratio):
    sum_count = 0
    for sub_item in Path(root_path).iterdir():
        if sub_item.is_dir():
            for type_item in sub_item.iterdir():
                if type_item.is_dir():
                    img_path_list = glob.glob(
                        os.path.join(str(type_item), "*.jpg"))
                    sample_count = int(len(img_path_list) / sampling_ratio)
                    if len(img_path_list) - sample_count * sampling_ratio != 0:
                        sample_count += 1
                    sum_count += sample_count
    return sum_count


def sampling(opt):
    try:
        original_root_path = opt["original_root_path"]
        simpled_root_path = opt["simpled_root_path"]
        # original_anno_csv_path = opt["original_anno_csv_path"]
        # anno_csv_path = opt["anno_csv_path"]
        SAMPLING_RATIO = opt["SAMPLING_RATIO"]
    except KeyError:
        print(f"Dataset {dataset} does not need to be sampled")
        print("terminate")
        exit(1)

    if not os.path.exists(original_root_path):
        print(f"path {original_root_path} is not exist")
        exit(1)

    sum_count = get_img_sampling_count(original_root_path, SAMPLING_RATIO)
    print("img count = ", sum_count)

    if not os.path.exists(simpled_root_path):
        os.makedirs(simpled_root_path)

    # sampling img
    with tqdm(total=sum_count) as tq:
        for sub_item in Path(original_root_path).iterdir():
            if not sub_item.is_dir():
                continue
            for type_item in sub_item.iterdir():
                if not sub_item.is_dir():
                    continue

                new_dir_path = os.path.join(simpled_root_path, sub_item.name,
                                            type_item.name)
                if not os.path.exists(new_dir_path):
                    os.makedirs(new_dir_path)

                img_path_list = glob.glob(
                    os.path.join(str(type_item), "*.jpg"))
                img_path_list.sort()

                sample_count = int(len(img_path_list) / SAMPLING_RATIO)
                if len(img_path_list) - sample_count * SAMPLING_RATIO != 0:
                    sample_count += 1

                sampling_index_list = [
                    i * SAMPLING_RATIO for i in range(sample_count)]

                for i, sampling_index in enumerate(sampling_index_list):
                    shutil.copy(
                        img_path_list[sampling_index],
                        os.path.join(
                            new_dir_path, "img_" + str(i).zfill(5) + ".jpg"))
                    tq.update()

    # sampling annotaion file
    # df = pd.read_csv(original_anno_csv_path)
    # df["start_frame"] = df["start_frame"] // SAMPLING_RATIO
    # df["apex_frame"] = df["apex_frame"] // SAMPLING_RATIO
    # df["end_frame"] = df["end_frame"] // SAMPLING_RATIO
    # df.to_csv(anno_csv_path, index=False)


if __name__ == "__main__":

    with open("/kaggle/working/ME-GCN-Project/feature_extraction/config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
    sampling(opt)
