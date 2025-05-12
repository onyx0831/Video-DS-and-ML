#import os
#import sys

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import preprocessing
from video_dataset import VideoDataset, TargetDataset
from utils.multi_dataset import MultiDataset
from utils.video_transform import VideoTransform
from ideo_dataloader import VideoCollator


def get_config(config_path):
    """
    Load the configuration file.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main():
    config = get_config("./Deep_Learning_with_Pytorch/training/train_config.yaml")
    input_data = config["preprocess"]["input_data"]
    df = pd.read_csv(input_data)
    seed = config["preprocess"]["seed"]
    test_size = config["preprocess"]["test_size"]
    label_column = config["preprocess"]["label_column"]

    train_df, eval_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True, stratify=df[label_column])
    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)
    
    feature_columns = config["preprocess"]["feature_columns"]
    train_df, le = preprocessing(train_df, label_col=label_column, feature_cols=feature_columns, inference=False)
    eval_df = preprocessing(eval_df, label_col=label_column, feature_cols=feature_columns, le=le, inference=False)

    model_type = config["model"]["use_model"]
    dataset_kwargs = dict(
        video_dir = config["preprocess"]["video_dir"],
        num_frames = config["model"][model_type]["num_frames"],
        interval_sec = config["preprocess"]["interval_sec"],
        transform = VideoTransform(
            config["model"][model_type]["image_size"],
            config["model"][model_type]["image_mean"],
            config["model"][model_type]["image_std"],
        ),
        return_frame_mask = config["model"][model_type]["padding_mask"],
    )
    train_feature_dataset = VideoDataset(train_df, **dataset_kwargs)
    eval_feature_dataset = VideoDataset(eval_df, **dataset_kwargs)
    train_target_dataset = TargetDataset(train_df)
    eval_target_dataset = TargetDataset(eval_df)

    train_dataset = MultiDataset(
        [train_feature_dataset, train_target_dataset],
        extractor=lambda x: {**x[0], **x[1]},
    )

    eval_dataset = MultiDataset(
        [eval_feature_dataset, eval_target_dataset], extractor=lambda x: {**x[0], **x[1]}
    )
    # print(f"train_dataset: {len(train_dataset)}")
    # print(f"eval_dataset: {len(eval_dataset)}")
    # print(f"train_dataset_one: {train_dataset[0]}")
    # print(f"eval_dataset_one: {eval_dataset[0]}")

    video_collator = VideoCollator()



if __name__ == "__main__":
    main()