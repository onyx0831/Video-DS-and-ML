import os
#import sys
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

from preprocessing import preprocessing
from video_dataset import VideoDataset, TargetDataset
from utils.multi_dataset import MultiDataset
from utils.video_transform import VideoTransform
from utils import tools
from video_dataloader import VideoCollator
from video_encoder import Encoder
from video_trainer import custom_compute_metrics, SaveEvalPrediction, save_training_history


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
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
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
    num_class = train_df['label'].nunique()

    """
    
    loss_name = config["train"]["loss"]
    if loss_name == "mse":
        loss_kwargs = dict(
            loss_fct=nn.MSELoss(), last_activation=nn.Sigmoid(), post_loss_process=None
        )
    elif loss_name == "bce":
        loss_kwargs = dict(
            loss_fct=nn.BCEWithLogitsLoss(),
            last_activation=None,
            post_loss_process=nn.Sigmoid(),
        )
    elif loss_name == "ce":
        loss_kwargs = dict(
            loss_fct=nn.CrossEntropyLoss(),
            last_activation=None,
            post_loss_process=nn.Softmax(),
        )
    else:
        loss_kwargs = {}
    """
    loss_kwargs = dict(
        loss_fct=nn.CrossEntropyLoss(),
        last_activation=None,
        post_loss_process=nn.Softmax(),
    )
    model_kwargs = dict(
        model_name=model_type,
        pretrained=True,
        model_kwargs=dict(config["model"][model_type]["model_kwargs"]),
        output_dim=num_class,
    )
    train_kwargs = dict(**loss_kwargs, **model_kwargs)
    output_dir = config["train"]["output_dir"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model = Encoder(**train_kwargs)

    # 学習・検証に関するパラメータの設定:詳しくはtransformers.trainerを参照
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config["train"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["train"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["train"]["gradient_accumulation_steps"],
        evaluation_strategy=config["train"]["evaluation_strategy"],
        logging_strategy=config["train"]["evaluation_strategy"],
        save_strategy=config["train"]["evaluation_strategy"],
        logging_steps=config["train"]["eval_steps"],
        save_steps=config["train"]["eval_steps"],
        eval_steps=config["train"]["eval_steps"],
        learning_rate=config["train"]["lr"],
        lr_scheduler_type=config["train"]["lr_scheduler_type"],
        weight_decay=config["train"]["weight_decay"],
        warmup_steps=config["train"]["warmup_steps"],
        overwrite_output_dir=True,
        save_total_limit=1,
        label_names=["targets"],
        num_train_epochs=config["train"]["num_train_epochs"],
        load_best_model_at_end=True,
        disable_tqdm=config["train"]["disable_trainer_tqdm"],
        fp16=(config["train"]["predicsion_type"] == "fp16"),
        bf16=(config["train"]["predicsion_type"] == "bf16"),
        dataloader_num_workers=config["train"]["num_workers"],
        save_safetensors=False,
        report_to="none",
    ) 

    trainer = Trainer(
        model=model,
        data_collator=video_collator,
        compute_metrics=custom_compute_metrics,#SaveEvalPrediction(custom_compute_metrics, output_dir=output_dir),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config["train"]["early_stopping_patience"])
        ],
    )

    trainer.train(
        ignore_keys_for_eval=[
            "encoded_state",
            "last_hidden_state",
            "attentions",
        ]
    )

    # 学習結果、学習済モデルの保存
    trainer.save_state()  # trainer_state.json
    trainer.save_model()  # pytorch_model.bin
    history_df = save_training_history(output_dir + "/trainer_state.json")
    history_df.to_csv(output_dir + "/training_history.csv", index=False)

    # 学習曲線のグラフの作成・保存
    save_dir = os.path.join(output_dir, "training_figs")
    os.makedirs(save_dir, exist_ok=True)
    # プロットに失敗しても止まらないようにtry句を使う
    try:
        tools.plot_learning_curves(history_df, save_dir)
    except Exception as e:
        message = f"Failed to make training curves; {str(e)}"
        print(message)
    print("Training finished.")


if __name__ == "__main__":
    main()
