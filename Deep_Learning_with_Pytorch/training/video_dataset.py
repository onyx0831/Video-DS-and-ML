import os
from typing import Callable
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.video_transform import VideoTransform
from utils.load_video_tensor import load_video_tensor

# video_idカラムから動画を取得し、torch.tensorに変換する
class VideoDataset(Dataset):
    def __init__(
            self, 
            df: pd.DataFrame, 
            video_dir: str, 
            num_frames: int = 16, 
            interval_sec: int = 2,
            transform: Callable = VideoTransform(
                224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),
            return_frame_mask: bool = False
    ):
        """
        Args:
            df:
            video_dir:動画ファイルが格納されているディレクトリ
            num_frames:モデルが許容するフレーム数
            interval_sec:獲得するフレームの間隔を決める秒数
            transform:画像の前処理を行う関数
            return_frame_mask:パディングしたフレームのマスクを返すかどうか
        """
        self.df = df
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.interval_sec = interval_sec
        self.transform = transform
        self.return_frame_mask = return_frame_mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        video_id = self.df.iloc[idx]["video_id"]
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            raise ValueError(f"Video file {video_path} does not exist.")
        
        video_tensor = load_video_tensor(
            video_path,
            num_frames=self.num_frames,
            interval_sec=self.interval_sec,
            transform=self.transform,
        )
        if self.return_frame_mask:
            frame_mask = generate_frame_mask(video_tensor)
            return {"input_values": video_tensor, "frame_mask": frame_mask}

        return {"input_values": video_tensor}


def generate_frame_mask(
    video_tensor: torch.Tensor, tolerance: float = 1e-4
) -> torch.BoolTensor:
    """
    全ての画素値が同じであるかどうかでパディングフレームかどうかを判定する
    パディングフレームは1、それ以外は0を返す
    Args:
        video_tensor: shape=(frame, 3, H, W)
    Returns:
        frame_mask: shape=(frame)
    """
    # 各フレーム、各チャンネルにおける最大値と最小値を個別に計算し、厳密な一致をチェック
    max_vals = torch.max(video_tensor, dim=-1)[0].max(dim=-1)[
        0
    ]  # 最後の2次元に対して最大値を取得
    min_vals = torch.min(video_tensor, dim=-1)[0].min(dim=-1)[
        0
    ]  # 最後の2次元に対して最小値を取得

    # 最大値と最小値の差が許容誤差内にあるかどうかをチェック
    is_close = (max_vals - min_vals) <= tolerance

    # 全てのチャンネルで等しい場合にTrueとなるように、フレームごとに全てTrueかどうかをチェック
    frame_mask = is_close.all(dim=1)
    return frame_mask


class TargetDataset(Dataset):
    """
    目的変数用のデータセット
    target_col_nameで指定したカラム名を目的変数とする
    """

    def __init__(self, df: pd.DataFrame, target_col_name: str="label"):
        self.targets = [{"targets": t} for t in df[target_col_name].tolist()]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.targets[idx]