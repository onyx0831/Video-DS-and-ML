from typing import Callable
import torch
from utils.video_reader import VideoReader
from utils.video_transform import VideoTransform

def load_videoframes(
    video_path: str,
    num_frames: int = 16,
    interval_sec: int = 2,
) -> list:
    """
    Args:
        video_path:動画のパス
        num_frames:モデルが許容するフレーム数
        interval_sec:獲得するフレームの間隔を決める秒数
    Return:
        フレーム画像の集合,listでlen(15)の中身がnp.ndarray(縦,横,色)
    """

    reader = VideoReader(video_path)
    video_sec = int(reader.get_duration())

    num_sec = list(range(0, video_sec + 1, interval_sec))
    frames = [reader.frame_idx_from_sec(i) for i in num_sec]
    frames = frames[:num_frames]

    img_group = []
    add_frame = num_frames - len(frames)  # 残りのフレーム数

    for i in frames:
        # 0フレーム目から画像を読み込む
        img = reader.get_frame_by_idx(i)
        img_group.append(img)

    if add_frame > 0:
        # framesがnum_frames以下の場合、パディングを残りフレーム分追加
        for i in range(add_frame):
            img_group.append(reader.black_frame())

    return img_group


def load_video_tensor(
    video_path: str,
    num_frames: int = 16,
    interval_sec: int = 2,
    transform: Callable = VideoTransform(
        224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    ),
) -> torch.Tensor:
    """
    Args:
        video_path:動画のパス
        num_frames:モデルが許容するフレーム数
        interval_sec:獲得するフレームの間隔を決める秒数
        transform:前処理用の関数
    Return:
        フレーム画像の集合, tensor(num_frames,3,224,224)
    """
    img_group = load_videoframes(video_path, num_frames, interval_sec)
    return transform(img_group)