import numpy as np
import cv2
import warnings

warnings.simplefilter("ignore")


class VideoReader:
    """
    OpenCVベースの動画読み込み用ユーティリティクラス。

    Attributes
    ----------
    cap : object
        path_input(動画URL)から映像を読み込む
        cv2.VideoCapture()
    duration : double
        動画の秒数
    size : tuple
        動画の横縦のサイズ
    """

    def __init__(self, path_input: str):

        self.path_input = path_input
        self.cap = cv2.VideoCapture(self.path_input)

        if not self.cap.isOpened():
            raise ValueError(f'video_path: "{self.path_input}" is invalid.')

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration = self.total_frames / self.fps if self.fps else 0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.size = tuple(
            map(
                int,
                (
                    self.width,
                    self.height,
                ),
            )
        )

    def frame_idx_from_sec(self, sec: float) -> int:
        # 秒数から獲得したいフレーム番号を取得
        # 30fpsで2秒目だと、フレーム番号59が返るが0秒以下の場合は0番が返る
        idx = int(sec * self.fps)
        return max(0, int(min(idx, self.total_frames - 1)))

    def _move_pos_frames(self, frame: int) -> bool:
        """
        指定したフレーム番号に移動する。
        （次のself.cap.read() で指定したフレーム番号を取得できる）
        self.capの内部状態を変更し、成否を返す。
        """
        initial_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame < initial_frame:
            warnings.warn(
                "Backward frame search may cause a degradation in read performance.: "
                f"current_frame: {initial_frame} -> VideoReader._move_frame(frame={frame})"
            )
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame) よりも一般的に早い
        # （全てのケースでより早くするためにはキーフレームの理解が必要）
        for _ in range(frame - initial_frame):
            success, _ = self.cap.read()
            if not success:
                return False
        # 上記のcap.readのステータスによって読み込みの成否はチェックされるため
        # 基本的には下記のエラーにはならないはずだが、
        # ロジックミスや予期しない状況の発生を警戒し、念のため確認する。
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame != frame:
            raise AssertionError(
                f"Unexpected frame position after reading. "
                f"expected frame: {frame}, current frame: {current_frame}"
            )
        return True  # succeeded

    def get_frame_by_idx(self, frame_idx: int) -> np.ndarray:
        """
        指定フレーム番号からフレームを取得する。
        フレームが存在しない、または読み取りに失敗した場合は黒画像を返す。

        Args:
            frame_idx (int): フレーム番号（0始まり）

        Returns:
            np.ndarray: フレーム画像または黒画像
        """

        success_move = self._move_pos_frames(frame_idx)
        success_read, img = self.cap.read()
        if success_move and success_read:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = self.black_frame()

        return img

    def get_frame_by_sec(self, sec: int) -> np.ndarray:
        # sec秒地点のフレームを取得、もしない場合は黒塗り画像を返す
        """
        指定秒数からフレームを取得する。
        秒数が範囲外、または読み取りに失敗した場合は黒画像を返す。

        Args:
            sec (float): 秒数

        Returns:
            np.ndarray: フレーム画像または黒画像
        """

        idx = self.frame_idx_from_sec(sec)
        return self.get_frame_safe_by_idx(idx)

    def get_duration(self) -> float:
        return self.duration

    def get_size(self) -> tuple:
        return self.size

    def get_fps(self) -> float:
        return self.fps

    def get_total_frame_count(self):
        # 総フレーム数を取得
        return self.total_frames

    def black_frame(self) -> np.ndarray:
        """黒塗りのダミーフレーム（画像）を生成して返す。"""
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def release(self):
        """VideoCaptureのリソースを解放する。"""
        self.cap.release()

    def __del__(self):
        """デストラクタ。リソース解放を保証。"""
        self.release()
