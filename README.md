# Video-DS-and-ML

サンプルデータのvideo_emotion_sample.csvは、下記論文が公開しているデータセットの中から
amused, eager, active, alert, cheerfulの5種類の感情からランダムに40個ずつ取得。合計200行。

Automatic Understanding of Image and Video Advertisements

## uvでの環境設定
ubuntu

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL -l
uv init
uv sync
. .venv/bin/activate
uv add ipython
uv add ipykernel
ipython kernel install --user --name=video-ds-and-ml
```
