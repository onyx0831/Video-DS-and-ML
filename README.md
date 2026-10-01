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

## Dockerでの環境設定

xformers / triton が Linux x86_64 向けにしか提供されていないため、イメージは linux/amd64 でビルドする。
Apple Silicon の Mac ではエミュレーション（CPU のみ）で動作する。

### JupyterLab

```bash
docker compose up --build
```

http://localhost:8888 を開く。

### VS Code Dev Container

Dev Containers 拡張機能を入れた VS Code でリポジトリを開き、「Reopen in Container」を実行する。
Python インタープリターは `/opt/venv/bin/python` を使う。

### スクリプトの実行

```bash
# MLflow サーバー（http://localhost:8081）
docker compose exec -w /workspace/Mlflow_Pipeline app bash mlflow_server.sh

# Gradio アプリ（http://localhost:7860）
docker compose exec -w /workspace/Deep_Learning_with_Pytorch/sketch_model app python app.py
```

### GCP の GPU VM で使う

VideoMAE の学習（xformers）には NVIDIA GPU が必要なため、学習は GPU VM で行う。
VM には NVIDIA ドライバ（CUDA 12.4 対応）、Docker、NVIDIA Container Toolkit を入れておく。

```bash
# VM 上で実行
git clone git@github.com:onyx0831/Video-DS-and-ML.git
cd Video-DS-and-ML
export COMPOSE_FILE=compose.yaml:compose.gpu.yaml
docker compose up -d --build
docker compose exec app python -c "import torch; print(torch.cuda.is_available())"
```

動画データは Git 管理外のため、ローカルから VM に転送する。

```bash
# ローカルで実行
gcloud compute scp --recurse data/video_data <VM名>:~/Video-DS-and-ML/data/
```

学習はリポジトリのルートから実行する。

```bash
docker compose exec app python Deep_Learning_with_Pytorch/training/train_trainer.py
```

ポートは VM の 127.0.0.1 にしか公開していないため、JupyterLab や MLflow はローカルから SSH ポートフォワードで開く。

```bash
# ローカルで実行
gcloud compute ssh <VM名> -- -L 8888:localhost:8888 -L 8081:localhost:8081
```
