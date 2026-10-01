# xformers / triton は Linux x86_64 向け wheel しかないため amd64 で固定する
FROM --platform=linux/amd64 python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /bin/

# opencv-python の実行に必要なライブラリと、Dev Container 用の git
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

# ソースを /workspace にマウントしても隠れないよう、仮想環境は別ディレクトリに置く
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--IdentityProvider.token="]
