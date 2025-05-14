FROM tensorflow/tensorflow:2.15.0-gpu

# Rust をインストールする
RUN apt-get update && apt-get install -y \
    sudo \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# vscode ユーザーを作成
RUN groupadd --gid 1000 vscode \
    && useradd -s /bin/bash --uid 1000 --gid 1000 -m vscode \
    && echo vscode ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode

USER vscode
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
