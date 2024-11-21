# ベースイメージの選択
FROM python:3.11-slim

# 必要なLinuxパッケージのインストール（MeCab関連）
RUN apt-get update && apt-get install -y \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# 必要なファイルをコピー
COPY requirements.txt requirements.txt
COPY app.py app.py
COPY lotiondic0712.csv lotiondic0712.csv
COPY shampoodic_floral_15.csv shampoodic_floral_15.csv

# Pythonパッケージのインストール
RUN pip install --no-cache-dir -r requirements.txt

# MeCabの設定ファイルパスを確認（必要ならリンクを作成）
RUN ln -s /etc/mecabrc /usr/local/etc/mecabrc

# ポートを設定
EXPOSE 10000

# アプリケーションの起動コマンド
CMD ["python", "app.py"]
