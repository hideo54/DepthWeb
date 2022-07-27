# DepthWeb

深度推定を行う機械学習モデルを用いて、アップロードされた画像の推定される深度を画像および点群プロットグラフで表示するアプリケーション。

Cloud Functions, Cloud Storage を使用する前提で設計されている。

## 詳細な解説

### 機械学習

Keras が [keras-team/keras-io](https://github.com/keras-team/keras-io) にて [Apache 2.0 ライセンス](https://www.apache.org/licenses/LICENSE-2.0)で提供している [`depth_estimation.py`](https://github.com/keras-team/keras-io/blob/master/examples/vision/depth_estimation.py) を複製・改変して使用している。

`pip install -r requirements.txt` により必要なモジュールをインストール可能。ただし、macOS では `tensorflow` の代わりに `tensorflow-macos` を使用する必要がある。

`main.py` の `init()` を実行することで、学習データをダウンロードし、学習を実行して、モデルをローカルに保存することができる。
深度を推定するには、適宜 `predict_depth()` 関数を用いる。

`make_predicted_image()` 関数は、HTTP POST リクエストにより与えられた画像に対して深度推定を実行し、画像を Cloud Storage にアップロードして、そのファイル名と、粗い深度データを返す関数である。Cloud Functions へのデプロイが想定されている。

### Web フロントエンド

`front/` 下に React コードを設置している。ディレクトリ内で `npm install` した上で `npm run build` を実行することで、`front/build/` 下に静的ファイルが生成される。

`create-react-app` により作成した。

## Deploy

モデルの用意と、フロントエンドのビルドを終えたあとに、デプロイコマンドを実行する。

推奨コマンド: `gcloud functions deploy depth-web --runtime python39 --region asia-northeast1 --entry-point make_predicted_image --memory 1024MB --trigger-http --allow-unauthenticated`

特に、メモリ指定は重要。標準では 256 MB だが、600 MB くらい使うので、1024 MB 割り当てるべきである。

また、依存の関係で Python 3.10 では動かないため、ランタイムに Python 3.9 (`python39`) を指定するべきである。
