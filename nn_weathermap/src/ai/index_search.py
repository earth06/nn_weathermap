import faiss
import numpy as np
import torch
import tqdm
import pandas as pd
import pickle
import xarray as xr
from src.ai.nn import MinMaxScaler, Simple2DCNN


class IndexSearch:
    def __init__(self):
        self.weather_index = None
        self.index2datetime = None
        self.scaler: MinMaxScaler = None
        self.model: Simple2DCNN = None
        self.model_path = "./model"
        self.weather_variables = ["t2m", "d2m", "msl"]

    def load_weather_index(self):
        self.weather_index = faiss.read_index(
            f"{self.model_path}/index/weathermap.index"
        )
        self.index2datetime = pd.read_parquet(
            f"{self.model_path}/index/index2datetime.parquet"
        )
        self.scaler = pickle.load(f"{self.model_path}/scaler.pkl")
        self.model = Simple2DCNN()
        check_point = torch.load(f"{self.model_path}/model_2d_cnn.pt")
        self.model.load_state_dict(check_point["model_state_dict"])

    def write_weather_index(self):
        faiss.write_index(
            self.weather_index, f"{self.model_path}/index/weathermap.index"
        )
        self.index2datetime.to_parquet(
            f"{self.model_path}/index/index2datetime.parquet", compression="lz4"
        )
        with open(f"{self.model_path}/scaler.pkl", "bw") as f:
            pickle.dump(self.scaler, f)
        # NNの保存、学習を入れるならoptimizerの保存等も追加する必要がある
        torch.save(
            {{"model_state_dict": self.model.state_dict()}},
            f"{self.model_path}/model_2d_cnn.pt",
        )

    def load_weather_data(self, filepath):
        """入力する気象データ"""
        ds = xr.open_dataset(filepath)
        return ds

    def preprocess(self, ds):
        """データの前処理"""
        # 正規化した上で1つの配列に統合する
        data_list = []
        for var in self.weather_variables:
            data_list.append(self.scaler.apply_scale(ds[var].values, var))
        N_ch = len(data_list)
        data_size, N_lat, N_lon = data_list[0].shape
        data = np.zeros((data_size, N_ch, N_lat, N_lon), dtype=np.float32)
        for ich in range(N_ch):
            data[:, ich, :, :] = data_list[ich]
        del data_list
        return data

    def train(self, ds):
        """
        正規化-> NNの作成 -> 特徴量の抽出 -> indexの作成
        """
        # 正規化のセットアップ
        self.scaler = MinMaxScaler()
        for var in self.weather_variables:
            self.scaler.set_scale(ds[var].values, var)
        # 正規化済みのnp配列に変換
        train_data = self.preprocess(ds)

        # NNの構築
        self.model = Simple2DCNN().eval()

        # 特徴量抽出~indexの作成
        self.create_index(train_data, ds["time"])

        # 検索indexを保存する
        self.write_weather_index()

    def predict(self, ds):
        """すでにscaler, model, indexがセット済みのときにのみ動作する"""
        predict_data = self.preprocess(ds)
        similar_dates = self.find_similar_data(predict_data)
        return similar_dates

    def create_index(self, data_train, datetimes_train):
        feature_matrix = np.array(
            [
                self.extract_features(torch.Tensor(data_train[i]), self.model)
                for i in tqdm.tqdm(range(data_train.shape[0]))
            ]
        ).squeeze()

        dim = feature_matrix.shape[1]
        self.weather_index = faiss.IndexFlatL2(dim)
        self.weather_index.add(feature_matrix)
        self.index2datetime = datetimes_train.to_dataframe().reset_index(drop=True)

    # 特徴量抽出のための関数
    def extract_features(self, tensor, model):
        tensor = tensor.unsqueeze(0)  # バッチの次元を追加
        with torch.no_grad():
            features = model(tensor)
        return features.numpy()

    # 4. 類似度の計算と出力
    def find_similar_data(self, data_predict):
        input_tensor = torch.Tensor(data_predict)
        input_features = self.extract_features(input_tensor, self.model)
        D, I = self.weather_index.search(np.array(input_features), 1)
        most_similar_data_index = I[0][0]
        similar_dates = self.index2datetime[most_similar_data_index]
        return similar_dates
