# -*- coding: utf-8 -*-

import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors  # k-NN

'''
手書き入力-------------------------------------------------------------------------------------------
'''

# データセット読み込み
dataset = pd.read_csv('Boruta_selected_dataset.csv', index_col=0) 
# AD 内となるトレーニングデータの割合。AD　のしきい値を決めるときに使用
rate_of_training_samples_inside_ad = 0.96
# k-NN における k
k_in_knn = 10 
# テストデータのサンプル数
number_of_test_samples = math.ceil(0.2*dataset.shape[0])
# k-NNのデータ間の距離の指標
metric = 'euclidean'

'''
-----------------------------------------------------------------------------------------------
'''

def set_ad_with_knn(dataset, rate_of_training_samples_inside_ad, k_in_knn, number_of_test_samples, metric):
    """
    ---input---
    dataset:0列目にindex、1列目に目的変数、2列目以降に説明変数が格納されたデータセットのcsv
    rate_of_training_samples_inside_ad:ADの閾値α
    k_in_knn:k-NNにおけるk
    number_of_test_samples:テストデータのサンプル数
    metric:k-NNのデータ間の距離の指標。defaultはユークリッド('euclidean')。
    　　　　　 特徴量間に相関がある場合は、metric = 'mahalanobis'を用いてマハラノビス距離を用いたほうが良い可能性がある。


    ---output---
    mean_of_knn_distance_train.csv:学習データにおける、距離の近い対象のサンプル以外のk個のサンプルとの距離の平均値
    inside_ad_flag_train.csv:学習データのADの判定結果（AD内:TRUE、AD外:FALSE）
    mean_of_knn_distance_test.csv:テストデータにおける、学習データの距離の近いk個のサンプルとの距離の平均値
    inside_ad_flag_test.csv:テストデータのADの判定結果（AD内:TRUE、AD外:FALSE）

    """
    
    # データ分割
    y = dataset.iloc[:, 0]  # 目的変数
    x = dataset.iloc[:, 1:]  # 説明変数
    # ランダムにトレーニングデータとテストデータとに分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, shuffle=True,
                                                        random_state=1)
    
    # 標準偏差が0の説明変数を削除(トレーニングとテスト)
    std_0_variable_flags = x_train.std() == 0
    x_train = x_train.drop(x_train.columns[std_0_variable_flags], axis=1)
    x_test = x_test.drop(x_test.columns[std_0_variable_flags], axis=1)
    
    # オートスケーリング
    autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()
    autoscaled_x_test = (x_test - x_train.mean()) / x_train.std()
    
    # k-NN による AD
    ad_model = NearestNeighbors(n_neighbors=k_in_knn, metric=metric)  # AD モデルの宣言。
    ad_model.fit(autoscaled_x_train)  # k-NN による AD では、トレーニングデータの x を model_ad に格納することに対応
    
    # サンプルごとの k 最近傍サンプルとの距離に加えて、k 最近傍サンプルのインデックス番号も一緒に出力されるため、出力用の変数を 2 つに
    # トレーニングデータでは k 最近傍サンプルの中に自分も含まれ、自分との距離の 0 を除いた距離を考える必要があるため、k_in_knn + 1 個と設定
    knn_distance_train, knn_index_train = ad_model.kneighbors(autoscaled_x_train, n_neighbors=k_in_knn + 1)
    knn_distance_train = pd.DataFrame(knn_distance_train, index=x_train.index)  
    # DataFrame型に変換
    mean_of_knn_distance_train = pd.DataFrame(knn_distance_train.iloc[:, 1:].mean(axis=1),
                                              columns=['mean_of_knn_distance'])  # 自分以外の k_in_knn 個の距離の平均
    mean_of_knn_distance_train.to_csv('mean_of_knn_distance_train.csv')  # csv ファイルに保存
    
    # トレーニングデータのサンプルの rate_of_training_samples_inside_ad * 100 % が含まれるようにしきい値を設定
    sorted_mean_of_knn_distance_train = mean_of_knn_distance_train.iloc[:, 0].sort_values(ascending=True)  # 距離の平均の小さい順に並び替え
    ad_threshold = sorted_mean_of_knn_distance_train.iloc[
        round(autoscaled_x_train.shape[0] * rate_of_training_samples_inside_ad) - 1]
    
    # トレーニングデータに対して、AD の中か外かを判定
    inside_ad_flag_train = mean_of_knn_distance_train <= ad_threshold # AD 内のサンプルのみ TRUE
    inside_ad_flag_train.columns=['inside_ad_flag'] 
    inside_ad_flag_train.to_csv('inside_ad_flag_train.csv')  # csv ファイルに保存。
    
    # テストデータに対する k-NN 距離の計算
    knn_distance_test, knn_index_test = ad_model.kneighbors(autoscaled_x_test)
    knn_distance_test = pd.DataFrame(knn_distance_test, index=x_test.index)  # DataFrame型に変換
    mean_of_knn_distance_test = pd.DataFrame(knn_distance_test.mean(axis=1),
                                             columns=['mean_of_knn_distance'])  # k_in_knn 個の距離の平均
    mean_of_knn_distance_test.to_csv('mean_of_knn_distance_test.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意
    
    # テストデータに対して、AD の中か外かを判定
    inside_ad_flag_test = mean_of_knn_distance_test <= ad_threshold # AD 内のサンプルのみ TRUE
    inside_ad_flag_test.columns=['inside_ad_flag'] 
    inside_ad_flag_test.to_csv('inside_ad_flag_test.csv')  # csv ファイルに保存。
    
    
set_ad_with_knn(dataset, rate_of_training_samples_inside_ad, k_in_knn, number_of_test_samples, metric)