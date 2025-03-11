# -*- coding: utf-8 -*-
# %reset -f

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster  # SciPy の中の階層的クラスタリングを実行したり樹形図を作成したりするためのライブラリをインポート
from sklearn.neighbors import NearestNeighbors

# ハイパーパラメータ
number_of_max_clusters = 10  # maximum number of clusters
k_in_knn = 3  # k in k-NN


def load_csv_from_folder():
    directory = os.path.join(os.getcwd(), "input")  # 現在のディレクトリのinputフォルダを取得

    # inputフォルダ内のすべてのファイルをチェックし、CSVファイルを探す
    if not os.path.exists(directory):
        print("inputフォルダが存在しません。")
        return None

    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    # CSVファイルが1種類のみ存在することを確認
    if len(csv_files) != 1:
        print("inputフォルダ内に1つだけのCSVファイルが存在する必要があります。")
        return None

    # ファイル名を取得して読み込む
    selected_file = csv_files[0]
    print(f"{selected_file} を読み込みます...")
    file_path = os.path.join(directory, selected_file)
    return pd.read_csv(file_path, index_col=0)

# datasetの読み込み
dataset = load_csv_from_folder()

#標準偏差が0の説明変数を削除
std_0_variable_flags = dataset.std() == 0
dataset = dataset.drop(dataset.columns[std_0_variable_flags], axis=1)

x = dataset.iloc[:, 1:]
autoscaled_x = (x - x.mean()) / x.std()  # オートスケーリング


# k-NN
knn = NearestNeighbors(n_neighbors=k_in_knn)
knn.fit(autoscaled_x)
knn_dist_all, knn_ind_all = knn.kneighbors(None)

# clustering
clustering_results = linkage(autoscaled_x, metric='euclidean', method='ward')

# 最適クラスター数の算出
true_rate = []
for number_of_clusters in range(1, number_of_max_clusters + 1):
    print(number_of_clusters, number_of_max_clusters)
    cluster_numbers = fcluster(clustering_results, number_of_clusters, criterion='maxclust')  # クラスターの数で分割し、クラスター番号を出力
    true_number = 0
    for i in range(knn_ind_all.shape[0]):
        true_number += len(np.where(cluster_numbers[knn_ind_all[i, :]] == cluster_numbers[i])[0])
    true_rate.append(true_number / (knn_ind_all.shape[0] * knn_ind_all.shape[1]))

plt.scatter(range(1, number_of_max_clusters + 1), true_rate, c='blue')  # 散布図の作成。クラスター番号ごとにプロットの色を変えています
plt.xlabel('number of cluster')
plt.ylabel('matching ratio')
plt.show()

true_rate = np.array(true_rate)
optimal_cluster_number = np.where(true_rate == 1)[0][-1] + 1
print('Optimum number of clusters :', optimal_cluster_number)

# デンドログラムの作成
plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
dendrogram(clustering_results, labels=list(x.index), color_threshold=0,
           orientation='right')  # デンドログラムの作成。labels=x.index でサンプル名を入れています
plt.xlabel('distance')  # 横軸の名前


# 最適クラスター数でのクラスター番号保存
number_of_clusters = optimal_cluster_number
cluster_numbers = fcluster(clustering_results, optimal_cluster_number, criterion='maxclust')  # クラスターの数で分割し、クラスター番号を出力
cluster_numbers = pd.DataFrame(cluster_numbers,index = x.index,
                               columns=['cluster_numbers']) # DataFrame型に変換。行の名前・列の名前も設定
cluster_numbers.to_csv("cluster_numbers.csv")

