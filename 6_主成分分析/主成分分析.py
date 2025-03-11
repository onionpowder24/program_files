# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import os

# 現在のディレクトリ内のinputフォルダ内のCSVファイルを探して読み込む関数
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


# datasetを読み込む
dataset = load_csv_from_folder()
x = dataset.iloc[:, 1:]
autoscaled_x = (x - x.mean()) / x.std()  # オートスケーリング

# PCA
pca = PCA()  # PCA を行ったり PCA の結果を格納したりするための変数を、pca として宣言
pca.fit(autoscaled_x)  # PCA を実行

# ローディング
loadings = pd.DataFrame(pca.components_.T, index=x.columns)
loadings.to_csv('pca_loadings.csv')

# スコア
score = pd.DataFrame(pca.transform(autoscaled_x), index=x.index)
score.to_csv('pca_score.csv')

# 寄与率、累積寄与率
contribution_ratios = pd.DataFrame(pca.explained_variance_ratio_)  # 寄与率を DataFrame 型に変換
cumulative_contribution_ratios = contribution_ratios.cumsum()  # cumsum() で寄与率の累積和を計算
cont_cumcont_ratios = pd.concat(
    [contribution_ratios, cumulative_contribution_ratios],
    axis=1).T
cont_cumcont_ratios.index = ['contribution_ratio', 'cumulative_contribution_ratio']  # 行の名前を変更
cont_cumcont_ratios.to_csv('pca_cont_cumcont_ratios.csv')

# 寄与率を棒グラフで、累積寄与率を線で入れたプロット図を重ねて描画
x_axis = range(1, contribution_ratios.shape[0] + 1)  # 1 から成分数までの整数が x 軸の値
plt.rcParams['font.size'] = 18
plt.bar(x_axis, contribution_ratios.iloc[:, 0], align='center')  # 寄与率の棒グラフ
plt.plot(x_axis, cumulative_contribution_ratios.iloc[:, 0], 'r.-')  # 累積寄与率の線を入れたプロット図
plt.xlabel('Number of principal components')  # 横軸の名前
plt.ylabel('Contribution ratio(blue),\nCumulative contribution ratio(red)')  # 縦軸の名前。\n で改行しています
plt.show()

# 第 1 主成分と第 2 主成分の散布図 (2列目の値でサンプルに色付け)
plt.scatter(score.iloc[:, 0], score.iloc[:, 1], c=dataset.iloc[:, 0], cmap=plt.get_cmap('jet'))
plt.colorbar()
plt.xlabel('t_1 (PCA)')
plt.ylabel('t_2 (PCA)')
plt.show()

# 第 1 主成分と第 2 主成分の散布図 (２列目の値でサンプルに色付け, Lot.No.記述)
plt.scatter(score.iloc[:, 0], score.iloc[:, 1], c=dataset.iloc[:, 0], cmap=plt.get_cmap('jet'))
plt.colorbar()
plt.rcParams['font.size'] = 10
for sample_number in range(score.shape[0]):
    plt.text(score.iloc[sample_number, 0], score.iloc[sample_number, 1], score.index[sample_number],
             horizontalalignment='center', verticalalignment='top')
plt.xlabel('t_1 (PCA)')
plt.ylabel('t_2 (PCA)')
plt.show()