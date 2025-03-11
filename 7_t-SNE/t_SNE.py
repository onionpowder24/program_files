# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sample_functions
import os
from sklearn.manifold import TSNE

tsne_perplexity_optimization = True  # True にすると t-SNE の perplexity を candidates_of_perplexity の中から k3n-error が最小になるように決めます(時間がかかります)。False にすると 下の perplexity が用いられます
perplexity = 20  # t-SNE の perplexity　（optimization = Falseのとき）
candidates_of_perplexity = np.arange(5, 105, 5, dtype=int)
k_in_k3n_error = 10

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

dataset = load_csv_from_folder()
dataset = dataset.dropna(how='any')#欠損値が１つでも含まれる行を削除

#標準偏差が0の説明変数を削除
std_0_variable_flags = dataset.std() == 0
dataset = dataset.drop(dataset.columns[std_0_variable_flags], axis=1)

x = dataset.iloc[:, 1:]
autoscaled_x = (x - x.mean()) / x.std()  # オートスケーリング

# t-SNE
if tsne_perplexity_optimization:
    # k3n-error を用いた perplexity の最適化 
    k3n_errors = []
    for index, perplexity in enumerate(candidates_of_perplexity):
        t = TSNE(perplexity=perplexity, n_components=2, init='pca', random_state=10).fit_transform(autoscaled_x)
        scaled_t = (t - t.mean(axis=0)) / t.std(axis=0, ddof=1)
    
        k3n_errors.append(
            sample_functions.k3n_error(autoscaled_x, scaled_t, k_in_k3n_error) + sample_functions.k3n_error(
                scaled_t, autoscaled_x, k_in_k3n_error))
    plt.rcParams['font.size'] = 18
    plt.scatter(candidates_of_perplexity, k3n_errors, c='blue')
    plt.xlabel("perplexity")
    plt.ylabel("k3n-errors")
    plt.show()
    optimal_perplexity = candidates_of_perplexity[np.where(k3n_errors == np.min(k3n_errors))[0][0]]
    print('\nk3n-error による perplexity の最適値 :', optimal_perplexity)
else:
    optimal_perplexity = perplexity


t = TSNE(perplexity=optimal_perplexity, n_components=2, init='pca', random_state=10).fit_transform(autoscaled_x)
t = pd.DataFrame(t, index=autoscaled_x.index, columns=['t_1 (t-SNE)', 't_2 (t-SNE)'])
t.columns = ['t_1 (t-SNE)', 't_2 (t-SNE)']
t.to_csv('tsne_t.csv')
# t1 と t2 の散布図 (目的変数の値でサンプルに色付け)
plt.rcParams['font.size'] = 18
plt.scatter(t.iloc[:, 0], t.iloc[:, 1], c=dataset.iloc[:, 0], cmap=plt.get_cmap('jet'))
plt.colorbar()
plt.xlabel('t_1 (t-SNE)')
plt.ylabel('t_2 (t-SNE)')
plt.show()
