# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import sample_functions
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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
dataset = dataset.dropna(how='any')#欠損値が１つでも含まれる行を削除

#標準偏差が0の説明変数を削除
std_0_variable_flags = dataset.std() == 0
dataset = dataset.drop(dataset.columns[std_0_variable_flags], axis=1)

# ハイパーパラメータ
rf_number_of_trees = dataset.shape[0]*2  # RF における決定木の数
number_of_test_samples = math.ceil(0.2*dataset.shape[0])  # テストデータのサンプル数(全体の２割に設定)
rf_x_variables_rates = np.arange(1, 11, dtype=float) / 10  # 1 つの決定木における説明変数の数の割合の候補

# データ分割
y = dataset.iloc[:, 0]  # 目的変数
x = dataset.iloc[:, 1:]  # 説明変数

# ランダムにトレーニングデータとテストデータとに分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, shuffle=True,
                                                    random_state=123)

# 標準偏差が 0 の特徴量の削除
deleting_variables = x_train.columns[x_train.std() == 0]
x_train = x_train.drop(deleting_variables, axis=1)
x_test = x_test.drop(deleting_variables, axis=1)


# Borutaのハイパーパラメータ選定(perc)
corr = [] #空のリストを作る
for index in range(10000): 
    print(index)
    #print(index + 1, '/', 10000)
    #コピーを作成して全体をランダムにシャッフルし、インデックスの振り直し
    x_copy = x_train.copy().sample(frac=1, random_state = index).reset_index(drop=True)
    #変数をランダムで抽出しシリーズに変換
    x_column_random_series =  x_copy.sample(axis=1, random_state = index).iloc[:,0]
    #相関係数を計算
    y_copy = y_train.copy().reset_index(drop=True)
    caliculated_corr = abs(y_copy.corr(x_column_random_series))
    #計算した相関係数の絶対値をリストに追加
    corr.append(caliculated_corr)
rccmax = max(corr)

# OOB (Out-Of-Bugs) による説明変数の数の割合の最適化
accuracy_oob = [] #空のリストを作る
#index とx_variables_rateにrf_x_variables_rateのindexと中身の数を渡すfor文
for index, x_variables_rate in enumerate(rf_x_variables_rates): 
    print(index + 1, '/', len(rf_x_variables_rates))
    model_in_validation = RandomForestRegressor(n_estimators=rf_number_of_trees, max_features=int(
        #math.cailは引数以上の最小の整数を返す。引数の１は最小値でも１を返すようにするため
        max(math.ceil(x_train.shape[1] * x_variables_rate), 1)), oob_score=True,random_state=123)
    model_in_validation.fit(x_train, y_train)
    #使用する説明変数を振り、それぞれでOOB正解率をリストに加える
    accuracy_oob.append(model_in_validation.oob_score_)
optimal_x_variables_rate = sample_functions.plot_and_selection_of_hyperparameter(rf_x_variables_rates,
                                                                                 accuracy_oob,
                                                                                 'rate of x-variables',
                                                                                 'accuracy for OOB')
print('\nOOB で最適化された説明変数の数の割合 :', optimal_x_variables_rate)

# 1 つの決定木における説明変数の数の割合はOOB正解率で最適化したものを使用する
# Borutaで使用するランダムフォレストモデルの宣言
model = RandomForestRegressor(n_estimators=rf_number_of_trees,
                               max_features=int(max(math.ceil(x_train.shape[1] * optimal_x_variables_rate), 1)),
                               oob_score=True,random_state=123) 

#　Borutaを実行 #特徴量の数が減っていってエラーが出るときは、max_iterを下げたり、random_stateかえたり。元は10000と123だった。
feat_selector = BorutaPy(model, n_estimators='auto', verbose=2, perc = 100*(1-rccmax),max_iter = 64, random_state=12)
feat_selector.fit(x_train.values, y_train.values)


# 選択された特徴量を確認
selected = feat_selector.support_
print('選択された特徴量の数: %d' % np.sum(selected))
print(selected)
print(x_train.columns[selected])

# 選択した特徴量のみのデータセットを保存
y = pd.concat([y_train,y_test],axis = 0)
x = pd.concat([x_train,x_test],axis = 0)
selected_dataset = pd.concat([y,x[x.columns[selected]]],axis = 1)
selected_dataset = selected_dataset.sort_index()
selected_dataset.to_csv('Boruta_selected_dataset.csv') 

# 削除されたサンプルを確認
not_selected_dataset = dataset.drop(selected_dataset.index)
not_selected_dataset.to_csv('not_selected_dataset.csv') 

# y_train と y_test のヒストグラムを確認
plt.hist(y_train)
plt.hist(y_test)