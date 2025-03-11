# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings
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
threshold_of_rate_of_same_value = 1  # カラム内の値がすべて同じ場合、削除になる
threshold_of_r = 0.2  # 相関係数がこの値以上になる変数があればそのうち片方を削除
warnings.filterwarnings('ignore')


#相違関係数が大きい方の片方を削除する関数の定義
def search_highly_correlated_variables(x, threshold_of_r):
    x = pd.DataFrame(x) 
    r_in_x = x.corr() #r_in_xに相関係数を計算し格納
    r_in_x = abs(r_in_x) #格納した相関係数を絶対値に変換
    for i in range(r_in_x.shape[0]):
        r_in_x.iloc[i, i] = 0 #対角成分を0にする
    highly_correlated_variable_numbers = [] #空のリストを作成
    for i in range(r_in_x.shape[0]):
        r_max = r_in_x.max() #各変数の中で相関係数の絶対値が最も大きい数を抽出
        r_max_max = r_max.max()#r_maxの中で最も相関係数が大きい変数の値を抽出
        if r_max_max >= threshold_of_r:
            print(i + 1)
            variable_number_1 = np.where(r_max == r_max_max)[0][0]#相関係数が最も高い変数の列番号を取得.[0][0]はTrueのインデックスを取得
            variable_number_2 = np.where(r_in_x.iloc[:, variable_number_1] == r_max_max)[0][0] #取得した列番号の中から最も相関係数が高い変数の列番号を抜き出す#.ilocは行番号と列番号で指定
            r_sum_1 = r_in_x.iloc[:, variable_number_1].sum()#相関係数を全て足し合わせる
            r_sum_2 = r_in_x.iloc[:, variable_number_2].sum()
            #相関係数の合計値が高い方の変数の全ての値を0にする
            if r_sum_1 >= r_sum_2:
                delete_x_number = variable_number_1
            else:
                delete_x_number = variable_number_2
            highly_correlated_variable_numbers.append(delete_x_number)
            r_in_x.iloc[:, delete_x_number] = 0
            r_in_x.iloc[delete_x_number, :] = 0
        else:
            break
    return highly_correlated_variable_numbers


#データセットの前処理
dataset = dataset.loc[:, dataset.mean().index]  # 平均を計算できる変数だけ選択
dataset = dataset.replace(np.inf, np.nan).fillna(np.nan)  # infをnanに置き換えておく
dataset = dataset.dropna(axis=1)  # nanのある変数を削除
x = dataset.iloc[:, 1:]

#同じ値の割合が高い変数を削除する
rate_of_same_value = list()
num = 0
for X_variable_name in x.columns:
    num += 1
    #    print('{0} / {1}'.format(num, x.shape[1]))
    #各変数の相関係数でユニークな数を調べる
    same_value_number = x[X_variable_name].value_counts()
    #同じ数の割合を調べる
    rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / x.shape[0]))
    #threshold_of_rate_of_same_valiueの値以上の割合の変数のインデックスを取得する
deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)


#threshold_of_rate_of_same_valiueの値以上の割合の変数を削除する
if len(deleting_variable_numbers[0]) != 0:
    x = x.drop(x.columns[deleting_variable_numbers], axis=1)

#変数の数を示す
print('# of X-variables: {0}'.format(x.shape[1]))

#高い相関があった変数の数を示す
highly_correlated_variable_numbers = search_highly_correlated_variables(x, threshold_of_r)
print('# of highly correlated X-variables: {0}'.format(len(highly_correlated_variable_numbers)))

#変数選択後の変数の数を示す
x_selected = x.drop(x.columns[highly_correlated_variable_numbers], axis=1)
print('# of selected X-variables: {0}'.format(x_selected.shape[1]))

#変数選択後のデータセットをDopt_datasetとしてcsv出力
x_selected.to_csv('correlation_reduced_data.csv')