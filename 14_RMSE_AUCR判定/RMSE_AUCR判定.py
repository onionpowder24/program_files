# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


# RMSEの計算を始めるデータポイントを定義
first_rmse_data_num = 0

# %pを定義
p = 100

# 保存先ディレクトリ
save_path = './coverage_RMSE_curve.png'

# 回帰モデル予測結果読み込み(テストデータにおけるyの推測値とyの実測値が格納されているcsv)
df_eval = pd.read_csv('estimated_y_test.csv', index_col=0)

# テストデータのknnの距離を格納したcsvを読み込み
df_knn = pd.read_csv('mean_of_knn_distance_test.csv', index_col=0)

# 読み込んだ2つのデータフレームを結合
df_concat = pd.concat([df_eval, df_knn], axis=1)

# knnの距離の短い順(ADの内側順)にデータフレームを並べかえ
df_concat_s = df_concat.sort_values('mean_of_knn_distance', ascending=True)

# 並べ替えたデータフレームからyの実測値とyの予測値を抽出
actual_y = df_concat_s['actual_y'].values
estimated_y = df_concat_s['estimated_y'].values

# p%-AUCRを計算する関数
def calculate_aucr(actual_y, estimated_y, p, first_rmse_data_num, save_path):
    """
    --input--
    actual_y:サンプルをAD指標の降順(ADの内側順)に並べ替えた後の実測値yの n×1の配列
    estimated_y:サンプルをAD指標の降順(ADの内側順)に並べ替えた後の推定値yの n×1の配列
    p:p%-AUCRにおけるcoverage[%]の閾値
    first_rmse_data_num:RMSEを計算し始めるデータ数
    save_path:coverage×RMSEのグラフの保存先
    
    --output--
    p_aucr:p%-AUCR
    """
    
    # テストデータ数を定義
    n = actual_y.shape[0]
    
    # coverageとRMSEの計算
    coverage_rmse = np.zeros((n-first_rmse_data_num, 2)) # 計算結果を格納するゼロ行列を作成
    for i in range(first_rmse_data_num, n):
        # coverageの計算
        coverage_rmse[i, 0] = (i+1) / n 
        # RMSEの計算
        coverage_rmse[i, 1] = metrics.mean_squared_error(actual_y[0:i+1], estimated_y[0:i+1]) ** 0.5
        
    # %p-AUCRの計算
    # coverageのp%以下に対応するインデックス取得
    index_of_coverage_lower_than_p = np.searchsorted(coverage_rmse[:, 0], p/100, side='right') - 1
    # %p-AUCR算出
    p_aucr = (sum(coverage_rmse[:index_of_coverage_lower_than_p,1]) 
              - (coverage_rmse[0, 1] + coverage_rmse[index_of_coverage_lower_than_p-1, 1])/2)*1/n
    
    # coverageとRMSEの関係を可視化して保存
    plt.figure()
    plt.rcParams['font.size'] = 18
    plt.scatter(coverage_rmse[:,0] * 100, coverage_rmse[:,1], c='blue')
    plt.xlabel('Coverage [%]')
    plt.ylabel('RMSE')
    plt.savefig(save_path, bbox_inches = 'tight')
    plt.show()
    
    return p_aucr
        
p_aucr = calculate_aucr(actual_y, estimated_y, p, first_rmse_data_num, save_path)

print(p_aucr)


'''
元のコード（MATLAB）

function pAUCR = calculatepAUCR(actualy, predictedy, p, firstRMSEdatanumber)

n = length(actualy);
% calculation of coverage and RMSE
coverageRMSE = zeros(n-firstRMSEdatanumber+1, 2);
for i = firstRMSEdatanumber:n
coverageRMSE(i-firstRMSEdatanumber+1,) = [i/n sqrt(sum((actualy(1:i)).2)/n)];
end
% calculation of p%-AUCR
indexofcoveragelowerthanp = find(coverageRMSE(:,1) <= p/100);
pAUCR = (sum(coverageRMSE(indexofcoveragelowerthanp,2)) - (coverageRMSE(1,2)+coverageRMSE(indexofcoveragelowerthanp(end),2))/2)*1/n;

'''