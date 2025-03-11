# -*- coding: utf-8 -*-
import matplotlib.figure as figure
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn import metrics
import math
import numpy.matlib
import shap
import os
import lightgbm as lgb
import xgboost as xgb
import catboost as  cb
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor


def plot_and_selection_of_hyperparameter_in_cv(hyperparameter_values, metrics_values, x_label, y_label,
                                         output_folder_name):
    # ハイパーパラメータ (成分数、k-NN の k など) の値ごとの統計量 (CV 後のr2, 正解率など) をプロット
    plt.figure()
    plt.rcParams['font.size'] = 18
    plt.scatter(hyperparameter_values, metrics_values, c='blue')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(output_folder_name + '/' + 'r2_in_cross-validation' + '.jpg',bbox_inches = 'tight')
    plt.show()
    # 統計量 (CV 後のr2, 正解率など) が最大のときのハイパーパラメータ (成分数、k-NN の k など) の値を選択
    return hyperparameter_values[metrics_values.index(max(metrics_values))]

def plot_and_selection_of_hyperparameter_in_oob(hyperparameter_values, metrics_values, x_label, y_label,
                                         output_folder_name):
    # ハイパーパラメータ (成分数、k-NN の k など) の値ごとの統計量 (CV 後のr2, 正解率など) をプロット
    plt.figure()
    plt.rcParams['font.size'] = 18
    plt.scatter(hyperparameter_values, metrics_values, c='blue')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(output_folder_name + '/' + 'r2_in_OOB' + '.jpg',bbox_inches = 'tight')
    plt.show()
    # 統計量 (CV 後のr2, 正解率など) が最大のときのハイパーパラメータ (成分数、k-NN の k など) の値を選択
    return hyperparameter_values[metrics_values.index(max(metrics_values))]


def estimation_and_performance_check_in_regression_train_and_test(model, x_train, y_train, x_test, y_test,
                                                                  output_folder_name):
    # トレーニングデータの推定
    estimated_y_train = model.predict(x_train) * y_train.std() + y_train.mean()  # y を推定し、スケールをもとに戻します
    estimated_y_train = pd.DataFrame(estimated_y_train, index=x_train.index,
                                     columns=['estimated_y'])  # Pandas の DataFrame 型に変換。行の名前・列の名前も設定
    
    # トレーニングデータの実測値 vs. 推定値のプロット
    plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
    plt.figure(figsize=figure.figaspect(1))  # 図の形を正方形に
    plt.scatter(y_train, estimated_y_train.iloc[:, 0], c='blue')  # 実測値 vs. 推定値プロット
    y_max = max(y_train.max(), estimated_y_train.iloc[:, 0].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
    y_min = min(y_train.min(), estimated_y_train.iloc[:, 0].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定
    plt.xlabel('actual y')  # x 軸の名前
    plt.ylabel('estimated y')  # y 軸の名前
    plt.savefig(output_folder_name + '/' + 'training_y_y_plot' + '.jpg',bbox_inches = 'tight') # 画像の保存
    plt.show()  # 以上の設定で描画
    
    
    # トレーニングデータの実測値 vs. 推定値のプロット (サンプル名を表示)
    plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
    plt.figure(figsize=figure.figaspect(1))  # 図の形を正方形に
    plt.scatter(y_train, estimated_y_train.iloc[:, 0], c='blue')  # 実測値 vs. 推定値プロット
    y_max = max(y_train.max(), estimated_y_train.iloc[:, 0].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
    y_min = min(y_train.min(), estimated_y_train.iloc[:, 0].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定
    
    for sample_number in range(y_train.shape[0]):
        plt.text(y_train.iloc[sample_number], estimated_y_train.iloc[sample_number], y_train.index[sample_number],
             horizontalalignment='left', verticalalignment='top')
    plt.xlabel('actual y')  # x 軸の名前
    plt.ylabel('estimated y')  # y 軸の名前
    plt.savefig(output_folder_name + '/' + 'training_y_y_plotwith_name' + '.jpg',bbox_inches = 'tight') # 画像の保存
    plt.show()  # 以上の設定で描画
    
    # トレーニングデータのr2, RMSE, MAE
    print('r^2 for training data :', metrics.r2_score(y_train, estimated_y_train))
    print('RMSE for training data :', metrics.mean_squared_error(y_train, estimated_y_train) ** 0.5)
    print('MAE for training data :', metrics.mean_absolute_error(y_train, estimated_y_train))

    # トレーニングデータの結果の保存
    y_train_for_save = pd.DataFrame(y_train)  # Series のため列名は別途変更
    y_train_for_save.columns = ['actual_y']
    y_error_train = y_train_for_save.iloc[:, 0] - estimated_y_train.iloc[:, 0]
    y_error_train = pd.DataFrame(y_error_train)  # Series のため列名は別途変更
    y_error_train.columns = ['error_of_y(actual_y-estimated_y)']
    results_train = pd.concat([estimated_y_train, y_train_for_save, y_error_train], axis=1)
    results_train.to_csv('{}/estimated_y_train.csv'.format(output_folder_name))  # 推定値を csv ファイルに保存。

    # テストデータの推定
    estimated_y_test = model.predict(x_test) * y_train.std() + y_train.mean()  # y を推定し、スケールをもとに戻します
    estimated_y_test = pd.DataFrame(estimated_y_test, index=x_test.index,
                                    columns=['estimated_y'])  # Pandas の DataFrame 型に変換。行の名前・列の名前も設定
   
    # テストデータの実測値 vs. 推定値のプロット
    plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
    plt.figure(figsize=figure.figaspect(1))  # 図の形を正方形に
    plt.scatter(y_test, estimated_y_test.iloc[:, 0], c='blue')  # 実測値 vs. 推定値プロット
    y_max = max(y_test.max(), estimated_y_test.iloc[:, 0].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
    y_min = min(y_test.min(), estimated_y_test.iloc[:, 0].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定
    plt.xlabel('actual y')  # x 軸の名前
    plt.ylabel('estimated y')  # y 軸の名前
    plt.savefig(output_folder_name + '/' + 'test_y_y_plot' + '.jpg',bbox_inches = 'tight') # 画像の保存
    plt.show()  # 以上の設定で描画
    
    # テストデータの実測値 vs. 推定値のプロット (サンプル名を表示)
    plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
    plt.figure(figsize=figure.figaspect(1))  # 図の形を正方形に
    plt.scatter(y_test, estimated_y_test.iloc[:, 0], c='blue')  # 実測値 vs. 推定値プロット
    y_max = max(y_test.max(), estimated_y_test.iloc[:, 0].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
    y_min = min(y_test.min(), estimated_y_test.iloc[:, 0].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定


    for sample_number in range(y_test.shape[0]):
        plt.text(y_test.iloc[sample_number], estimated_y_test.iloc[sample_number], y_test.index[sample_number],
             horizontalalignment='left', verticalalignment='top')
    plt.xlabel('actual y')  # x 軸の名前
    plt.ylabel('estimated y')  # y 軸の名前
    plt.savefig(output_folder_name + '/' + 'test_y_y_plot_with_sample_name' + '.jpg',bbox_inches = 'tight') # 画像の保存
    plt.show()  # 以上の設定で描画

    # テストデータのr2, RMSE, MAE
    print('r^2 for test data :', metrics.r2_score(y_test, estimated_y_test))
    print('RMSE for test data :', metrics.mean_squared_error(y_test, estimated_y_test) ** 0.5)
    print('MAE for test data :', metrics.mean_absolute_error(y_test, estimated_y_test))

    # テストデータの結果の保存
    y_test_for_save = pd.DataFrame(y_test)  # Series のため列名は別途変更
    y_test_for_save.columns = ['actual_y']
    y_error_test = y_test_for_save.iloc[:, 0] - estimated_y_test.iloc[:, 0]
    y_error_test = pd.DataFrame(y_error_test)  # Series のため列名は別途変更
    y_error_test.columns = ['error_of_y(actual_y-estimated_y)']
    results_test = pd.concat([estimated_y_test, y_test_for_save, y_error_test], axis=1)
    results_test.to_csv('{}/estimated_y_test.csv'.format(output_folder_name))  # 推定値を csv ファイルに保存。
    
    #　各種計算結果の保存
    with open('{}/R^2,RMSE,MAE.txt'.format(output_folder_name), 'w') as f:
        print('r^2 for training data :', metrics.r2_score(y_train, estimated_y_train), file =f)
        print('RMSE for training data :', metrics.mean_squared_error(y_train, estimated_y_train) ** 0.5, file = f)
        print('MAE for training data :', metrics.mean_absolute_error(y_train, estimated_y_train), file = f)
        print('r^2 for test data :', metrics.r2_score(y_test, estimated_y_test), file = f)
        print('RMSE for test data :', metrics.mean_squared_error(y_test, estimated_y_test) ** 0.5, file = f)
        print('MAE for test data :', metrics.mean_absolute_error(y_test, estimated_y_test), file = f)

# トレーニングデータとテストデータそれぞれのr2,rmse,maeを算出し、dataframe型で返す関数
def evaluation_function(model, x_train, y_train, x_test, y_test, output_folder_name):
    
    # トレーニングデータの推定
    estimated_y_train = model.predict(x_train) * y_train.std() + y_train.mean()  # y を推定し、スケールをもとに戻します
    
    # 算出した評価関数を格納する空のlist作成
    evaluation_list = []
    
    # r2,rmse,maeを算出し、evaluation_listに追加
    r2_train = metrics.r2_score(y_train, estimated_y_train)
    rmse_train = metrics.mean_squared_error(y_train, estimated_y_train) ** 0.5
    mae_train = metrics.mean_absolute_error(y_train, estimated_y_train)
    evaluation_list.extend([r2_train, rmse_train, mae_train])
    
    # テストデータの推定
    estimated_y_test = model.predict(x_test) * y_train.std() + y_train.mean()
        
    # r2,rmse,maeを算出し、evaluation_listに追加
    r2_test = metrics.r2_score(y_test, estimated_y_test)
    rmse_test = metrics.mean_squared_error(y_test, estimated_y_test) ** 0.5
    mae_test = metrics.mean_absolute_error(y_test, estimated_y_test)
    evaluation_list.extend([r2_test, rmse_test, mae_test])

    # listをdataframeに変換
    df_evaluation = pd.DataFrame([evaluation_list], index=[output_folder_name.split('/')[1]],
                                 columns=['r2_train', 'RMSE_train', 'MAE_train',
                                          'r2_test', 'RMSE_test', 'MAE_test'])
    return df_evaluation



# PLS におけるVIPを算出
def calculate_vips(model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(np.matmul(np.matmul(np.matmul(t.T,t),q.T), q)).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
        vips[i] = np.sqrt(p*(np.matmul(s.T, weight))/total_s)
    return vips

# SVRのハイパーパラメータγ
def gamma_optimization_with_variance(x, gammas):
    """
    DataFrame型もしくは array 型の x において、カーネル関数におけるグラム行列の分散を最大化することによって
    γ を最適化する関数

    Parameters
    ----------
    x: pandas.DataFrame or numpy.array
    gammas: list

    Returns
    -------
    optimized gamma : scalar
    
    """
    print('カーネル関数において、グラム行列の分散を最大化することによる γ の最適化')
    variance_of_gram_matrix = list()
    for index, ocsvm_gamma in enumerate(gammas):
        print(index + 1, '/', len(gammas))
        gram_matrix = np.exp(-ocsvm_gamma * cdist(x, x, metric='seuclidean'))
        variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
    return gammas[variance_of_gram_matrix.index(max(variance_of_gram_matrix))]


def plot_GPR_samples(x_obs, y_obs, x_test, y_m, y_s, output_folder_name):
    """
    Plot for sampled and estimated response
    """
    if len(y_obs.shape)==2:
        y_obs = y_obs.copy().reshape(-1,)
    n_samples = x_obs.shape[0] + x_test.shape[0]
    n_trains = x_obs.shape[0]
    plt.figure()
    plt.plot(range(n_trains), y_obs, 'o')
    plt.plot(range(n_trains, n_samples), y_m)
    plt.fill_between(range(n_trains, n_samples), y_m-y_s,y_m+y_s,
                    alpha=.3, color='b')
    plt.xlim([0, n_samples])
    plt.ylim(np.array([np.amin(y_m-y_s) + np.amin(y_m-y_s)/5, np.amax(y_m+y_s) + np.amax(y_m+y_s) / 5]))
    plt.xlabel('sample index'); plt.ylabel('$y$')
    plt.savefig(output_folder_name + '/' + 'GPR_plot' + '.jpg',bbox_inches = 'tight') # 画像の保存



def LWPLS_hyperparameter_CV(autoscaled_x_train, autoscaled_y_train,
                            autoscaled_x_test,max_number_of_principal_components,lwpls_lambdas,
                            fold_number):
    # CV によるハイパーパラメータの最適化
    autoscaled_x_train = np.array(autoscaled_x_train)
    autoscaled_y_train = np.array(autoscaled_y_train)
    y_train_array = np.array(autoscaled_y_train)
    r2cvs = np.empty(
        (min(np.linalg.matrix_rank(autoscaled_x_train), max_number_of_principal_components), len(lwpls_lambdas)))
    min_number = math.floor(autoscaled_x_train.shape[0] / fold_number)
    mod_numbers = autoscaled_x_train.shape[0] - min_number * fold_number
    index = np.matlib.repmat(np.arange(1, fold_number + 1, 1), 1, min_number).ravel()
    if mod_numbers != 0:
        index = np.r_[index, np.arange(1, mod_numbers + 1, 1)]
    np.random.seed(123)
    indexes_for_division_in_cv = np.random.permutation(index)
    for parameter_number, lambda_in_similarity in enumerate(lwpls_lambdas):
        estimated_y_in_cv = np.empty((len(y_train_array), r2cvs.shape[0]))
        for fold in np.arange(1, fold_number + 1, 1):
            autoscaled_x_train_in_cv = autoscaled_x_train[indexes_for_division_in_cv != fold, :]
            autoscaled_y_train_in_cv = autoscaled_y_train[indexes_for_division_in_cv != fold]
            autoscaled_x_validation_in_cv = autoscaled_x_train[indexes_for_division_in_cv == fold, :]

            estimated_y_validation_in_cv = lwpls(autoscaled_x_train_in_cv, autoscaled_y_train_in_cv,
                                                                  autoscaled_x_validation_in_cv, r2cvs.shape[0],
                                                                  lambda_in_similarity)
            estimated_y_in_cv[indexes_for_division_in_cv == fold, :] = estimated_y_validation_in_cv * y_train_array.std(
                ddof=1) + y_train_array.mean()

        estimated_y_in_cv[np.isnan(estimated_y_in_cv)] = 99999
        ss = (y_train_array - y_train_array.mean()).T.dot(y_train_array - y_train_array.mean())
        press = np.diag(
            (np.matlib.repmat(y_train_array.reshape(len(y_train_array), 1), 1,
                              estimated_y_in_cv.shape[1]) - estimated_y_in_cv).T.dot(
                np.matlib.repmat(y_train_array.reshape(len(y_train_array), 1), 1,
                                 estimated_y_in_cv.shape[1]) - estimated_y_in_cv))
        r2cvs[:, parameter_number] = 1 - press / ss

    best_candidate_number = np.where(r2cvs == r2cvs.max())

    optimal_component_number = best_candidate_number[0][0] + 1
    optimal_lambda_in_similarity = lwpls_lambdas[best_candidate_number[1][0]]
     
    return(optimal_component_number,optimal_lambda_in_similarity)


def lwpls(x_train, y_train, x_test, max_component_number, lambda_in_similarity):
    """
    Locally-Weighted Partial Least Squares (LWPLS)
    
    Predict y-values of test samples using LWPLS

    Parameters
    ----------
    x_train: numpy.array or pandas.DataFrame
        autoscaled m x n matrix of X-variables of training data,
        m is the number of training sammples and
        n is the number of X-variables
    y_train: numpy.array or pandas.DataFrame
        autoscaled m x 1 vector of a Y-variable of training data
    x_test: numpy.array or pandas.DataFrame
        k x n matrix of X-variables of test data, which is autoscaled with training data,
        and k is the number of test samples
    max_component_number: int
        number of maximum components
    lambda_in_similarity: float
        parameter in similarity matrix

    Returns
    -------
    estimated_y_test : numpy.array
        k x 1 vector of estimated y-values of test data
    """

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = np.reshape(y_train, (len(y_train), 1))
    x_test = np.array(x_test)

    estimated_y_test = np.zeros((x_test.shape[0], max_component_number))
    distance_matrix = cdist(x_train, x_test, 'euclidean')
    for test_sample_number in range(x_test.shape[0]):
        query_x_test = x_test[test_sample_number, :]
        query_x_test = np.reshape(query_x_test, (1, len(query_x_test)))
        distance = distance_matrix[:, test_sample_number]
        similarity = np.diag(np.exp(-distance / distance.std(ddof=1) / lambda_in_similarity))
        #        similarity_matrix = np.diag(similarity)

        y_w = y_train.T.dot(np.diag(similarity)) / similarity.sum()
        x_w = np.reshape(x_train.T.dot(np.diag(similarity)) / similarity.sum(), (1, x_train.shape[1]))
        centered_y = y_train - y_w
        centered_x = x_train - np.ones((x_train.shape[0], 1)).dot(x_w)
        centered_query_x_test = query_x_test - x_w
        estimated_y_test[test_sample_number, :] += y_w
        for component_number in range(max_component_number):
            w_a = np.reshape(centered_x.T.dot(similarity).dot(centered_y) / np.linalg.norm(
                centered_x.T.dot(similarity).dot(centered_y)), (x_train.shape[1], 1))
            t_a = np.reshape(centered_x.dot(w_a), (x_train.shape[0], 1))
            p_a = np.reshape(centered_x.T.dot(similarity).dot(t_a) / t_a.T.dot(similarity).dot(t_a),
                             (x_train.shape[1], 1))
            q_a = centered_y.T.dot(similarity).dot(t_a) / t_a.T.dot(similarity).dot(t_a)
            t_q_a = centered_query_x_test.dot(w_a)
            estimated_y_test[test_sample_number, component_number:] = estimated_y_test[test_sample_number,
                                                                                       component_number:] + t_q_a * q_a
            if component_number != max_component_number:
                centered_x = centered_x - t_a.dot(p_a.T)
                centered_y = centered_y - t_a * q_a
                centered_query_x_test = centered_query_x_test - t_q_a.dot(p_a.T)

    return estimated_y_test



def plot_for_lwpls(estimated_y_train,estimated_y_test, x_train, y_train, x_test, y_test, output_folder_name):
    # トレーニングデータの推定
    estimated_y_train = estimated_y_train * y_train.std() + y_train.mean()  # y を推定し、スケールをもとに戻します
    estimated_y_train = pd.DataFrame(estimated_y_train, index=x_train.index,
                                     columns=['estimated_y'])  # Pandas の DataFrame 型に変換。行の名前・列の名前も設定
    
    # トレーニングデータの実測値 vs. 推定値のプロット
    plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
    plt.figure(figsize=figure.figaspect(1))  # 図の形を正方形に
    plt.scatter(y_train, estimated_y_train.iloc[:, 0], c='blue')  # 実測値 vs. 推定値プロット
    y_max = max(y_train.max(), estimated_y_train.iloc[:, 0].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
    y_min = min(y_train.min(), estimated_y_train.iloc[:, 0].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定
    plt.xlabel('actual y')  # x 軸の名前
    plt.ylabel('estimated y')  # y 軸の名前
    plt.savefig(output_folder_name + '/' + 'training_y_y_plot' + '.jpg',bbox_inches = 'tight') # 画像の保存
    plt.show()  # 以上の設定で描画
    
    # トレーニングデータの実測値 vs. 推定値のプロット (サンプル名を表示)
    plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
    plt.figure(figsize=figure.figaspect(1))  # 図の形を正方形に
    plt.scatter(y_train, estimated_y_train.iloc[:, 0], c='blue')  # 実測値 vs. 推定値プロット
    y_max = max(y_train.max(), estimated_y_train.iloc[:, 0].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
    y_min = min(y_train.min(), estimated_y_train.iloc[:, 0].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定
    
    for sample_number in range(y_train.shape[0]):
        plt.text(y_train.iloc[sample_number], estimated_y_train.iloc[sample_number], y_train.index[sample_number],
             horizontalalignment='left', verticalalignment='top')
    plt.xlabel('actual y')  # x 軸の名前
    plt.ylabel('estimated y')  # y 軸の名前
    plt.savefig(output_folder_name + '/' + 'training_y_y_plotwith_name' + '.jpg',bbox_inches = 'tight') # 画像の保存
    plt.show()  # 以上の設定で描画

    # トレーニングデータのr2, RMSE, MAE
    print('r^2 for training data :', metrics.r2_score(y_train, estimated_y_train))
    print('RMSE for training data :', metrics.mean_squared_error(y_train, estimated_y_train) ** 0.5)
    print('MAE for training data :', metrics.mean_absolute_error(y_train, estimated_y_train))

    # トレーニングデータの結果の保存
    y_train_for_save = pd.DataFrame(y_train)  # Series のため列名は別途変更
    y_train_for_save.columns = ['actual_y']
    y_error_train = y_train_for_save.iloc[:, 0] - estimated_y_train.iloc[:, 0]
    y_error_train = pd.DataFrame(y_error_train)  # Series のため列名は別途変更
    y_error_train.columns = ['error_of_y(actual_y-estimated_y)']
    results_train = pd.concat([estimated_y_train, y_train_for_save, y_error_train], axis=1)
    results_train.to_csv('{}/estimated_y_train.csv'.format(output_folder_name))  # 推定値を csv ファイルに保存。

    # テストデータの推定
    estimated_y_test = estimated_y_test * y_train.std() + y_train.mean()  # y を推定し、スケールをもとに戻します
    estimated_y_test = pd.DataFrame(estimated_y_test, index=x_test.index,
                                    columns=['estimated_y'])  # Pandas の DataFrame 型に変換。行の名前・列の名前も設定
   
    # テストデータの実測値 vs. 推定値のプロット
    plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
    plt.figure(figsize=figure.figaspect(1))  # 図の形を正方形に
    plt.scatter(y_test, estimated_y_test.iloc[:, 0], c='blue')  # 実測値 vs. 推定値プロット
    y_max = max(y_test.max(), estimated_y_test.iloc[:, 0].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
    y_min = min(y_test.min(), estimated_y_test.iloc[:, 0].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定
    plt.xlabel('actual y')  # x 軸の名前
    plt.ylabel('estimated y')  # y 軸の名前
    plt.savefig(output_folder_name + '/' + 'test_y_y_plot' + '.jpg',bbox_inches = 'tight') # 画像の保存
    plt.show()  # 以上の設定で描画
    
    # テストデータの実測値 vs. 推定値のプロット (サンプル名を表示)
    plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
    plt.figure(figsize=figure.figaspect(1))  # 図の形を正方形に
    plt.scatter(y_test, estimated_y_test.iloc[:, 0], c='blue')  # 実測値 vs. 推定値プロット
    y_max = max(y_test.max(), estimated_y_test.iloc[:, 0].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
    y_min = min(y_test.min(), estimated_y_test.iloc[:, 0].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定


    for sample_number in range(y_test.shape[0]):
        plt.text(y_test.iloc[sample_number], estimated_y_test.iloc[sample_number], y_test.index[sample_number],
             horizontalalignment='left', verticalalignment='top')
    plt.xlabel('actual y')  # x 軸の名前
    plt.ylabel('estimated y')  # y 軸の名前
    plt.savefig(output_folder_name + '/' + 'test_y_y_plot_with_sample_name' + '.jpg',bbox_inches = 'tight') # 画像の保存
    plt.show()  # 以上の設定で描画

    # テストデータのr2, RMSE, MAE
    print('r^2 for test data :', metrics.r2_score(y_test, estimated_y_test))
    print('RMSE for test data :', metrics.mean_squared_error(y_test, estimated_y_test) ** 0.5)
    print('MAE for test data :', metrics.mean_absolute_error(y_test, estimated_y_test))

    # テストデータの結果の保存
    y_test_for_save = pd.DataFrame(y_test)  # Series のため列名は別途変更
    y_test_for_save.columns = ['actual_y']
    y_error_test = y_test_for_save.iloc[:, 0] - estimated_y_test.iloc[:, 0]
    y_error_test = pd.DataFrame(y_error_test)  # Series のため列名は別途変更
    y_error_test.columns = ['error_of_y(actual_y-estimated_y)']
    results_test = pd.concat([estimated_y_test, y_test_for_save, y_error_test], axis=1)
    results_test.to_csv('{}/estimated_y_test.csv'.format(output_folder_name))  # 推定値を csv ファイルに保存。
    
    #　各種計算結果の保存
    with open('{}/R^2,RMSE,MAE.txt'.format(output_folder_name), 'w') as f:
        print('r^2 for training data :', metrics.r2_score(y_train, estimated_y_train), file =f)
        print('RMSE for training data :', metrics.mean_squared_error(y_train, estimated_y_train) ** 0.5, file = f)
        print('MAE for training data :', metrics.mean_absolute_error(y_train, estimated_y_train), file = f)
        print('r^2 for test data :', metrics.r2_score(y_test, estimated_y_test), file = f)
        print('RMSE for test data :', metrics.mean_squared_error(y_test, estimated_y_test) ** 0.5, file = f)
        print('MAE for test data :', metrics.mean_absolute_error(y_test, estimated_y_test), file = f)

# トレーニングデータとテストデータそれぞれのr2,rmse,maeを算出し、dataframe型で返す関数
def evaluation_function_for_lwpls(estimated_y_train, estimated_y_test, x_train, y_train, x_test, y_test, output_folder_name):
    
    # 算出した評価関数を格納する空のlist作成
    evaluation_list = []
    
    # r2,rmse,maeを算出し、evaluation_listに追加
    r2_train = metrics.r2_score(y_train, estimated_y_train)
    rmse_train = metrics.mean_squared_error(y_train, estimated_y_train) ** 0.5
    mae_train = metrics.mean_absolute_error(y_train, estimated_y_train)
    evaluation_list.extend([r2_train, rmse_train, mae_train])
    
    # r2,rmse,maeを算出し、evaluation_listに追加
    r2_test = metrics.r2_score(y_test, estimated_y_test)
    rmse_test = metrics.mean_squared_error(y_test, estimated_y_test) ** 0.5
    mae_test = metrics.mean_absolute_error(y_test, estimated_y_test)
    evaluation_list.extend([r2_test, rmse_test, mae_test])

    # listをdataframeに変換
    df_evaluation = pd.DataFrame([evaluation_list], index=[output_folder_name.split('/')[1]],
                                 columns=['r2_train', 'RMSE_train', 'MAE_train',
                                          'r2_test', 'RMSE_test', 'MAE_test'])
    return df_evaluation

def shap_for_any_model(output_folder_name, model, x_train, x_test):
        
    # SHAPの出力結果の保存先のディレクトリ作成
    os.makedirs(output_folder_name + '/SHAP', exist_ok=True)
    
    # SHAPのKernelExplainerオブジェクト作成
    explainer = shap.KernelExplainer(model.predict, x_train)
    
    # SHAP値の算出
    shap_values = explainer.shap_values(x_test)
    
    # SHAP値で特徴量軸を比較
    plt.figure()
    shap.summary_plot(shap_values, features=x_test, show=False)
    pylab.tight_layout()
    plt.savefig(output_folder_name + '/SHAP' + '/summary_plot.png')
    
    # SHAP値の絶対値で特徴量軸を比較
    plt.figure()
    shap.summary_plot(shap_values, x_test, plot_type='bar', show=False)
    pylab.tight_layout()
    plt.savefig(output_folder_name + '/SHAP' + '/abs_summary_plot.png')
    
    # 各サンプルの特徴量の寄与を算出
    os.makedirs(output_folder_name + '/SHAP' + '/waterfall_plot', exist_ok=True)
    for i in range(len(x_test)):
        plt.figure()
        shap.plots._waterfall.waterfall_legacy(expected_value=explainer.expected_value,
                                               shap_values=shap_values[i],
                                               features=x_test.iloc[i],
                                               feature_names=x_test.columns, show=False)
        pylab.tight_layout()
        plt.savefig(output_folder_name + '/SHAP' + '/waterfall_plot' + '/' + x_test.index[i] + '.png')
    
    # shap値のcsv保存
    shap_values_df = pd.DataFrame(data=shap_values, index=x_test.index, columns=x_test.columns)
    shap_values_df.T.to_csv(output_folder_name + '/SHAP' + '/shap_values.csv')

    
    
def shap_for_tree_model(output_folder_name, model, x_test):
    
    # SHAPの出力結果の保存先のディレクトリ作成
    os.makedirs(output_folder_name + '/SHAP', exist_ok=True)
    
    # SHAPのKernelExplainerオブジェクト作成
    explainer = shap.TreeExplainer(model)
    
    # SHAP値の算出
    shap_values = explainer(x_test)
    
    # SHAP値で特徴量軸を比較
    plt.figure()
    shap.summary_plot(shap_values=shap_values, features=x_test,
                      feature_names=x_test.columns, show=False)
    pylab.tight_layout()
    plt.savefig(output_folder_name + '/SHAP' + '/summary_plot.png')
    
    # SHAP値の絶対値で特徴量軸を比較
    plt.figure()
    shap.summary_plot(shap_values=shap_values, features=x_test,
                      feature_names=x_test.columns, plot_type='bar', show=False)
    pylab.tight_layout()
    plt.savefig(output_folder_name + '/SHAP' + '/abs_summary_plot.png')
    
    # 各サンプルの特徴量の寄与を算出
    os.makedirs(output_folder_name + '/SHAP' + '/waterfall_plot', exist_ok=True)
    for i in range(len(x_test)):
        plt.figure()
        shap.waterfall_plot(shap_values[i], show=False)
        pylab.tight_layout()
        plt.savefig(output_folder_name + '/SHAP' + '/waterfall_plot' + '/' + x_test.index[i] + '.png')
  
    # shap値のcsv保存   
    shap_values_df = pd.DataFrame(data=shap_values.values, index=x_test.index, columns=x_test.columns)
    shap_values_df.T.to_csv(output_folder_name + '/SHAP' + '/shap_values.csv')
  
def shap_for_dt_rf(output_folder_name, model, x_test):
    
    # SHAPの出力結果の保存先のディレクトリ作成
    os.makedirs(output_folder_name + '/SHAP', exist_ok=True)
    
    # SHAPのKernelExplainerオブジェクト作成
    explainer = shap.TreeExplainer(model)
    
    # SHAP値の算出
    shap_values = explainer(x_test)
    
    # SHAP値で特徴量軸を比較
    plt.figure()
    shap.summary_plot(shap_values=shap_values, features=x_test,
                      feature_names=x_test.columns, show=False)
    pylab.tight_layout()
    plt.savefig(output_folder_name + '/SHAP' + '/summary_plot.png')
    
    # SHAP値の絶対値で特徴量軸を比較
    plt.figure()
    shap.summary_plot(shap_values=shap_values, features=x_test,
                      feature_names=x_test.columns, plot_type='bar', show=False)
    pylab.tight_layout()
    plt.savefig(output_folder_name + '/SHAP' + '/abs_summary_plot.png')
    
    # 各サンプルの特徴量の寄与を算出
    os.makedirs(output_folder_name + '/SHAP' + '/decision_plot', exist_ok=True)
    for i in range(len(x_test)):
        plt.figure()
        shap.decision_plot(explainer.expected_value, explainer.shap_values(x_test)[i], x_test.iloc[i,:], show=False)
        pylab.tight_layout()
        plt.savefig(output_folder_name + '/SHAP' + '/decision_plot' + '/' + x_test.index[i] + '.png')
    
    # shap値のcsv保存 
    shap_values_df = pd.DataFrame(data=shap_values.values, index=x_test.index, columns=x_test.columns)
    shap_values_df.T.to_csv(output_folder_name + '/SHAP' + '/shap_values.csv')
  
    
def shap_for_dnn_model(output_folder_name, model, x_train, x_test):  
    
    # SHAPの出力結果の保存先のディレクトリ作成
    os.makedirs(output_folder_name + '/SHAP', exist_ok=True)
    
    # SHAPのKernelExplainerオブジェクト作成
    explainer = shap.DeepExplainer(model, x_train)
    
    # SHAP値の算出のため、x_testをnumpyに変換
    x_test_np = x_test.values
    
    # SHAP値の算出
    shap_values = explainer.shap_values(x_test_np)
    
    # SHAP値で特徴量軸を比較
    plt.figure()
    shap.summary_plot(shap_values[0], features=x_test, show=False)
    pylab.tight_layout()
    plt.savefig(output_folder_name + '/SHAP' + '/summary_plot.png')
    
    # SHAP値の絶対値で特徴量軸を比較
    plt.figure()
    shap.summary_plot(shap_values, x_test, plot_type='bar', show=False)
    pylab.tight_layout()
    plt.savefig(output_folder_name + '/SHAP' + '/abs_summary_plot.png')
    
    # 各サンプルの特徴量の寄与を算出
    os.makedirs(output_folder_name + '/SHAP' + '/decision_plot', exist_ok=True)
    for i in range(len(x_test)):
        plt.figure()
        shap.decision_plot(explainer.expected_value, shap_values[0][i,:], x_test.iloc[i,:], show=False)
        pylab.tight_layout()
        plt.savefig(output_folder_name + '/SHAP' + '/decision_plot' + '/' + x_test.index[i] + '.png')
        
    # shap値のcsv保存
    shap_values_df = pd.DataFrame(data=shap_values[0], index=x_test.index, columns=x_test.columns)
    shap_values_df.T.to_csv(output_folder_name + '/SHAP' + '/shap_values.csv')


# optunaを用いたLightGBMのハイパーパラメータを最適化する関数
def lgb_optuna(best_n_estimators_for_optuna, fold_number, autoscaled_x_train, autoscaled_y_train, y_train):
   
    def objective(trial):
   
        param = {
            'verbosity': -1,
            'boosting_type': trial.suggest_categorical('boosting', ['gbdt', 'dart', 'goss']),
            'num_leaves': trial.suggest_int('gbmnum_leaves', 10, 1000),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 1.0)
        }
        
        if param['boosting_type'] == 'dart':
            param['drop_rate'] = trial.suggest_loguniform('drop_rate', 1e-8, 1.0)
            param['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
        if param['boosting_type'] == 'goss':
            param['top_rate'] = trial.suggest_uniform('top_rate', 0.0, 1.0)
            param['other_rate'] = trial.suggest_uniform('other_rate', 0.0, 1.0 - param['top_rate'])
        
        model = lgb.LGBMRegressor(**param, n_estimators=best_n_estimators_for_optuna)
        estimated_y_in_cv = model_selection.cross_val_predict(model, autoscaled_x_train, autoscaled_y_train, cv=fold_number)
        estimated_y_in_cv = estimated_y_in_cv * y_train.std() + y_train.mean()
        r2 = metrics.r2_score(y_train, estimated_y_in_cv)
        return 1.0 - r2
    
    return objective


# optunaを用いたXGBoostのハイパーパラメータを最適化する関数
def xgb_optuna(best_n_estimators_for_optuna, fold_number, autoscaled_x_train, autoscaled_y_train, y_train):

    # optunaを用いたハイパーパラメータの最適化
    def objective(trial):
        param = {
            'objective': 'reg:squarederror',
            'booster': trial.suggest_categorical('booster', ['gbtree']),
            'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0)
        }
    
        if param['booster'] == 'gbtree' or param['booster'] == 'dart':
            param['max_depth'] = trial.suggest_int('max_depth', 1, 9)
            param['eta'] = trial.suggest_loguniform('eta', 1e-8, 1.0)
            param['gamma'] = trial.suggest_loguniform('gamma', 1e-8, 1.0)
            param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        if param['booster'] == 'dart':
            param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
            param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
            param['rate_drop'] = trial.suggest_loguniform('rate_drop', 1e-8, 1.0)
            param['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
    
        model = xgb.XGBRegressor(**param, n_estimators=best_n_estimators_for_optuna)
        estimated_y_in_cv = model_selection.cross_val_predict(model, autoscaled_x_train, autoscaled_y_train, cv=fold_number)
        estimated_y_in_cv = estimated_y_in_cv * y_train.std() + y_train.mean()
        r2 = metrics.r2_score(y_train, estimated_y_in_cv)
        return 1.0 - r2
    
    return objective

# optunaを用いたGBDTのハイパーパラメータを最適化する関数
def gbdt_optuna(best_n_estimators_for_optuna, fold_number, autoscaled_x_train, autoscaled_y_train, y_train):
    
    def objective(trial):
        param = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1),
            'max_depth': trial.suggest_int('max_depth', 1, 9),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 3, 20),
            'max_features': trial.suggest_loguniform('max_features', 0.1, 1.0)
        }
        
        model = GradientBoostingRegressor(**param, n_estimators=best_n_estimators_for_optuna)
        estimated_y_in_cv = model_selection.cross_val_predict(model, autoscaled_x_train, autoscaled_y_train, cv=fold_number)
        estimated_y_in_cv = estimated_y_in_cv * y_train.std() + y_train.mean()
        r2 = metrics.r2_score(y_train, estimated_y_in_cv)
        return 1.0 - r2
    
    return objective


# optunaを用いたCatBoostのハイパーパラメータを最適化する関数
def cb_optuna(fold_number, autoscaled_x_train, autoscaled_y_train, y_train):
    
    def objective(trial):
        param = {
        'iterations' : trial.suggest_int('iterations', 50, 300),
        'depth' : trial.suggest_int('depth', 4, 10)  ,                                     
        'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.3) ,             
        'random_strength' : trial.suggest_int('random_strength', 0, 100),
        'bagging_temperature' : trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
        'od_type' : trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
        'od_wait' : trial.suggest_int('od_wait', 10, 50)
                             }
    
        model = cb.CatBoostRegressor(**param, allow_writing_files=False)
        estimated_y_in_cv = model_selection.cross_val_predict(model, autoscaled_x_train, autoscaled_y_train, cv=fold_number)
        estimated_y_in_cv = estimated_y_in_cv * y_train.std() + y_train.mean()
        r2 = metrics.r2_score(y_train, estimated_y_in_cv)
        return 1.0 - r2
    
    return objective

# DNNの学習曲線をプロットし、保存する関数
def acc_and_loss_plot(hist, output_folder_name):
    
    # mae
    fig = plt.figure(figsize=(8,6))
    plt.plot(hist.history["mae"], label = "mae", marker = "o")
    plt.plot(hist.history["val_mae"], label = "val_mae", marker = "o")
    plt.title('mae history')
    plt.xlabel('epochs')
    plt.ylabel('mae')
    plt.legend(loc='upper right')
    plt.show()
    fig.savefig(output_folder_name + '/mae.png')
       
    # loss
    fig = plt.figure(figsize=(8,6))
    plt.plot(hist.history['loss'], label="train", marker = "o")
    plt.plot(hist.history['val_loss'], label="test", marker = "o")
    plt.title('loss history')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.show()
    fig.savefig(output_folder_name + '/loss.png')
