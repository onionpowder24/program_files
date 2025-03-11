# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import glob
import math
import os
import optuna
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn.gaussian_process as gp
from sklearn.gaussian_process import kernels
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.linear_model import Ridge, Lasso, ElasticNet, ElasticNetCV
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.initializers import he_normal
import catboost as  cb
import lightgbm as lgb
import xgboost as xgb
import pickle
import time

import cv2
import img2pdf
from PIL import Image
from natsort import natsorted
import shutil

import functions

# SHAPによるモデルの評価を行うか
evaluate_with_shap = True # 行うならtrue、行わないならfalse

# 処理時間の計測を行うか
exe_time = True

# 作業ディレクトリ内のcsv ファイルを読み込む
files = glob.glob('*.csv') # ワイルドカードが使用可能、今回はcsvファイルを取得
for file in files:
    dataset = pd.read_csv(file,index_col=0)

# テストサンプルの割合とCVのfold数
number_of_test_samples = math.ceil(0.2*dataset.shape[0])  # テストデータのサンプル数(全体の２割に設定)
fold_number = 10  # N-fold CV の N

# バリデーションデータのサンプル数の割合(LightGBM, XGBoost, GBDT, DNNの検証に使用)
fraction_of_validation_samples = 0.2 

# PLSのハイパーパラメータ
max_number_of_principal_components = math.ceil(dataset.shape[1]*0.7)  # 使用する主成分の最大数

# SVRのハイパーパラメータ候補
svr_cs = 2 ** np.arange(-5, 11, dtype=float)  # C の候補
svr_epsilons = 2 ** np.arange(-10, 1, dtype=float)  # ε の候補
svr_gammas = 2 ** np.arange(-20, 11, dtype=float)  # γ の候補

# DTにおける木の深さの最大値の候補
max_depths = np.arange(1, 31) # 木の深さの最大値の候補

# RFのハイパーパラメータ
ratios_of_x = np.arange(0.1, 1.1, 0.1) # 用いる説明変数の割合の候補
n_estimators =  500 # サブデータセットの数

# LWPLSのハイパーパラメータ
# 主成分の最大数はPLSと同じ数を使用する
lwpls_lambdas = 2 ** np.arange(-9, 6, dtype=float)

# Lassoのハイパーパラメータ候補
lasso_lambdas = np.arange(0.01, 0.71, 0.01, dtype=float)  # L1 weight in LASSO

# Ridgeのハイパーパラメータ候補
ridge_lambdas = np.arange(0.01, 0.71, 0.01, dtype=float)  # L2 weight in ridge regression

# ElasticNetのハイパーパラメータ候補
elastic_net_lambdas = np.arange(0.01, 0.71, 0.01, dtype=float)  # Lambda in elastic net
elastic_net_alphas = np.arange(0.01, 1.00, 0.01, dtype=float)  # Alpha in elastic net

# 勾配Boosting系(LightGBM, XGBoost, GBDT)のハイパーパラメータ候補
decision_of_submodel = False # サブモデルの数を決め打ちにする場合はTrue、バリデーションデータを用いてearly_stoppingで決める場合はFalse
lgb_number_of_sub_models = 500 # LightGBMのサブモデルの数
xgb_number_of_sub_models = 500 # XGBoostのサブモデルの数
gbdt_number_of_sub_models = 500 # GBDTのサブモデルの数

# 目的変数と説明変数
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
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std()
autoscaled_x_test = (x_test - x_train.mean()) / x_train.std()

# 学習データをtrainデータとバリデーションデータに分割
autoscaled_x_train_tmp, autoscaled_x_validation, autoscaled_y_train_tmp, autoscaled_y_validation = train_test_split(autoscaled_x_train,
                                                                                                                      autoscaled_y_train,
                                                                                                                      test_size=fraction_of_validation_samples,
                                                                                                                      random_state=123)
# DNNのハイパーパラメータ
num_of_unit_1 = int((len(autoscaled_x_train_tmp.columns) + 1) * 2/3) # 1つ目の中間層のユニット数(デフォルトは(入力層+出力層)×2/3)
num_of_unit_2 = int(num_of_unit_1 / 2) # 2つ目の中間層のユニット数(デフォルトは1層目の半分)
epochs = 100 # エポック数
batch_size=len(autoscaled_x_train_tmp.index) # バッチサイズ

#　各回帰モデルのr2,rmse,maeを保持するための空のdataframe作成
all_df_evaluation = pd.DataFrame(index=[], columns=['r2_train', 'RMSE_train', 'MAE_train',
                                                    'r2_test', 'RMSE_test', 'MAE_test'])

# 各モデルの名前を保持するための空のリストを作成
model_name_list = []

# 各モデルの実行時間を保持するための空のリストを作成
model_exe_time = {}

'''
PLS--------------------------------------------------------------------------------------------------------
'''
if exe_time:
    # 計測開始
    start = time.time()

#出力フォルダ作成
output_folder_name = './PLS'
os.makedirs(output_folder_name, exist_ok=True)
model_name_list.append(output_folder_name.split('/')[1])
    
# output_folder_name内に保存したモデルがなければモデルの構築を行い、あればモデルの構築をとばして推定へ
if os.path.exists(output_folder_name + '/model.pkl') is False:

    # CV による成分数の最適化
    components = []  # 空の list の変数を作成して、成分数をこの変数に追加していく
    r2_in_cv_all = []  # 空の list の変数を作成して、成分数ごとのクロスバリデーション後の r2 をこの変数に追加
    for component in range(1, min(np.linalg.matrix_rank(autoscaled_x_train), max_number_of_principal_components) + 1):
        # PLS
        model = PLSRegression(n_components=component)  # PLS モデルの宣言
        estimated_y_in_cv = pd.DataFrame(cross_val_predict(model, autoscaled_x_train, autoscaled_y_train,
                                                               cv=fold_number))  # クロスバリデーション推定値の計算し、DataFrame型に変換
        estimated_y_in_cv = estimated_y_in_cv * y_train.std() + y_train.mean()  # スケールをもとに戻す
        r2_in_cv = metrics.r2_score(y_train, estimated_y_in_cv)  # r2 を計算
        print(component, r2_in_cv)  # 成分数と r2 を表示
        r2_in_cv_all.append(r2_in_cv)  # r2 を追加
        components.append(component)  # 成分数を追加
    
    # 成分数ごとの CV 後の r2 をプロットし、CV 後のr2が最大のときを最適成分数に
    optimal_component_number = functions.plot_and_selection_of_hyperparameter_in_cv(components, r2_in_cv_all,
                                                                              'number of components',
                                                                              'r2 in cross-validation',
                                                                               output_folder_name)
    print('\nCV で最適化された成分数 :', optimal_component_number)
    
    # 最適化された成分数の保存
    with open('{}/optimal_component_number.txt'.format(output_folder_name), 'w') as f:
        print('CV で最適化された成分数 :', optimal_component_number, file = f)
    
    # PLS
    model = PLSRegression(n_components=optimal_component_number)  # モデルの宣言
    model.fit(autoscaled_x_train, autoscaled_y_train)  # モデルの構築
    
    # 標準回帰係数
    standard_regression_coefficients = pd.DataFrame(model.coef_, index=x_train.columns,
                                                        columns=['standard_regression_coefficients'])
    standard_regression_coefficients.to_csv('{}/pls_standard_regression_coefficients.csv'.format(output_folder_name))  # csv ファイルに保存。
    
    # モデルの保存
    with open(output_folder_name + '/model.pkl', mode='wb') as fp:
        pickle.dump(model, fp)
        
# モデルの読み込み
with open(output_folder_name + '/model.pkl', mode='rb') as fp:
    model = pickle.load(fp)

# トレーニングデータ・テストデータの推定、実測値 vs. 推定値のプロット、r2, RMSE, MAE の値の表示、推定値の保存
functions.estimation_and_performance_check_in_regression_train_and_test(model, autoscaled_x_train, y_train,
                                                                               autoscaled_x_test, y_test,
                                                                               output_folder_name)

# 変数重要度VIPの計算とcsv保存
vip = functions.calculate_vips(model)
vip = pd.DataFrame(vip, index = x_train.columns, columns = ['VIP'])
vip.to_csv('{}/pls_vip.csv'.format(output_folder_name))

# 学習データとテストデータのr2,rmse,maeを格納したdataframeの作成
df_evaluation = functions.evaluation_function(model, autoscaled_x_train, y_train,
                                    autoscaled_x_test, y_test, output_folder_name)

# all_df_evaluationに追加
all_df_evaluation = pd.concat([all_df_evaluation, df_evaluation])

if exe_time:
    
    # 計測終了
    process_time = time.time() - start
    
    #計測時間を辞書に追加
    model_exe_time[output_folder_name.split('/')[1]] = process_time

'''
SVR --------------------------------------------------------------------------------------------------------
'''

if exe_time:
    # 計測開始
    start = time.time()

output_folder_name = './SVR'
os.makedirs(output_folder_name, exist_ok=True)
model_name_list.append(output_folder_name.split('/')[1])

# output_folder_name内に保存したモデルがなければモデルの構築を行い、あればモデルの構築をとばして推定へ
if os.path.exists(output_folder_name + '/model.pkl') is False:
    
    # グラム行列の分散を最大化することによる γ の最適化
    optimal_svr_gamma = functions.gamma_optimization_with_variance(autoscaled_x_train, svr_gammas)
    
    # CV による ε の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_svr_gamma), {'epsilon': svr_epsilons},
                                   cv=fold_number, iid=False)
    model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
    optimal_svr_epsilon = model_in_cv.best_params_['epsilon']
    
    # CV による C の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma),
                                   {'C': svr_cs}, cv=fold_number, iid=False)
    model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
    optimal_svr_c = model_in_cv.best_params_['C']
    
    # CV による γ の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, C=optimal_svr_c),
                                   {'gamma': svr_gammas}, cv=fold_number, iid=False)
    model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
    optimal_svr_gamma = model_in_cv.best_params_['gamma']
    
    # 最適化された C, ε, γ
    print('C : {0}\nε : {1}\nGamma : {2}'.format(optimal_svr_c, optimal_svr_epsilon, optimal_svr_gamma))
    
    # 最適化されたC, ε, γ の保存
    with open('{}/C,ε,γ.txt'.format(output_folder_name), 'w') as f:
        print('C : {0}\nε : {1}\nGamma : {2}'.format(optimal_svr_c, optimal_svr_epsilon, optimal_svr_gamma), file = f)
    
    # SVR
    model = svm.SVR(kernel='rbf', C=optimal_svr_c, epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma)  # モデルの宣言
    model.fit(autoscaled_x_train, autoscaled_y_train)  # モデルの構築
    
    # モデルの保存
    with open(output_folder_name + '/model.pkl', mode='wb') as fp:
        pickle.dump(model, fp)

# モデルの読み込み
with open(output_folder_name + '/model.pkl', mode='rb') as fp:
    model = pickle.load(fp)
   
# トレーニングデータ・テストデータの推定、実測値 vs. 推定値のプロット、r2, RMSE, MAE の値の表示、推定値の保存
functions.estimation_and_performance_check_in_regression_train_and_test(model, autoscaled_x_train, y_train,
                                                                               autoscaled_x_test, y_test,
                                                                               output_folder_name)


# SHAPによるモデルの解釈
if evaluate_with_shap:
    functions.shap_for_any_model(output_folder_name, model, autoscaled_x_train, autoscaled_x_test)

# 学習データとテストデータのr2,rmse,maeを格納したdataframeの作成
df_evaluation = functions.evaluation_function(model, autoscaled_x_train, y_train,
                                    autoscaled_x_test, y_test, output_folder_name)

# all_df_evaluationに追加
all_df_evaluation = pd.concat([all_df_evaluation, df_evaluation])

if exe_time:
    
    # 計測終了
    process_time = time.time() - start
    
    #計測時間を辞書に追加
    model_exe_time[output_folder_name.split('/')[1]] = process_time

'''
DT(決定木)----------------------------------------------------------------------------------------------------------       
'''

if exe_time:
    # 計測開始
    start = time.time()

output_folder_name = './DT'
os.makedirs(output_folder_name, exist_ok=True)
model_name_list.append(output_folder_name.split('/')[1])

# output_folder_name内に保存したモデルがなければモデルの構築を行い、あればモデルの構築をとばして推定へ
if os.path.exists(output_folder_name + '/model.pkl') is False:
    
    r2cvs = [] # 空の list。木の深さの最大値の候補ごとに、クロスバリデーション後の r2 を入れていく
    for max_depth in max_depths:
        model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=3,random_state = 123)
        estimated_y_in_cv = cross_val_predict(model, autoscaled_x_train, autoscaled_y_train, cv=fold_number)
        r2cvs.append(metrics.r2_score(autoscaled_y_train, estimated_y_in_cv))
    
    optimal_max_depth = max_depths[r2cvs.index(max(r2cvs))] # クロスバリデーション後のr2が最大となる木の深さ
    
    # 最適化された木の深さ保存
    with open('{}/optimal_max_depth.txt'.format(output_folder_name), 'w') as f:
            print('CV で最適化された木の深さ :', optimal_max_depth, file = f)
    
    model = DecisionTreeRegressor(max_depth=optimal_max_depth, min_samples_leaf=3) # DTモデルの宣言
    model.fit(autoscaled_x_train, autoscaled_y_train) # DT モデル構築
    
    # 構築されたDTモデルの確認
    # tree.dot をGraphvizで開くと木の内容を確認できる
    # 今回は標準化しているので分かりづらい
    with open('{}/tree.dot'.format(output_folder_name), 'w') as f:
        export_graphviz(model, out_file=f, feature_names=x_train.columns, class_names=y_train.name)
        
    # モデルの保存
    with open(output_folder_name + '/model.pkl', mode='wb') as fp:
        pickle.dump(model, fp)

# モデルの読み込み
with open(output_folder_name + '/model.pkl', mode='rb') as fp:
    model = pickle.load(fp)

# トレーニングデータ・テストデータの推定、実測値 vs. 推定値のプロット、r2, RMSE, MAE の値の表示、推定値の保存
functions.estimation_and_performance_check_in_regression_train_and_test(model, autoscaled_x_train, y_train,
                                                                               autoscaled_x_test, y_test,
                                                                               output_folder_name)

# SHAPによるモデルの解釈
if evaluate_with_shap:
    functions.shap_for_dt_rf(output_folder_name, model, autoscaled_x_test)

# 学習データとテストデータのr2,rmse,maeを格納したdataframeの作成
df_evaluation = functions.evaluation_function(model, autoscaled_x_train, y_train,
                                    autoscaled_x_test, y_test, output_folder_name)

# all_df_evaluationに追加
all_df_evaluation = pd.concat([all_df_evaluation, df_evaluation])

if exe_time:
    
    # 計測終了
    process_time = time.time() - start
    
    #計測時間を辞書に追加
    model_exe_time[output_folder_name.split('/')[1]] = process_time



'''
DT( 決定木：非標準化、木構造の可視化)----------------------------------------------------------------------------------------------------------
'''

if exe_time:
    # 計測開始
    start = time.time()

output_folder_name = './DT_not_autoscaled'
os.makedirs(output_folder_name, exist_ok=True)

# output_folder_name内に保存したモデルがなければモデルの構築を行い、あればモデルの構築をとばして推定へ
if os.path.exists(output_folder_name + '/model.pkl') is False:

    r2cvs = [] # 空の list。木の深さの最大値の候補ごとに、クロスバリデーション後の r2 を入れていく
    for max_depth in max_depths:
        model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=3,random_state = 123)
        estimated_y_in_cv = cross_val_predict(model, x_train, y_train, cv=fold_number)
        r2cvs.append(metrics.r2_score(y_train, estimated_y_in_cv))
    
    optimal_max_depth = max_depths[r2cvs.index(max(r2cvs))] # クロスバリデーション後のr2が最大となる木の深さ
    
    # 最適化された木の深さ保存
    with open('{}/optimal_max_depth.txt'.format(output_folder_name), 'w') as f:
            print('CV で最適化された木の深さ :', optimal_max_depth, file = f)
    
    model = DecisionTreeRegressor(max_depth=optimal_max_depth, min_samples_leaf=3) # DTモデルの宣言
    model.fit(x_train, y_train) # DT モデル構築
        
    # モデルの保存
    with open(output_folder_name + '/model.pkl', mode='wb') as fp:
        pickle.dump(model, fp)
        
# モデルの読み込み
with open(output_folder_name + '/model.pkl', mode='rb') as fp:
    model = pickle.load(fp)

# 構築されたDTモデルの確認
# tree.dot をGraphvizで開くと木の内容を確認できる
with open('{}/tree.dot'.format(output_folder_name), 'w') as f:
    export_graphviz(model, out_file=f, feature_names=x_train.columns, class_names=y_train.name)

if exe_time:
    
    # 計測終了
    process_time = time.time() - start
    
    #計測時間を辞書に追加
    model_exe_time[output_folder_name.split('/')[1]] = process_time


'''
RF----------------------------------------------------------------------------------------------------------
'''

if exe_time:
    # 計測開始
    start = time.time()

output_folder_name = './RF'
os.makedirs(output_folder_name, exist_ok=True)
model_name_list.append(output_folder_name.split('/')[1])

# output_folder_name内に保存したモデルがなければモデルの構築を行い、あればモデルの構築をとばして推定へ
if os.path.exists(output_folder_name + '/model.pkl') is False:

    # ハイパーパラメータをOOB(Out-Of-Bag)正解率で最適化
    r2_oob = [] # 空の list。説明変数の数の割合ごとに、OOB における r2 を入れていく
    for ratio_of_x in ratios_of_x:
        model = RandomForestRegressor(n_estimators=500, max_features=ratio_of_x, oob_score=True, random_state= 123)
        model.fit(autoscaled_x_train, autoscaled_y_train)
        r2_oob.append(model.oob_score_)

    # 成分数ごとの CV 後の r2 をプロットし、CV 後のr2が最大のときを最適成分数に
    optimal_ratio_of_x = functions.plot_and_selection_of_hyperparameter_in_oob(ratios_of_x.tolist(),r2_oob,
                                                                          'ratios_of_x','r2 in OOB',
                                                                          output_folder_name)
    
    # OOB正解率で最適化した説明変数の数を用いてモデル構築
    model = RandomForestRegressor(n_estimators = n_estimators, max_features=optimal_ratio_of_x, oob_score=True)
    model.fit(autoscaled_x_train,autoscaled_y_train)
    
    # 構築されたRFモデルにおける説明変数ｘの重要度
    RF_feature_importance = model.feature_importances_
    RF_feature_importance = pd.DataFrame(RF_feature_importance, index = x_train.columns, columns = ['RF_feature_importance'])
    RF_feature_importance.to_csv('{}/RF_feature_importance.csv'.format(output_folder_name))
    
    # モデルの保存
    with open(output_folder_name + '/model.pkl', mode='wb') as fp:
        pickle.dump(model, fp)

# モデルの読み込み
with open(output_folder_name + '/model.pkl', mode='rb') as fp:
    model = pickle.load(fp)

# トレーニングデータ・テストデータの推定、実測値 vs. 推定値のプロット、r2, RMSE, MAE の値の表示、推定値の保存
functions.estimation_and_performance_check_in_regression_train_and_test(model, autoscaled_x_train, y_train,
                                                                               autoscaled_x_test, y_test,
                                                                               output_folder_name)
# SHAPによるモデルの解釈
if evaluate_with_shap:
    functions.shap_for_dt_rf(output_folder_name, model, autoscaled_x_test)

# 学習データとテストデータのr2,rmse,maeを格納したdataframeの作成
df_evaluation = functions.evaluation_function(model, autoscaled_x_train, y_train,
                                    autoscaled_x_test, y_test, output_folder_name)

# all_df_evaluationに追加
all_df_evaluation = pd.concat([all_df_evaluation, df_evaluation])

if exe_time:
    
    # 計測終了
    process_time = time.time() - start
    
    #計測時間を辞書に追加
    model_exe_time[output_folder_name.split('/')[1]] = process_time

'''
GPR (Gaussian Process Regression)----------------------------------------------------------------------------------------------------------
'''

if exe_time:
    # 計測開始
    start = time.time()

output_folder_name = './GPR'
os.makedirs(output_folder_name, exist_ok=True)
model_name_list.append(output_folder_name.split('/')[1])

# output_folder_name内に保存したモデルがなければモデルの構築を行い、あればモデルの構築をとばして推定へ
if os.path.exists(output_folder_name + '/model.pkl') is False:

    # カーネル関数の定義
    # 周辺尤度最大化の過程で不要なカーネル関数は重みが小さくなり、適切なカーネルの組み合わせが選ばれる。
    k1 = kernels.Sum(C()*kernels.RBF(), C()*kernels.RationalQuadratic())
    k2 = kernels.Sum(C()*kernels.Matern(), C()*kernels.ExpSineSquared())
    k3 = kernels.Sum(C()*kernels.DotProduct(), kernels.ConstantKernel())
    ks = kernels.Sum(k1, k2)
    ks = kernels.Sum(ks, k3)
    sum_kernel = kernels.Sum(ks, kernels.WhiteKernel())
    
    # GPRに標準化とカーネルの選択を追加しgprとする
    # 可視化する際にオートスケーリングしていると見にくいため、StandardScalerを利用
    #model = make_pipeline(StandardScaler(), 
    #                            gp.GaussianProcessRegressor(kernel=sum_kernel, normalize_y=True))
    # エラーが出たらこちらを使う。(alphaを大きくすると安定する)
    model = make_pipeline(StandardScaler(), 
                                gp.GaussianProcessRegressor(kernel=sum_kernel,alpha=10, normalize_y=True))
    
    #GPRの実行, 
    model.fit(x_train,y_train)
    
    #推定値と推定値の標準偏差が入ったデータフレームを作成し
    y_m, y_s = model.predict(x_test, return_std=True)
    y_m = y_m.reshape(-1,)
    y_m_df = pd.Series(y_m, name='Estimated_value')
    y_s_df = pd.Series(y_s, name='std_of_Estimated_value')
    
    # 推定値とその標準偏差をcsvで保存
    estimated_y_m_y_s = pd.concat([y_m_df,y_s_df], axis = 1)
    estimated_y_m_y_s.index = x_test.index
    estimated_y_m_y_s.to_csv('{}/estimated_y_m_y_s.csv'.format(output_folder_name))
    
    #グラフで可視化とそのグラフの保存
    functions.plot_GPR_samples(x_train,y_train,
                               x_test, y_m, y_s, output_folder_name)
    
    # 決定係数等 算出用にモデルを再Fit
    model.fit(autoscaled_x_train,autoscaled_y_train)
    
    # モデルの保存
    with open(output_folder_name + '/model.pkl', mode='wb') as fp:
        pickle.dump(model, fp)
        
# モデルの読み込み
with open(output_folder_name + '/model.pkl', mode='rb') as fp:
    model = pickle.load(fp)

# トレーニングデータ・テストデータの推定、実測値 vs. 推定値のプロット、r2, RMSE, MAE の値の表示、推定値の保存
functions.estimation_and_performance_check_in_regression_train_and_test(model, autoscaled_x_train, y_train,
                                                                               autoscaled_x_test, y_test,
                                                                               output_folder_name)

# SHAPによるモデルの解釈
if evaluate_with_shap:
    functions.shap_for_any_model(output_folder_name, model, autoscaled_x_train, autoscaled_x_test)

# 学習データとテストデータのr2,rmse,maeを格納したdataframeの作成
df_evaluation = functions.evaluation_function(model, autoscaled_x_train, y_train,
                                    autoscaled_x_test, y_test, output_folder_name)

# all_df_evaluationに追加
all_df_evaluation = pd.concat([all_df_evaluation, df_evaluation])

if exe_time:
    
    # 計測終了
    process_time = time.time() - start
    
    #計測時間を辞書に追加
    model_exe_time[output_folder_name.split('/')[1]] = process_time

'''
LWPLS (Locally-Weighted Partial Least Squares)----------------------------------------------------------------------------------------------------------
'''

if exe_time:
    # 計測開始
    start = time.time()

output_folder_name = './LWPLS'
os.makedirs(output_folder_name, exist_ok=True)
model_name_list.append(output_folder_name.split('/')[1])

# CV でLWPLSのハイパーパラメータを決定する
optimal_component_number,optimal_lambda_in_similarity = functions.LWPLS_hyperparameter_CV(autoscaled_x_train, autoscaled_y_train, autoscaled_x_test,
                                  max_number_of_principal_components,lwpls_lambdas,fold_number)

# トレーニングデータでのyの推定
autoscaled_estimated_y_train = functions.lwpls(autoscaled_x_train, autoscaled_y_train,
                                                                     autoscaled_x_train, optimal_component_number,
                                                                     optimal_lambda_in_similarity)
autoscaled_estimated_y_train = autoscaled_estimated_y_train[:, optimal_component_number - 1]

# テストデータでのyの推定
autoscaled_estimated_y_test = functions.lwpls(autoscaled_x_train, autoscaled_y_train,
                                                                     autoscaled_x_test, optimal_component_number,
                                                                     optimal_lambda_in_similarity)
autoscaled_estimated_y_test = autoscaled_estimated_y_test[:, optimal_component_number - 1]

# トレーニングデータ・テストデータの推定、実測値 vs. 推定値のプロット、r2, RMSE, MAE の値の表示、推定値の保存
functions.plot_for_lwpls(autoscaled_estimated_y_train,autoscaled_estimated_y_test,
                         autoscaled_x_train, y_train, autoscaled_x_test, y_test, output_folder_name)

# SHAPによるモデルの解釈
if evaluate_with_shap:
    functions.shap_for_any_model(output_folder_name, model, autoscaled_x_train, autoscaled_x_test)

# 学習データとテストデータのr2,rmse,maeを格納したdataframeの作成
df_evaluation = functions.evaluation_function_for_lwpls(autoscaled_estimated_y_train,autoscaled_estimated_y_test,
                                                        autoscaled_x_train, y_train, autoscaled_x_test, y_test,
                                                        output_folder_name)

# all_df_evaluationに追加
all_df_evaluation = pd.concat([all_df_evaluation, df_evaluation])

if exe_time:
    
    # 計測終了
    process_time = time.time() - start
    
    #計測時間を辞書に追加
    model_exe_time[output_folder_name.split('/')[1]] = process_time

'''
Lasso----------------------------------------------------------------------------------------------------------
'''

if exe_time:
    # 計測開始
    start = time.time()

output_folder_name = './Lasso'
os.makedirs(output_folder_name, exist_ok=True)
model_name_list.append(output_folder_name.split('/')[1])

# output_folder_name内に保存したモデルがなければモデルの構築を行い、あればモデルの構築をとばして推定へ
if os.path.exists(output_folder_name + '/model.pkl') is False:

    # CVによるλの最適化
    r2_in_cv_all = list()
    for lasso_lambda in lasso_lambdas:
        model = Lasso(alpha=lasso_lambda)
        estimated_y_in_cv = pd.DataFrame(cross_val_predict(model, autoscaled_x_train, autoscaled_y_train,
                                                              cv=fold_number))
        estimated_y_in_cv = estimated_y_in_cv * y_train.std() + y_train.mean()
        r2_in_cv = metrics.r2_score(y_train, estimated_y_in_cv)
        r2_in_cv_all.append(r2_in_cv)
    
    optimal_lasso_lambda = functions.plot_and_selection_of_hyperparameter_in_cv(lasso_lambdas, r2_in_cv_all,
                                                                              'lasso_lambdas',
                                                                              'r2 in cross-validation',
                                                                               output_folder_name)
    
    # 最適化された λ
    print('optimal_lasso_lambda : {0}'.format(optimal_lasso_lambda))
    
    # 最適化されたλの保存
    with open('{}/optimal_lasso_lambda.txt'.format(output_folder_name), 'w') as f:
            print('optimal_lasso_lambda : {0}'.format(optimal_lasso_lambda), file = f)
    
    # Lasso
    model = Lasso(alpha=optimal_lasso_lambda) # モデルの宣言
    model.fit(autoscaled_x_train, autoscaled_y_train)
    
    # 標準回帰係数
    standard_regression_coefficients = pd.DataFrame(model.coef_, index=x_train.columns,
                                                        columns=['standard_regression_coefficients'])
    standard_regression_coefficients.to_csv('{}/lasso_standard_regression_coefficients.csv'.format(output_folder_name))  # csv ファイルに保存。
    
    # モデルの保存
    with open(output_folder_name + '/model.pkl', mode='wb') as fp:
        pickle.dump(model, fp)
        
# モデルの読み込み
with open(output_folder_name + '/model.pkl', mode='rb') as fp:
    model = pickle.load(fp)
    
# トレーニングデータ・テストデータの推定、実測値 vs. 推定値のプロット、r2, RMSE, MAE の値の表示、推定値の保存
functions.estimation_and_performance_check_in_regression_train_and_test(model, autoscaled_x_train, y_train,
                                                                                   autoscaled_x_test, y_test,
                                                                                   output_folder_name)

# SHAPによるモデルの解釈
if evaluate_with_shap:
    functions.shap_for_any_model(output_folder_name, model, autoscaled_x_train, autoscaled_x_test)

# 学習データとテストデータのr2,rmse,maeを格納したdataframeの作成
df_evaluation = functions.evaluation_function(model, autoscaled_x_train, y_train,
                                    autoscaled_x_test, y_test, output_folder_name)

# all_df_evaluationに追加
all_df_evaluation = pd.concat([all_df_evaluation, df_evaluation])

if exe_time:
    
    # 計測終了
    process_time = time.time() - start
    
    #計測時間を辞書に追加
    model_exe_time[output_folder_name.split('/')[1]] = process_time

'''
Ridge----------------------------------------------------------------------------------------------------------
'''

if exe_time:
    # 計測開始
    start = time.time()

output_folder_name = './Ridge'
os.makedirs(output_folder_name, exist_ok=True)
model_name_list.append(output_folder_name.split('/')[1])

# output_folder_name内に保存したモデルがなければモデルの構築を行い、あればモデルの構築をとばして推定へ
if os.path.exists(output_folder_name + '/model.pkl') is False:

    # CVによるλの最適化
    r2_in_cv_all = list()
    for ridge_lambda in ridge_lambdas:
        model = Ridge(alpha=ridge_lambda)
        estimated_y_in_cv = pd.DataFrame(cross_val_predict(model, autoscaled_x_train, autoscaled_y_train,
                                                              cv=fold_number))
        estimated_y_in_cv = estimated_y_in_cv * y_train.std() + y_train.mean()
        r2_in_cv = metrics.r2_score(y_train, estimated_y_in_cv)
        r2_in_cv_all.append(r2_in_cv)
    
    optimal_ridge_lambda = functions.plot_and_selection_of_hyperparameter_in_cv(ridge_lambdas, r2_in_cv_all,
                                                                              'ridge_lambdas',
                                                                              'r2 in cross-validation',
                                                                                output_folder_name)
    
    # 最適化された λ
    print('optimal_ridge_lambda : {0}'.format(optimal_ridge_lambda))
    
    # 最適化されたλの保存
    with open('{}/optimal_ridge_lambda.txt'.format(output_folder_name), 'w') as f:
            print('optimal_ridge_lambda : {0}'.format(optimal_ridge_lambda), file = f)

    # Ridge
    model = Ridge(alpha=optimal_ridge_lambda) # モデルの宣言
    model.fit(autoscaled_x_train, autoscaled_y_train)
    
    # 標準回帰係数
    standard_regression_coefficients = pd.DataFrame(model.coef_, index=x_train.columns,
                                                        columns=['standard_regression_coefficients'])
    standard_regression_coefficients.to_csv('{}/ridge_standard_regression_coefficients.csv'.format(output_folder_name))  # csv ファイルに保存。
    
    # モデルの保存
    with open(output_folder_name + '/model.pkl', mode='wb') as fp:
        pickle.dump(model, fp)
        
# モデルの読み込み
with open(output_folder_name + '/model.pkl', mode='rb') as fp:
    model = pickle.load(fp)

# トレーニングデータ・テストデータの推定、実測値 vs. 推定値のプロット、r2, RMSE, MAE の値の表示、推定値の保存
functions.estimation_and_performance_check_in_regression_train_and_test(model, autoscaled_x_train, y_train,
                                                                               autoscaled_x_test, y_test,
                                                                               output_folder_name)
        
# SHAPによるモデルの解釈
if evaluate_with_shap:
    functions.shap_for_any_model(output_folder_name, model, autoscaled_x_train, autoscaled_x_test)

# 学習データとテストデータのr2,rmse,maeを格納したdataframeの作成
df_evaluation = functions.evaluation_function(model, autoscaled_x_train, y_train,
                                    autoscaled_x_test, y_test, output_folder_name)

# all_df_evaluationに追加
all_df_evaluation = pd.concat([all_df_evaluation, df_evaluation])

if exe_time:
    
    # 計測終了
    process_time = time.time() - start
    
    #計測時間を辞書に追加
    model_exe_time[output_folder_name.split('/')[1]] = process_time

'''
ElasticNet----------------------------------------------------------------------------------------------------------
'''

if exe_time:
    # 計測開始
    start = time.time()

output_folder_name = './ElasticNet'
os.makedirs(output_folder_name, exist_ok=True)
model_name_list.append(output_folder_name.split('/')[1])

# output_folder_name内に保存したモデルがなければモデルの構築を行い、あればモデルの構築をとばして推定へ
if os.path.exists(output_folder_name + '/model.pkl') is False:

    # CVによるα,λの最適化
    elastic_net_in_cv = ElasticNetCV(cv=fold_number, l1_ratio=elastic_net_lambdas, alphas=elastic_net_alphas)
    elastic_net_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
    optimal_elastic_net_alpha = elastic_net_in_cv.alpha_
    optimal_elastic_net_lambda = elastic_net_in_cv.l1_ratio_
    
    # 最適化されたα,λ
    print('optimal_elastic_net_alpha : {0}\noptimal_elastic_net_lambda : {1}'.format(optimal_elastic_net_alpha, optimal_elastic_net_lambda))
    
    # 最適化されたα,λの保存
    with open('{}/optimal_en_alpha_and_lambda.txt'.format(output_folder_name), 'w') as f:
        print('optimal_elastic_net_alpha : {0}\noptimal_elastic_net_lambda : {1}'.format(optimal_elastic_net_alpha, optimal_elastic_net_lambda), file = f)
    
    # ElasticNet
    model = ElasticNet(l1_ratio=optimal_elastic_net_lambda, alpha=optimal_elastic_net_alpha) # モデルの宣言
    model.fit(autoscaled_x_train, autoscaled_y_train)
    
    # 標準回帰係数
    standard_regression_coefficients = pd.DataFrame(model.coef_, index=x_train.columns,
                                                        columns=['standard_regression_coefficients'])
    standard_regression_coefficients.to_csv('{}/en_standard_regression_coefficients.csv'.format(output_folder_name))  # csv ファイルに保存。
    
     # モデルの保存
    with open(output_folder_name + '/model.pkl', mode='wb') as fp:
        pickle.dump(model, fp)
        
# モデルの読み込み
with open(output_folder_name + '/model.pkl', mode='rb') as fp:
    model = pickle.load(fp)
    
# トレーニングデータ・テストデータの推定、実測値 vs. 推定値のプロット、r2, RMSE, MAE の値の表示、推定値の保存
functions.estimation_and_performance_check_in_regression_train_and_test(model, autoscaled_x_train, y_train,
                                                                               autoscaled_x_test, y_test,
                                                                               output_folder_name)
    
# SHAPによるモデルの解釈
if evaluate_with_shap:
    functions.shap_for_any_model(output_folder_name, model, autoscaled_x_train, autoscaled_x_test)

# 学習データとテストデータのr2,rmse,maeを格納したdataframeの作成
df_evaluation = functions.evaluation_function(model, autoscaled_x_train, y_train,
                                    autoscaled_x_test, y_test, output_folder_name)

# all_df_evaluationに追加
all_df_evaluation = pd.concat([all_df_evaluation, df_evaluation])

if exe_time:
    
    # 計測終了
    process_time = time.time() - start
    
    #計測時間を辞書に追加
    model_exe_time[output_folder_name.split('/')[1]] = process_time

'''
LightGBM_optuna----------------------------------------------------------------------------------------------------------
'''

if exe_time:
    # 計測開始
    start = time.time()

output_folder_name = './LightGBM_optuna'
os.makedirs(output_folder_name, exist_ok=True)
model_name_list.append(output_folder_name.split('/')[1])

# output_folder_name内に保存したモデルがなければモデルの構築を行い、あればモデルの構築をとばして推定へ
if os.path.exists(output_folder_name + '/model.pkl') is False:
    
    # サブモデル数を、決め打ち or early_stoppingで決定するかで分岐処理
    if decision_of_submodel:
        best_n_estimators_for_optuna = lgb_number_of_sub_models #サブモデル数決め打ち
    else:
        # optunaで用いるサブモデルの数 (決定木の数)の最適化
        model = lgb.LGBMRegressor(n_estimators=1000)
        model.fit(autoscaled_x_train_tmp, autoscaled_y_train_tmp, eval_set=(autoscaled_x_validation, autoscaled_y_validation),
                  eval_metric='l2', early_stopping_rounds=100)
        best_n_estimators_for_optuna = model.best_iteration_
        
    # サブモデルの数が0になってしまった場合、サブモデルの数を決め打ちにする(early_stoppingで決めるとたまに0になる場合がある)
    if best_n_estimators_for_optuna == 0:
        best_n_estimators_for_optuna = lgb_number_of_sub_models
    
    # optunaで用いる最適化されたサブモデルの数
    print(best_n_estimators_for_optuna)
    
    # optunaで用いる最適化されたサブモデルの数の保存
    with open('{}/best_n_estimators_in_cv.txt'.format(output_folder_name), 'w') as f:
            print('best_n_estimators_in_cv : {0}'.format(best_n_estimators_for_optuna), file = f)
        
    # optunaを用いてハイパーパラメータの最適化(探索範囲はfunction.pyのlgb_optuna参照)
    study = optuna.create_study()
    study.optimize(functions.lgb_optuna(best_n_estimators_for_optuna, fold_number, autoscaled_x_train, autoscaled_y_train, y_train), n_trials=100)
    
    # 最適化されたハイパーパラメータ
    print(study.best_params)
    
    # 最適化されたハイパーパラメータの保存
    with open('{}/params.txt'.format(output_folder_name), 'w') as f:
            print(study.best_params, file = f)
            
    # サブモデル数を、決め打ち or early_stoppingで決定するかで分岐処理
    if decision_of_submodel:
        best_n_estimators = lgb_number_of_sub_models
    else:
        # サブモデルの数 (決定木の数)の最適化
        model = lgb.LGBMRegressor(**study.best_params, n_estimators=1000)
        model.fit(autoscaled_x_train_tmp, autoscaled_y_train_tmp, eval_set=(autoscaled_x_validation, autoscaled_y_validation),
                  eval_metric='l2', early_stopping_rounds=100)
        best_n_estimators = model.best_iteration_
        
    # サブモデルの数が0になってしまった場合、サブモデルの数を決め打ちにする(early_stoppingで決めるとたまに0になる場合がある)
    if best_n_estimators == 0:
        best_n_estimators = lgb_number_of_sub_models
    
    # 最適化されたサブモデルの数
    print(best_n_estimators)
    
    # 最適化されたサブモデルの数の保存
    with open('{}/best_n_estimators.txt'.format(output_folder_name), 'w') as f:
            print('best_n_estimators : {0}'.format(best_n_estimators), file = f)
                
    # LightGBM
    model = lgb.LGBMRegressor(**study.best_params, n_estimators=best_n_estimators) # モデルの宣言
    model.fit(autoscaled_x_train, autoscaled_y_train)
    
    # モデルの保存
    with open(output_folder_name + '/model.pkl', mode='wb') as fp:
        pickle.dump(model, fp)
        
# モデルの読み込み
with open(output_folder_name + '/model.pkl', mode='rb') as fp:
    model = pickle.load(fp)

# トレーニングデータ・テストデータの推定、実測値 vs. 推定値のプロット、r2, RMSE, MAE の値の表示、推定値の保存
functions.estimation_and_performance_check_in_regression_train_and_test(model, autoscaled_x_train, y_train,
                                                                               autoscaled_x_test, y_test,
                                                                               output_folder_name)

# SHAPによるモデルの解釈
if evaluate_with_shap:
    functions.shap_for_tree_model(output_folder_name, model, autoscaled_x_test)

# 学習データとテストデータのr2,rmse,maeを格納したdataframeの作成
df_evaluation = functions.evaluation_function(model, autoscaled_x_train, y_train,
                                    autoscaled_x_test, y_test, output_folder_name)

# all_df_evaluationに追加
all_df_evaluation = pd.concat([all_df_evaluation, df_evaluation])

if exe_time:
    
    # 計測終了
    process_time = time.time() - start
    
    #計測時間を辞書に追加
    model_exe_time[output_folder_name.split('/')[1]] = process_time

'''
LightGBM_default----------------------------------------------------------------------------------------------------------
'''

if exe_time:
    # 計測開始
    start = time.time()

output_folder_name = './LightGBM_default'
os.makedirs(output_folder_name, exist_ok=True)
model_name_list.append(output_folder_name.split('/')[1])

# output_folder_name内に保存したモデルがなければモデルの構築を行い、あればモデルの構築をとばして推定へ
if os.path.exists(output_folder_name + '/model.pkl') is False:

    # LightGBM
    model = lgb.LGBMRegressor()
    model.fit(autoscaled_x_train, autoscaled_y_train)
    
    # モデルの保存
    with open(output_folder_name + '/model.pkl', mode='wb') as fp:
        pickle.dump(model, fp)
        
# モデルの読み込み
with open(output_folder_name + '/model.pkl', mode='rb') as fp:
    model = pickle.load(fp)

# トレーニングデータ・テストデータの推定、実測値 vs. 推定値のプロット、r2, RMSE, MAE の値の表示、推定値の保存
functions.estimation_and_performance_check_in_regression_train_and_test(model, autoscaled_x_train, y_train,
                                                                               autoscaled_x_test, y_test,
                                                                               output_folder_name)

# SHAPによるモデルの解釈
if evaluate_with_shap:
    functions.shap_for_tree_model(output_folder_name, model, autoscaled_x_test)

# 学習データとテストデータのr2,rmse,maeを格納したdataframeの作成
df_evaluation = functions.evaluation_function(model, autoscaled_x_train, y_train,
                                    autoscaled_x_test, y_test, output_folder_name)

# all_df_evaluationに追加
all_df_evaluation = pd.concat([all_df_evaluation, df_evaluation])

if exe_time:
    
    # 計測終了
    process_time = time.time() - start
    
    #計測時間を辞書に追加
    model_exe_time[output_folder_name.split('/')[1]] = process_time

'''
XGBoost_optuna----------------------------------------------------------------------------------------------------------
'''

if exe_time:
    # 計測開始
    start = time.time()

output_folder_name = './XGBoost_optuna'
os.makedirs(output_folder_name, exist_ok=True)
model_name_list.append(output_folder_name.split('/')[1])

# output_folder_name内に保存したモデルがなければモデルの構築を行い、あればモデルの構築をとばして推定へ
if os.path.exists(output_folder_name + '/model.pkl') is False:
    
    # サブモデル数を、決め打ち or early_stoppingで決定するかで分岐処理
    if decision_of_submodel:
        best_n_estimators_for_optuna = xgb_number_of_sub_models # サブモデル数を決め打ち
    else:
        # optunaで用いるサブモデルの数 (決定木の数)の最適化
        model = xgb.XGBRegressor(n_estimators=1000)
        model.fit(autoscaled_x_train_tmp, autoscaled_y_train_tmp,
                  eval_set=[(autoscaled_x_validation, autoscaled_y_validation)],eval_metric='rmse', early_stopping_rounds=100)
        best_n_estimators_for_optuna = model.best_iteration
        
    # サブモデルの数が0になってしまった場合、サブモデルの数を決め打ちにする(early_stoppingで決めるとたまに0になる場合がある)
    if best_n_estimators_for_optuna == 0:
        best_n_estimators_for_optuna = xgb_number_of_sub_models
    
    # optunaで用いる最適化されたサブモデルの数
    print(best_n_estimators_for_optuna)
    
    # optunaで用いる最適化されたサブモデルの数の保存
    with open('{}/best_n_estimators_in_cv.txt'.format(output_folder_name), 'w') as f:
            print('best_n_estimators_in_cv : {0}'.format(best_n_estimators_for_optuna), file = f)
              
    # optunaを用いてハイパーパラメータの最適化(探索範囲はfunction.pyのxgb_optuna参照)
    study = optuna.create_study()
    study.optimize(functions.xgb_optuna(best_n_estimators_for_optuna, fold_number, autoscaled_x_train, autoscaled_y_train, y_train), n_trials=100)
    
    # 最適化されたハイパーパラメータ
    print(study.best_params)
    
    # 最適化されたハイパーパラメータの保存
    with open('{}/params.txt'.format(output_folder_name), 'w') as f:
            print(study.best_params, file = f)
    
    # サブモデル数を、決め打ち or early_stoppingで決定するかで分岐処理
    if decision_of_submodel:
        best_n_estimators = xgb_number_of_sub_models
    else:
        # サブモデルの数 (決定木の数)の最適化
        model = xgb.XGBRegressor(**study.best_params, n_estimators=1000)
        model.fit(autoscaled_x_train_tmp, autoscaled_y_train_tmp,
                  eval_set=[(autoscaled_x_validation, autoscaled_y_validation)],
                  eval_metric='rmse', early_stopping_rounds=100)
        best_n_estimators = model.get_booster().best_iteration
        
    # サブモデルの数が0になってしまった場合、サブモデルの数を決め打ちにする(early_stoppingで決めるとたまに0になる場合がある)
    if best_n_estimators == 0:
        best_n_estimators = xgb_number_of_sub_models
    
    # 最適化されたサブモデルの数
    print(best_n_estimators)
    
    # 最適化されたサブモデルの数の保存
    with open('{}/best_n_estimators.txt'.format(output_folder_name), 'w') as f:
            print('best_n_estimators : {0}'.format(best_n_estimators), file = f) 
                        
    # XGBoost
    model = xgb.XGBRegressor(**study.best_params) # モデルの宣言
    model.fit(autoscaled_x_train, autoscaled_y_train)
    
    # モデルの保存
    with open(output_folder_name + '/model.pkl', mode='wb') as fp:
        pickle.dump(model, fp)
        
# モデルの読み込み
with open(output_folder_name + '/model.pkl', mode='rb') as fp:
    model = pickle.load(fp)

# トレーニングデータ・テストデータの推定、実測値 vs. 推定値のプロット、r2, RMSE, MAE の値の表示、推定値の保存
functions.estimation_and_performance_check_in_regression_train_and_test(model, autoscaled_x_train, y_train,
                                                                               autoscaled_x_test, y_test,
                                                                               output_folder_name)

# SHAPによるモデルの解釈
if evaluate_with_shap:
    functions.shap_for_tree_model(output_folder_name, model, autoscaled_x_test)

# 学習データとテストデータのr2,rmse,maeを格納したdataframeの作成
df_evaluation = functions.evaluation_function(model, autoscaled_x_train, y_train,
                                    autoscaled_x_test, y_test, output_folder_name)

# all_df_evaluationに追加
all_df_evaluation = pd.concat([all_df_evaluation, df_evaluation])

if exe_time:
    
    # 計測終了
    process_time = time.time() - start
    
    #計測時間を辞書に追加
    model_exe_time[output_folder_name.split('/')[1]] = process_time

'''
XGBoost_default----------------------------------------------------------------------------------------------------------
'''

if exe_time:
    # 計測開始
    start = time.time()

output_folder_name = './XGBoost_default'
os.makedirs(output_folder_name, exist_ok=True)
model_name_list.append(output_folder_name.split('/')[1])

# output_folder_name内に保存したモデルがなければモデルの構築を行い、あればモデルの構築をとばして推定へ
if os.path.exists(output_folder_name + '/model.pkl') is False:
    
    # XGBoost
    model = xgb.XGBRegressor()
    model.fit(autoscaled_x_train, autoscaled_y_train)
    
    # モデルの保存
    with open(output_folder_name + '/model.pkl', mode='wb') as fp:
        pickle.dump(model, fp)
        
# モデルの読み込み
with open(output_folder_name + '/model.pkl', mode='rb') as fp:
    model = pickle.load(fp)

# トレーニングデータ・テストデータの推定、実測値 vs. 推定値のプロット、r2, RMSE, MAE の値の表示、推定値の保存
functions.estimation_and_performance_check_in_regression_train_and_test(model, autoscaled_x_train, y_train,
                                                                               autoscaled_x_test, y_test,
                                                                               output_folder_name)

# SHAPによるモデルの解釈
if evaluate_with_shap:
    functions.shap_for_tree_model(output_folder_name, model, autoscaled_x_test)

# 学習データとテストデータのr2,rmse,maeを格納したdataframeの作成
df_evaluation = functions.evaluation_function(model, autoscaled_x_train, y_train,
                                    autoscaled_x_test, y_test, output_folder_name)

# all_df_evaluationに追加
all_df_evaluation = pd.concat([all_df_evaluation, df_evaluation])

if exe_time:
    
    # 計測終了
    process_time = time.time() - start
    
    #計測時間を辞書に追加
    model_exe_time[output_folder_name.split('/')[1]] = process_time

'''
GBDT_optuna----------------------------------------------------------------------------------------------------------
'''

if exe_time:
    # 計測開始
    start = time.time()

output_folder_name = './GBDT_optuna'
os.makedirs(output_folder_name, exist_ok=True)
model_name_list.append(output_folder_name.split('/')[1])

# output_folder_name内に保存したモデルがなければモデルの構築を行い、あればモデルの構築をとばして推定へ
if os.path.exists(output_folder_name + '/model.pkl') is False:
    
    # サブモデル数を、決め打ち or early_stoppingで決定するかで分岐処理
    if decision_of_submodel:
        best_n_estimators_for_optuna = gbdt_number_of_sub_models
    else:
        # optunaで用いるサブモデルの数 (決定木の数)の最適化
        model = GradientBoostingRegressor(n_estimators=1000, validation_fraction=fraction_of_validation_samples,
                                          n_iter_no_change=100)
        model.fit(autoscaled_x_train, autoscaled_y_train)
        best_n_estimators_for_optuna = len(model.estimators_)
        
    # サブモデルの数が0になってしまった場合、サブモデルの数を決め打ちにする(early_stoppingで決めるとたまに0になる場合がある)
    if best_n_estimators_for_optuna == 0:
        best_n_estimators_for_optuna = gbdt_number_of_sub_models
    
    # optunaで用いる最適化されたサブモデルの数
    print(best_n_estimators_for_optuna)
    
    # optunaで用いる最適化されたサブモデルの数の保存
    with open('{}/best_n_estimators_in_cv.txt'.format(output_folder_name), 'w') as f:
            print('best_n_estimators_in_cv : {0}'.format(best_n_estimators_for_optuna), file = f)
        
    # optunaを用いてハイパーパラメータの最適化(探索範囲はfunction.pyのgbdt_optuna参照)
    study = optuna.create_study()
    study.optimize(functions.gbdt_optuna(best_n_estimators_for_optuna, fold_number, autoscaled_x_train, autoscaled_y_train, y_train), n_trials=100)
    
    # 最適化されたハイパーパラメータ
    print(study.best_params)
    
    # 最適化されたハイパーパラメータの保存
    with open('{}/params.txt'.format(output_folder_name), 'w') as f:
            print(study.best_params, file = f)
    
    # サブモデル数を、決め打ち or early_stoppingで決定するかで分岐処理
    if decision_of_submodel:
        best_n_estimators = gbdt_number_of_sub_models
    else:
        # サブモデルの数 (決定木の数)の最適化
        model = GradientBoostingRegressor(**study.best_params, n_estimators=1000,
                                          validation_fraction=fraction_of_validation_samples, n_iter_no_change=100)
        model.fit(autoscaled_x_train, autoscaled_y_train)
        best_n_estimators = len(model.estimators_)
        
    # サブモデルの数が0になってしまった場合、サブモデルの数を決め打ちにする(early_stoppingで決めるとたまに0になる場合がある)
    if best_n_estimators == 0:
        best_n_estimators = gbdt_number_of_sub_models
    
    # 最適化されたサブモデルの数
    print(best_n_estimators)
    
    # 最適化されたサブモデルの数の保存
    with open('{}/best_n_estimators.txt'.format(output_folder_name), 'w') as f:
            print('best_n_estimators : {0}'.format(best_n_estimators), file = f) 
                
    # GBDT
    model = GradientBoostingRegressor(**study.best_params, n_estimators=best_n_estimators) # モデルの宣言
    model.fit(autoscaled_x_train, autoscaled_y_train)
    
    # モデルの保存
    with open(output_folder_name + '/model.pkl', mode='wb') as fp:
        pickle.dump(model, fp)
        
# モデルの読み込み
with open(output_folder_name + '/model.pkl', mode='rb') as fp:
    model = pickle.load(fp)

# トレーニングデータ・テストデータの推定、実測値 vs. 推定値のプロット、r2, RMSE, MAE の値の表示、推定値の保存
functions.estimation_and_performance_check_in_regression_train_and_test(model, autoscaled_x_train, y_train,
                                                                               autoscaled_x_test, y_test,
                                                                               output_folder_name)

# SHAPによるモデルの解釈
if evaluate_with_shap:
    functions.shap_for_any_model(output_folder_name, model,autoscaled_x_train, autoscaled_x_test)

# 学習データとテストデータのr2,rmse,maeを格納したdataframeの作成
df_evaluation = functions.evaluation_function(model, autoscaled_x_train, y_train,
                                    autoscaled_x_test, y_test, output_folder_name)

# all_df_evaluationに追加
all_df_evaluation = pd.concat([all_df_evaluation, df_evaluation])

if exe_time:
    
    # 計測終了
    process_time = time.time() - start
    
    #計測時間を辞書に追加
    model_exe_time[output_folder_name.split('/')[1]] = process_time

'''
GBDT_default----------------------------------------------------------------------------------------------------------
'''

if exe_time:
    # 計測開始
    start = time.time()

output_folder_name = './GBDT_default'
os.makedirs(output_folder_name, exist_ok=True)
model_name_list.append(output_folder_name.split('/')[1])

# output_folder_name内に保存したモデルがなければモデルの構築を行い、あればモデルの構築をとばして推定へ
if os.path.exists(output_folder_name + '/model.pkl') is False:

    # GBDT
    model = GradientBoostingRegressor()
    model.fit(autoscaled_x_train, autoscaled_y_train)
    
    # モデルの保存
    with open(output_folder_name + '/model.pkl', mode='wb') as fp:
        pickle.dump(model, fp)
        
# モデルの読み込み
with open(output_folder_name + '/model.pkl', mode='rb') as fp:
    model = pickle.load(fp)

# トレーニングデータ・テストデータの推定、実測値 vs. 推定値のプロット、r2, RMSE, MAE の値の表示、推定値の保存
functions.estimation_and_performance_check_in_regression_train_and_test(model, autoscaled_x_train, y_train,
                                                                               autoscaled_x_test, y_test,
                                                                               output_folder_name)

# SHAPによるモデルの解釈
if evaluate_with_shap:
    functions.shap_for_any_model(output_folder_name, model,autoscaled_x_train, autoscaled_x_test)

# 学習データとテストデータのr2,rmse,maeを格納したdataframeの作成
df_evaluation = functions.evaluation_function(model, autoscaled_x_train, y_train,
                                    autoscaled_x_test, y_test, output_folder_name)

# all_df_evaluationに追加
all_df_evaluation = pd.concat([all_df_evaluation, df_evaluation])

if exe_time:
    
    # 計測終了
    process_time = time.time() - start
    
    #計測時間を辞書に追加
    model_exe_time[output_folder_name.split('/')[1]] = process_time

'''
CatBoost_optuna----------------------------------------------------------------------------------------------------------
'''

if exe_time:
    # 計測開始
    start = time.time()

output_folder_name = './CatBoost_optuna'
os.makedirs(output_folder_name, exist_ok=True)
model_name_list.append(output_folder_name.split('/')[1])

# output_folder_name内に保存したモデルがなければモデルの構築を行い、あればモデルの構築をとばして推定へ
if os.path.exists(output_folder_name + '/model.pkl') is False:
    
    # optunaを用いてハイパーパラメータの最適化(探索範囲はfunction.pyのcb_optuna参照)
    study = optuna.create_study()
    study.optimize(functions.cb_optuna(fold_number, autoscaled_x_train, autoscaled_y_train, y_train), n_trials=100)
    
    # 最適化されたハイパーパラメータ
    print(study.best_params)
    
    # 最適化されたハイパーパラメータの保存
    with open('{}/params.txt'.format(output_folder_name), 'w') as f:
            print(study.best_params, file = f)
    
    # CatBoost
    model = cb.CatBoostRegressor(**study.best_params, allow_writing_files=False) # モデルの宣言
    model.fit(autoscaled_x_train, autoscaled_y_train)
    
    # モデルの保存
    with open(output_folder_name + '/model.pkl', mode='wb') as fp:
        pickle.dump(model, fp)
        
# モデルの読み込み
with open(output_folder_name + '/model.pkl', mode='rb') as fp:
    model = pickle.load(fp)

# トレーニングデータ・テストデータの推定、実測値 vs. 推定値のプロット、r2, RMSE, MAE の値の表示、推定値の保存
functions.estimation_and_performance_check_in_regression_train_and_test(model, autoscaled_x_train, y_train,
                                                                               autoscaled_x_test, y_test,
                                                                               output_folder_name)

# SHAPによるモデルの解釈
if evaluate_with_shap:
    functions.shap_for_tree_model(output_folder_name, model, autoscaled_x_test)

# 学習データとテストデータのr2,rmse,maeを格納したdataframeの作成
df_evaluation = functions.evaluation_function(model, autoscaled_x_train, y_train,
                                    autoscaled_x_test, y_test, output_folder_name)

# all_df_evaluationに追加
all_df_evaluation = pd.concat([all_df_evaluation, df_evaluation])

if exe_time:
    
    # 計測終了
    process_time = time.time() - start
    
    #計測時間を辞書に追加
    model_exe_time[output_folder_name.split('/')[1]] = process_time

'''
CatBoost_default----------------------------------------------------------------------------------------------------------
'''

if exe_time:
    # 計測開始
    start = time.time()

output_folder_name = './CatBoost_default'
os.makedirs(output_folder_name, exist_ok=True)
model_name_list.append(output_folder_name.split('/')[1])

# output_folder_name内に保存したモデルがなければモデルの構築を行い、あればモデルの構築をとばして推定へ
if os.path.exists(output_folder_name + '/model.pkl') is False:

    # CatBoost
    model = cb.CatBoostRegressor(allow_writing_files=False) # モデルの宣言
    model.fit(autoscaled_x_train, autoscaled_y_train)
    
    # モデルの保存
    with open(output_folder_name + '/model.pkl', mode='wb') as fp:
        pickle.dump(model, fp)
        
# モデルの読み込み
with open(output_folder_name + '/model.pkl', mode='rb') as fp:
    model = pickle.load(fp)

# トレーニングデータ・テストデータの推定、実測値 vs. 推定値のプロット、r2, RMSE, MAE の値の表示、推定値の保存
functions.estimation_and_performance_check_in_regression_train_and_test(model, autoscaled_x_train, y_train,
                                                                               autoscaled_x_test, y_test,
                                                                               output_folder_name)

# SHAPによるモデルの解釈
if evaluate_with_shap:
    functions.shap_for_tree_model(output_folder_name, model, autoscaled_x_test)

# 学習データとテストデータのr2,rmse,maeを格納したdataframeの作成
df_evaluation = functions.evaluation_function(model, autoscaled_x_train, y_train,
                                    autoscaled_x_test, y_test, output_folder_name)

# all_df_evaluationに追加
all_df_evaluation = pd.concat([all_df_evaluation, df_evaluation])

if exe_time:
    
    # 計測終了
    process_time = time.time() - start
    
    #計測時間を辞書に追加
    model_exe_time[output_folder_name.split('/')[1]] = process_time

'''
DNN----------------------------------------------------------------------------------------------------------
'''

if exe_time:
    # 計測開始
    start = time.time()

output_folder_name = './DNN'
os.makedirs(output_folder_name, exist_ok=True)
model_name_list.append(output_folder_name.split('/')[1])

if os.path.exists(output_folder_name + '/model.pkl') is False:

    # モデル作成
    model = Sequential() #インスタンス作成
    
    # 全結合層1
    model.add(Dense(num_of_unit_1, input_dim=len(autoscaled_x_train_tmp.columns),
                    kernel_initializer=he_normal(123)))
    model.add(Activation("relu"))
    
    # 全結合層2
    model.add(Dense(num_of_unit_2, kernel_initializer=he_normal(123)))
    model.add(Activation("relu"))
    
    # 出力層
    model.add(Dense(1,kernel_initializer=he_normal(123)))
    model.add(Activation("linear"))
    
    # モデルのコンパイル
    model.compile(optimizer='Adam', loss="mse", metrics=['mae', 'mse'])
    
    # モデルサマリー保存
    model.summary()
    with open(output_folder_name + '/model_summary.txt', "w") as fp:
        model.summary(print_fn=lambda x: fp.write(x + "\r\n"))
    
    # EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss',  patience=10, verbose=1)
    
    # reduce learning rate
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    
    # モデルの学習
    hist = model.fit(autoscaled_x_train_tmp, autoscaled_y_train_tmp,
                     verbose=1, epochs=epochs, batch_size=batch_size, 
                     validation_data=(autoscaled_x_validation, autoscaled_y_validation),
                     callbacks=[early_stopping, reduce_lr])
    
    # 学習曲線をプロットして保存
    functions.acc_and_loss_plot(hist, output_folder_name)
    
    # モデルの保存
    with open(output_folder_name + '/model.pkl', mode='wb') as fp:
        pickle.dump(model, fp)
        
# モデルの読み込み
with open(output_folder_name + '/model.pkl', mode='rb') as fp:
    model = pickle.load(fp)

# トレーニングデータ・テストデータの推定、実測値 vs. 推定値のプロット、r2, RMSE, MAE の値の表示、推定値の保存
functions.estimation_and_performance_check_in_regression_train_and_test(model, autoscaled_x_train, y_train,
                                                                               autoscaled_x_test, y_test,
                                                                               output_folder_name)

# SHAPによるモデルの解釈
if evaluate_with_shap:
    functions.shap_for_dnn_model(output_folder_name, model, autoscaled_x_train, autoscaled_x_test)

# 学習データとテストデータのr2,rmse,maeを格納したdataframeの作成
df_evaluation = functions.evaluation_function(model, autoscaled_x_train, y_train,
                                    autoscaled_x_test, y_test, output_folder_name)

# all_df_evaluationに追加
all_df_evaluation = pd.concat([all_df_evaluation, df_evaluation])

# all_df_evaluationをcsv保存
all_df_evaluation.to_csv('all_model_evaluation.csv')

if exe_time:
    
    # 計測終了
    process_time = time.time() - start
    
    #計測時間を辞書に追加
    model_exe_time[output_folder_name.split('/')[1]] = process_time

# 計測時間を保持した辞書をデータフレームに変換して、csv保存
df_exe_time = pd.DataFrame(model_exe_time.values(), index=model_exe_time.keys())
df_exe_time = df_exe_time.rename(columns={0:'exe_time_sec'})
df_exe_time.to_csv('exe_time.csv')

'''
各モデルのy-yプロットのPDF化------------------------------------------------------------------------------------
'''
        
img_filename_list = ['training_y_y_plot.jpg', 'training_y_y_plotwith_name.jpg',
                      'test_y_y_plot.jpg', 'test_y_y_plot_with_sample_name.jpg']

# 画像一時保存フォルダ作成
os.makedirs('./tmp', exist_ok=True)
concat_img_list = []

# 各モデルの4枚のプロットの画像を読み込んで、正方形にそろえて、連結後、一時保存フォルダに保存
size=500 # 1枚の画像の縦（横）の長さ
for i, model_name in enumerate(model_name_list):
    img_list = []
    
    # 画像の読み込んで正方形にそろえる
    for img_filename in img_filename_list:
        img = cv2.imread('./' + model_name + '/' + img_filename, cv2.IMREAD_COLOR)
        orgHeight, orgWidth = img.shape[:2] # 高さと幅を読み込む
        new_img = np.full((size, size, 3),255,np.uint8) # 画像を保存するメモリを確保する
        if orgHeight > orgWidth: #縦長の画像の場合
            newHeight = size
            newWidth = int(orgWidth / orgHeight * size)
            compressed_img = cv2.resize(img, (newWidth, newHeight))
            start =(size - newWidth) // 2 
            new_img[:, start : start + newWidth] = compressed_img
        else: #横長の画像の場合
            newWidth = size
            newHeight = int(orgHeight / orgWidth * size)
            compressed_img = cv2.resize(img, (newWidth, newHeight))
            start =(size - newHeight) // 2 
            new_img[start : start + newHeight, :] = compressed_img
        new_img = new_img[np.newaxis]
        img_list.append(new_img)
    
    # 画像を2枚ずつ横に連結
    img_1 = cv2.hconcat([img_list[0][0], img_list[1][0]])
    img_2 = cv2.hconcat([img_list[2][0], img_list[3][0]])
    
    # pdf1ページにmodelの名前が入るようにテキストを書き込んだ画像を作成
    text_img_1 = np.full((40, 1000, 3),255,np.uint8)
    cv2.putText(text_img_1, model_name, (0, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2, cv2.LINE_AA)
    text_img_2 = np.full((40, 1000, 3),255,np.uint8)
    cv2.putText(text_img_2, 'training_y_y_plot', (0, 35), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1, cv2.LINE_AA)
    text_img_3 = np.full((30, 1000, 3),255,np.uint8)
    cv2.putText(text_img_3, 'test_y_y_plot', (0, 21), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1, cv2.LINE_AA)
    
    # 各画像を縦に連結
    concat_img = cv2.vconcat([text_img_1, text_img_2, img_1, text_img_3, img_2])
    cv2.imwrite('./tmp/' + str(i) + '.png',concat_img) # 連結した画像の保存

pdf_file_name = './all_model_plot.pdf' # 出力するPDFの名前

conca_png_folder = './tmp/' # 連結した画像フォルダ
concat_img_list = os.listdir(conca_png_folder) #ファイル名の取得
concat_img_list = natsorted(concat_img_list) #自然順でソート

# 連結した画像をpdfで保存
with open(pdf_file_name,"wb") as f:
    # 画像フォルダの中にあるPNGファイルを取得し配列に追加、バイナリ形式でファイルに書き込む
    f.write(img2pdf.convert([Image.open(conca_png_folder+j).filename for j in concat_img_list]))
shutil.rmtree('./tmp/') # 一時フォルダの削除