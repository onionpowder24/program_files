# -*- coding: utf-8 -*-

import glob
import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import os
import time
from dcekit.validation import double_cross_validation, DCEGridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn import metrics
from functions import plot_y_vs_estimated_y
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import kernels
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn.gaussian_process as gp
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import lightgbm as lgb
import xgboost as xgb
import catboost as cb


# 処理時間の計測を行うか
exe_time = True

# 作業ディレクトリ内のcsvファイルを読み込む
files = glob.glob('*.csv')  # ワイルドカードが使用可能、今回はcsvファイルを取得
for file in files:
    dataset = pd.read_csv(file, index_col=0)

number_of_test_samples = math.ceil(0.2 * dataset.shape[0])  # テストデータ(全体の2割)

# バリデーションデータのサンプル数の割合 (LightGBM, XGBoost, GBDT, DNNの検証に使用)
fraction_of_validation_samples = 0.2

# PLSのハイパーパラメータ
max_number_of_principal_components = math.ceil(dataset.shape[1] * 0.7)

# SVRのハイパーパラメータ候補
svr_cs = 2 ** np.arange(-5, 11, dtype=float)
svr_epsilons = 2 ** np.arange(-10, 1, dtype=float)
svr_gammas = 2 ** np.arange(-20, 11, dtype=float)

# DTにおける木の深さの最大値の候補
max_depths = np.arange(1, 31)

# RFのハイパーパラメータ
ratios_of_x = np.arange(0.1, 1.1, 0.1)
n_estimators = 500

# Lassoのハイパーパラメータ候補
lasso_lambdas = np.arange(0.01, 0.71, 0.01, dtype=float)

# Ridgeのハイパーパラメータ候補
ridge_lambdas = np.arange(0.01, 0.71, 0.01, dtype=float)

# ElasticNetのハイパーパラメータ候補
elastic_net_lambdas = np.arange(0.01, 0.71, 0.01, dtype=float)
elastic_net_alphas = np.arange(0.01, 1.00, 0.01, dtype=float)

y = dataset.iloc[:, 0]
x = dataset.iloc[:, 1:]

# 標準偏差が0の説明変数を削除(トレーニングとテスト)
std_0_variable_flags = x.std() == 0
x = x.drop(x.columns[std_0_variable_flags], axis=1)
autoscaled_x = (x - x.mean()) / x.std()

# ランダムにトレーニングデータとテストデータに分割
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=number_of_test_samples, shuffle=True, random_state=1
)

std_0_variable_flags = x_train.std() == 0
x_train = x_train.drop(x_train.columns[std_0_variable_flags], axis=1)
x_test = x_test.drop(x_test.columns[std_0_variable_flags], axis=1)

# オートスケーリング
autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std()
autoscaled_x_test = (x_test - x_train.mean()) / x_train.std()

# 学習データをtrainデータとバリデーションデータに分割
autoscaled_x_train_tmp, autoscaled_x_validation, autoscaled_y_train_tmp, autoscaled_y_validation = train_test_split(
    autoscaled_x_train,
    autoscaled_y_train,
    test_size=fraction_of_validation_samples,
    random_state=123
)

inner_fold_number = len(x) - 1
outer_fold_number = len(x)

# 実行時間を記録する辞書
model_exe_time = {}

'''
PLS --------------------------------------------------------------------------------------------------------
'''
if exe_time:
    start = time.time()

output_folder_name = './PLS'
os.makedirs(output_folder_name, exist_ok=True)

if not os.path.exists(output_folder_name + '/model.pkl'):
    pls_components = np.arange(1, max_number_of_principal_components + 1)
    inner_cv = DCEGridSearchCV(PLSRegression(), {'n_components': pls_components}, cv=inner_fold_number)
    estimated_y = double_cross_validation(
        gs_cv=inner_cv,
        x=autoscaled_x,
        y=y,
        outer_fold_number=outer_fold_number,
        do_autoscaling=True,
        random_state=0
    )
    r2_in_cv = metrics.r2_score(y, estimated_y)
    print(r2_in_cv)

# y-yプロットを作成 (タイトルに save_path を使うため、関数内で save_path を反映)
plot_path = os.path.join(output_folder_name, 'y_vs_estimated_y.png')
plot_y_vs_estimated_y(y, estimated_y, plot_path)

# メトリクスの計算
r2_dcv = metrics.r2_score(y, estimated_y)
rmse_dcv = np.sqrt(metrics.mean_squared_error(y, estimated_y))
mae_dcv = metrics.mean_absolute_error(y, estimated_y)
print(f'r2dcv: {r2_dcv}')
print(f'RMSEdcv: {rmse_dcv}')
print(f'MAEdcv: {mae_dcv}')

# メトリクスの保存
metrics_text = f"""Model Evaluation Metrics:
R²dcv: {r2_dcv}
RMSEdcv: {rmse_dcv}
MAEdcv: {mae_dcv}
"""
metrics_path = os.path.join(output_folder_name, 'metrics.txt')
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write(metrics_text)
print(f"メトリクスのテキストファイルを保存しました: {metrics_path}")

# 推定値・実測値・誤差をCSVで保存
estimated_y_df = pd.DataFrame(estimated_y, index=y.index, columns=['estimated_y'])
actual_y_df = pd.DataFrame(y, index=y.index)
actual_y_df.columns = ['actual_y']
error_df = actual_y_df['actual_y'] - estimated_y_df['estimated_y']
error_df = error_df.to_frame(name='error_of_y')
results_df = pd.concat([estimated_y_df, actual_y_df, error_df], axis=1)
save_path = os.path.join(output_folder_name, 'estimated_y_with_actual_error.csv')
results_df.to_csv(save_path, index=True)
print(f"推定値・実測値・誤差をまとめて保存しました: {save_path}")

if exe_time:
    process_time = time.time() - start
    model_exe_time[output_folder_name.split('/')[1]] = process_time


'''
SVR --------------------------------------------------------------------------------------------------------
'''
if exe_time:
    start = time.time()

output_folder_name = './SVR'
os.makedirs(output_folder_name, exist_ok=True)

if not os.path.exists(output_folder_name + '/model.pkl'):
    svr_params = {
        'C': svr_cs,
        'epsilon': svr_epsilons,
        'gamma': svr_gammas
    }
    inner_cv = DCEGridSearchCV(svm.SVR(), svr_params, cv=inner_fold_number)
    estimated_y = double_cross_validation(
        gs_cv=inner_cv,
        x=autoscaled_x,
        y=y,
        outer_fold_number=outer_fold_number,
        do_autoscaling=False,
        random_state=0
    )
    r2_in_cv = metrics.r2_score(y, estimated_y)
    print(f"SVR R² in CV: {r2_in_cv}")

plot_path = os.path.join(output_folder_name, 'y_vs_estimated_y.png')
plot_y_vs_estimated_y(y, estimated_y, plot_path)

r2_dcv = metrics.r2_score(y, estimated_y)
rmse_dcv = np.sqrt(metrics.mean_squared_error(y, estimated_y))
mae_dcv = metrics.mean_absolute_error(y, estimated_y)
print(f'r2dcv: {r2_dcv}')
print(f'RMSEdcv: {rmse_dcv}')
print(f'MAEdcv: {mae_dcv}')

metrics_text = f"""Model Evaluation Metrics:
R²dcv: {r2_dcv}
RMSEdcv: {rmse_dcv}
MAEdcv: {mae_dcv}
"""
metrics_path = os.path.join(output_folder_name, 'metrics.txt')
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write(metrics_text)
print(f"メトリクスのテキストファイルを保存しました: {metrics_path}")

estimated_y_df = pd.DataFrame(estimated_y, index=y.index, columns=['estimated_y'])
actual_y_df = pd.DataFrame(y, index=y.index)
actual_y_df.columns = ['actual_y']
error_df = actual_y_df['actual_y'] - estimated_y_df['estimated_y']
error_df = error_df.to_frame(name='error_of_y')
results_df = pd.concat([estimated_y_df, actual_y_df, error_df], axis=1)
save_path = os.path.join(output_folder_name, 'estimated_y_with_actual_error.csv')
results_df.to_csv(save_path, index=True)
print(f"推定値・実測値・誤差をまとめて保存しました: {save_path}")

if exe_time:
    process_time = time.time() - start
    model_exe_time[output_folder_name.split('/')[1]] = process_time


'''
DT(決定木) --------------------------------------------------------------------------------------------------
'''
if exe_time:
    start = time.time()

output_folder_name = './DT'
os.makedirs(output_folder_name, exist_ok=True)

if not os.path.exists(output_folder_name + '/model.pkl'):
    dt_params = {'max_depth': max_depths}
    inner_cv = DCEGridSearchCV(
        estimator=DecisionTreeRegressor(random_state=0, min_samples_leaf=3),
        param_grid=dt_params,
        cv=inner_fold_number
    )
    estimated_y = double_cross_validation(
        gs_cv=inner_cv,
        x=x,
        y=y,
        outer_fold_number=outer_fold_number,
        do_autoscaling=False,
        random_state=0
    )
    r2_in_cv = metrics.r2_score(y, estimated_y)
    print(f"DT R² in CV: {r2_in_cv}")

plot_path = os.path.join(output_folder_name, 'y_vs_estimated_y.png')
plot_y_vs_estimated_y(y, estimated_y, plot_path)

r2_dcv = metrics.r2_score(y, estimated_y)
rmse_dcv = np.sqrt(metrics.mean_squared_error(y, estimated_y))
mae_dcv = metrics.mean_absolute_error(y, estimated_y)
print(f'r2dcv: {r2_dcv}')
print(f'RMSEdcv: {rmse_dcv}')
print(f'MAEdcv: {mae_dcv}')

metrics_text = f"""Model Evaluation Metrics:
R²dcv: {r2_dcv}
RMSEdcv: {rmse_dcv}
MAEdcv: {mae_dcv}
"""
metrics_path = os.path.join(output_folder_name, 'metrics.txt')
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write(metrics_text)
print(f"メトリクスのテキストファイルを保存しました: {metrics_path}")

estimated_y_df = pd.DataFrame(estimated_y, index=y.index, columns=['estimated_y'])
actual_y_df = pd.DataFrame(y, index=y.index)
actual_y_df.columns = ['actual_y']
error_df = actual_y_df['actual_y'] - estimated_y_df['estimated_y']
error_df = error_df.to_frame(name='error_of_y')
results_df = pd.concat([estimated_y_df, actual_y_df, error_df], axis=1)
save_path = os.path.join(output_folder_name, 'estimated_y_with_actual_error.csv')
results_df.to_csv(save_path, index=True)
print(f"推定値・実測値・誤差をまとめて保存しました: {save_path}")

if exe_time:
    process_time = time.time() - start
    model_exe_time[output_folder_name.split('/')[1]] = process_time


'''
RF ---------------------------------------------------------------------------------------------------------
'''
if exe_time:
    start = time.time()

output_folder_name = './RF'
os.makedirs(output_folder_name, exist_ok=True)

if not os.path.exists(output_folder_name + '/model.pkl'):
    rf_params = {
        'max_features': ratios_of_x
    }
    inner_cv = DCEGridSearchCV(
        estimator=RandomForestRegressor(random_state=0, n_estimators=n_estimators),
        param_grid=rf_params,
        cv=inner_fold_number
    )
    estimated_y = double_cross_validation(
        gs_cv=inner_cv,
        x=x,
        y=y,
        outer_fold_number=outer_fold_number,
        do_autoscaling=False,
        random_state=0
    )
    r2_in_cv = metrics.r2_score(y, estimated_y)
    print(f"Random Forest R² in CV: {r2_in_cv}")

plot_path = os.path.join(output_folder_name, 'y_vs_estimated_y.png')
plot_y_vs_estimated_y(y, estimated_y, plot_path)

r2_dcv = metrics.r2_score(y, estimated_y)
rmse_dcv = np.sqrt(metrics.mean_squared_error(y, estimated_y))
mae_dcv = metrics.mean_absolute_error(y, estimated_y)
print(f'r2dcv: {r2_dcv}')
print(f'RMSEdcv: {rmse_dcv}')
print(f'MAEdcv: {mae_dcv}')

metrics_text = f"""Model Evaluation Metrics:
R²dcv: {r2_dcv}
RMSEdcv: {rmse_dcv}
MAEdcv: {mae_dcv}
"""
metrics_path = os.path.join(output_folder_name, 'metrics.txt')
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write(metrics_text)
print(f"メトリクスのテキストファイルを保存しました: {metrics_path}")

estimated_y_df = pd.DataFrame(estimated_y, index=y.index, columns=['estimated_y'])
actual_y_df = pd.DataFrame(y, index=y.index)
actual_y_df.columns = ['actual_y']
error_df = actual_y_df['actual_y'] - estimated_y_df['estimated_y']
error_df = error_df.to_frame(name='error_of_y')
results_df = pd.concat([estimated_y_df, actual_y_df, error_df], axis=1)
save_path = os.path.join(output_folder_name, 'estimated_y_with_actual_error.csv')
results_df.to_csv(save_path, index=True)
print(f"推定値・実測値・誤差をまとめて保存しました: {save_path}")

if exe_time:
    process_time = time.time() - start
    model_exe_time[output_folder_name.split('/')[1]] = process_time


'''
GPR (Gaussian Process Regression) ---------------------------------------------------------------------------
'''
if exe_time:
    start = time.time()

output_folder_name = './GPR'
os.makedirs(output_folder_name, exist_ok=True)

kfold = KFold(n_splits=outer_fold_number, shuffle=False, random_state=0)
estimated_y_all = np.zeros(len(y))

# カーネル定義
k1 = kernels.Sum(C() * kernels.RBF(), C() * kernels.RationalQuadratic())
k2 = kernels.Sum(C() * kernels.Matern(), C() * kernels.ExpSineSquared())
k3 = kernels.Sum(C() * kernels.DotProduct(), kernels.ConstantKernel())
ks = kernels.Sum(k1, k2)
ks = kernels.Sum(ks, k3)
sum_kernel = kernels.Sum(ks, kernels.WhiteKernel())

fold_id = 1
for train_idx, test_idx in kfold.split(x):
    print(f"\n========== Fold {fold_id} ==========")
    x_train_cv = autoscaled_x.iloc[train_idx, :]
    y_train_cv = y.iloc[train_idx]
    x_test_cv = autoscaled_x.iloc[test_idx, :]
    y_test_cv = y.iloc[test_idx]

    model_cv = make_pipeline(
        StandardScaler(),
        gp.GaussianProcessRegressor(kernel=sum_kernel)
    )
    model_cv.fit(x_train_cv, y_train_cv)
    y_pred_test_cv, y_std_test_cv = model_cv.predict(x_test_cv, return_std=True)
    estimated_y_all[test_idx] = y_pred_test_cv

    r2_fold = metrics.r2_score(y_test_cv, y_pred_test_cv)
    rmse_fold = np.sqrt(metrics.mean_squared_error(y_test_cv, y_pred_test_cv))
    mae_fold = metrics.mean_absolute_error(y_test_cv, y_pred_test_cv)
    print(f"Fold {fold_id} R2 = {r2_fold:.3f}, RMSE = {rmse_fold:.3f}, MAE = {mae_fold:.3f}")
    fold_id += 1

r2_dcv = metrics.r2_score(y, estimated_y_all)
rmse_dcv = np.sqrt(metrics.mean_squared_error(y, estimated_y_all))
mae_dcv = metrics.mean_absolute_error(y, estimated_y_all)
print("\n=== Overall CV Results ===")
print(f"R²dcv   : {r2_dcv:.3f}")
print(f"RMSEdcv : {rmse_dcv:.3f}")
print(f"MAEdcv  : {mae_dcv:.3f}")

plot_path = os.path.join(output_folder_name, 'y_vs_estimated_y.png')
# GPRの場合は estimated_y_all を使う
plot_y_vs_estimated_y(y, estimated_y_all, plot_path)

metrics_text = f"""Model Evaluation Metrics (DCV):
R²cv: {r2_dcv}
RMSEcv: {rmse_dcv}
MAEcv: {mae_dcv}
"""
metrics_path = os.path.join(output_folder_name, 'metrics.txt')
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write(metrics_text)
print(f"\nメトリクスのテキストファイルを保存しました: {metrics_path}")

estimated_y_df = pd.DataFrame(estimated_y_all, index=y.index, columns=['estimated_y'])
actual_y_df = pd.DataFrame(y, index=y.index)
actual_y_df.columns = ['actual_y']
error_df = actual_y_df['actual_y'] - estimated_y_df['estimated_y']
error_df = error_df.to_frame(name='error_of_y')
results_df = pd.concat([estimated_y_df, actual_y_df, error_df], axis=1)
save_path = os.path.join(output_folder_name, 'estimated_y_with_actual_error.csv')
results_df.to_csv(save_path, index=True)
print(f"推定値・実測値・誤差をまとめて保存しました: {save_path}")

if exe_time:
    process_time = time.time() - start
    print(f"\n処理時間: {process_time:.2f} 秒\n")


'''
Lasso -------------------------------------------------------------------------------------------------------
'''
if exe_time:
    start = time.time()

output_folder_name = './LASSO'
os.makedirs(output_folder_name, exist_ok=True)

if not os.path.exists(os.path.join(output_folder_name, 'model.pkl')):
    lasso_params = {'alpha': lasso_lambdas}
    inner_cv = DCEGridSearchCV(
        estimator=Lasso(max_iter=100000),
        param_grid=lasso_params,
        cv=inner_fold_number
    )
    estimated_y = double_cross_validation(
        gs_cv=inner_cv,
        x=autoscaled_x,
        y=y,
        outer_fold_number=outer_fold_number,
        do_autoscaling=False,
        random_state=0
    )
    r2_in_cv = metrics.r2_score(y, estimated_y)
    print(f"Lasso R² in CV: {r2_in_cv}")

plot_path = os.path.join(output_folder_name, 'y_vs_estimated_y.png')
plot_y_vs_estimated_y(y, estimated_y, plot_path)

r2_dcv = metrics.r2_score(y, estimated_y)
rmse_dcv = np.sqrt(metrics.mean_squared_error(y, estimated_y))
mae_dcv = metrics.mean_absolute_error(y, estimated_y)
print(f'r2dcv: {r2_dcv}')
print(f'RMSEdcv: {rmse_dcv}')
print(f'MAEdcv: {mae_dcv}')

metrics_text = f"""Model Evaluation Metrics:
R²dcv: {r2_dcv}
RMSEdcv: {rmse_dcv}
MAEdcv: {mae_dcv}
"""
metrics_path = os.path.join(output_folder_name, 'metrics.txt')
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write(metrics_text)
print(f"メトリクスのテキストファイルを保存しました: {metrics_path}")

estimated_y_df = pd.DataFrame(estimated_y, index=y.index, columns=['estimated_y'])
actual_y_df = pd.DataFrame(y, index=y.index)
actual_y_df.columns = ['actual_y']
error_df = actual_y_df['actual_y'] - estimated_y_df['estimated_y']
error_df = error_df.to_frame(name='error_of_y')
results_df = pd.concat([estimated_y_df, actual_y_df, error_df], axis=1)
save_path = os.path.join(output_folder_name, 'estimated_y_with_actual_error.csv')
results_df.to_csv(save_path, index=True)
print(f"推定値・実測値・誤差をまとめて保存しました: {save_path}")

if exe_time:
    process_time = time.time() - start
    model_exe_time[output_folder_name.split('/')[1]] = process_time


'''
Ridge -------------------------------------------------------------------------------------------------------
'''
if exe_time:
    start = time.time()

output_folder_name = './RIDGE'
os.makedirs(output_folder_name, exist_ok=True)

if not os.path.exists(os.path.join(output_folder_name, 'model.pkl')):
    ridge_params = {'alpha': ridge_lambdas}
    inner_cv = DCEGridSearchCV(
        estimator=Ridge(),
        param_grid=ridge_params,
        cv=inner_fold_number
    )
    estimated_y = double_cross_validation(
        gs_cv=inner_cv,
        x=autoscaled_x,
        y=y,
        outer_fold_number=outer_fold_number,
        do_autoscaling=False,
        random_state=0
    )
    r2_in_cv = metrics.r2_score(y, estimated_y)
    print(f"Ridge R² in CV: {r2_in_cv}")

plot_path = os.path.join(output_folder_name, 'y_vs_estimated_y.png')
plot_y_vs_estimated_y(y, estimated_y, plot_path)

r2_dcv = metrics.r2_score(y, estimated_y)
rmse_dcv = np.sqrt(metrics.mean_squared_error(y, estimated_y))
mae_dcv = metrics.mean_absolute_error(y, estimated_y)
print(f'r2dcv: {r2_dcv}')
print(f'RMSEdcv: {rmse_dcv}')
print(f'MAEdcv: {mae_dcv}')

metrics_text = f"""Model Evaluation Metrics:
R²dcv: {r2_dcv}
RMSEdcv: {rmse_dcv}
MAEdcv: {mae_dcv}
"""
metrics_path = os.path.join(output_folder_name, 'metrics.txt')
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write(metrics_text)
print(f"メトリクスのテキストファイルを保存しました: {metrics_path}")

estimated_y_df = pd.DataFrame(estimated_y, index=y.index, columns=['estimated_y'])
actual_y_df = pd.DataFrame(y, index=y.index)
actual_y_df.columns = ['actual_y']
error_df = actual_y_df['actual_y'] - estimated_y_df['estimated_y']
error_df = error_df.to_frame(name='error_of_y')
results_df = pd.concat([estimated_y_df, actual_y_df, error_df], axis=1)
save_path = os.path.join(output_folder_name, 'estimated_y_with_actual_error.csv')
results_df.to_csv(save_path, index=True)
print(f"推定値・実測値・誤差をまとめて保存しました: {save_path}")

if exe_time:
    process_time = time.time() - start
    model_exe_time[output_folder_name.split('/')[1]] = process_time


'''
ElasticNet --------------------------------------------------------------------------------------------------
'''
if exe_time:
    start = time.time()

output_folder_name = './ElasticNet'
os.makedirs(output_folder_name, exist_ok=True)

if not os.path.exists(os.path.join(output_folder_name, 'model.pkl')):
    enet_params = {
        'l1_ratio': elastic_net_lambdas,
        'alpha': elastic_net_alphas
    }
    inner_cv = DCEGridSearchCV(
        estimator=ElasticNet(max_iter=100000),
        param_grid=enet_params,
        cv=inner_fold_number
    )
    estimated_y = double_cross_validation(
        gs_cv=inner_cv,
        x=autoscaled_x,
        y=y,
        outer_fold_number=outer_fold_number,
        do_autoscaling=False,
        random_state=0
    )
    r2_in_cv = metrics.r2_score(y, estimated_y)
    print(f"ElasticNet R² in CV: {r2_in_cv}")

plot_path = os.path.join(output_folder_name, 'y_vs_estimated_y.png')
plot_y_vs_estimated_y(y, estimated_y, plot_path)

r2_dcv = metrics.r2_score(y, estimated_y)
rmse_dcv = np.sqrt(metrics.mean_squared_error(y, estimated_y))
mae_dcv = metrics.mean_absolute_error(y, estimated_y)
print(f'r2dcv: {r2_dcv}')
print(f'RMSEdcv: {rmse_dcv}')
print(f'MAEdcv: {mae_dcv}')

metrics_text = f"""Model Evaluation Metrics:
R²dcv: {r2_dcv}
RMSEdcv: {rmse_dcv}
MAEdcv: {mae_dcv}
"""
metrics_path = os.path.join(output_folder_name, 'metrics.txt')
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write(metrics_text)
print(f"メトリクスのテキストファイルを保存しました: {metrics_path}")

estimated_y_df = pd.DataFrame(estimated_y, index=y.index, columns=['estimated_y'])
actual_y_df = pd.DataFrame(y, index=y.index)
actual_y_df.columns = ['actual_y']
error_df = actual_y_df['actual_y'] - estimated_y_df['estimated_y']
error_df = error_df.to_frame(name='error_of_y')
results_df = pd.concat([estimated_y_df, actual_y_df, error_df], axis=1)
save_path = os.path.join(output_folder_name, 'estimated_y_with_actual_error.csv')
results_df.to_csv(save_path, index=True)
print(f"推定値・実測値・誤差をまとめて保存しました: {save_path}")

if exe_time:
    process_time = time.time() - start
    model_exe_time[output_folder_name.split('/')[1]] = process_time


'''
LightGBM_default --------------------------------------------------------------------------------------------
'''
if exe_time:
    start = time.time()

output_folder_name = './LightGBM'
os.makedirs(output_folder_name, exist_ok=True)

kfold = KFold(n_splits=outer_fold_number, shuffle=False, random_state=0)
estimated_y_all = np.zeros(len(y))

fold_id = 1
for train_idx, test_idx in kfold.split(x):
    print(f"\n========== Fold {fold_id} ==========")
    x_train_cv = autoscaled_x.iloc[train_idx, :]
    y_train_cv = y.iloc[train_idx]
    x_test_cv = autoscaled_x.iloc[test_idx, :]
    y_test_cv = y.iloc[test_idx]

    model_cv = lgb.LGBMRegressor(random_state=0, n_jobs=-1)
    model_cv.fit(x_train_cv, y_train_cv)
    y_pred_test_cv = model_cv.predict(x_test_cv)
    estimated_y_all[test_idx] = y_pred_test_cv

    r2_fold = metrics.r2_score(y_test_cv, y_pred_test_cv)
    rmse_fold = np.sqrt(metrics.mean_squared_error(y_test_cv, y_pred_test_cv))
    mae_fold = metrics.mean_absolute_error(y_test_cv, y_pred_test_cv)
    print(f"Fold {fold_id} R2 = {r2_fold:.3f}, RMSE = {rmse_fold:.3f}, MAE = {mae_fold:.3f}")
    fold_id += 1

r2_dcv = metrics.r2_score(y, estimated_y_all)
rmse_dcv = np.sqrt(metrics.mean_squared_error(y, estimated_y_all))
mae_dcv = metrics.mean_absolute_error(y, estimated_y_all)
print("\n=== Overall CV Results ===")
print(f"R²dcv   : {r2_dcv:.3f}")
print(f"RMSEdcv : {rmse_dcv:.3f}")
print(f"MAEdcv  : {mae_dcv:.3f}")

plot_path = os.path.join(output_folder_name, 'y_vs_estimated_y.png')
plot_y_vs_estimated_y(y, estimated_y_all, plot_path)

metrics_text = f"""Model Evaluation Metrics (DCV):
R²cv: {r2_dcv}
RMSEcv: {rmse_dcv}
MAEcv: {mae_dcv}
"""
metrics_path = os.path.join(output_folder_name, 'metrics.txt')
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write(metrics_text)
print(f"\nメトリクスのテキストファイルを保存しました: {metrics_path}")

estimated_y_df = pd.DataFrame(estimated_y_all, index=y.index, columns=['estimated_y'])
actual_y_df = pd.DataFrame(y, index=y.index)
actual_y_df.columns = ['actual_y']
error_df = actual_y_df['actual_y'] - estimated_y_df['estimated_y']
error_df = error_df.to_frame(name='error_of_y')
results_df = pd.concat([estimated_y_df, actual_y_df, error_df], axis=1)
save_path = os.path.join(output_folder_name, 'estimated_y_with_actual_error.csv')
results_df.to_csv(save_path, index=True)
print(f"推定値・実測値・誤差をまとめて保存しました: {save_path}")

if exe_time:
    process_time = time.time() - start
    print(f"\n処理時間: {process_time:.2f} 秒\n")


'''
XGBoost_default ---------------------------------------------------------------------------------------------
'''
if exe_time:
    start = time.time()

output_folder_name = './XGBoost_default'
os.makedirs(output_folder_name, exist_ok=True)

kfold = KFold(n_splits=outer_fold_number, shuffle=False, random_state=0)
estimated_y_all = np.zeros(len(y))

fold_id = 1
for train_idx, test_idx in kfold.split(x):
    print(f"\n========== Fold {fold_id} ==========")
    x_train_cv = autoscaled_x.iloc[train_idx, :]
    y_train_cv = y.iloc[train_idx]
    x_test_cv = autoscaled_x.iloc[test_idx, :]
    y_test_cv = y.iloc[test_idx]

    model_cv = xgb.XGBRegressor(random_state=0, n_jobs=-1)
    model_cv.fit(x_train_cv, y_train_cv)
    y_pred_test_cv = model_cv.predict(x_test_cv)
    estimated_y_all[test_idx] = y_pred_test_cv

    r2_fold = metrics.r2_score(y_test_cv, y_pred_test_cv)
    rmse_fold = np.sqrt(metrics.mean_squared_error(y_test_cv, y_pred_test_cv))
    mae_fold = metrics.mean_absolute_error(y_test_cv, y_pred_test_cv)
    print(f"Fold {fold_id} R2 = {r2_fold:.3f}, RMSE = {rmse_fold:.3f}, MAE = {mae_fold:.3f}")
    fold_id += 1

r2_dcv = metrics.r2_score(y, estimated_y_all)
rmse_dcv = np.sqrt(metrics.mean_squared_error(y, estimated_y_all))
mae_dcv = metrics.mean_absolute_error(y, estimated_y_all)
print("\n=== Overall CV Results ===")
print(f"R²dcv   : {r2_dcv:.3f}")
print(f"RMSEdcv : {rmse_dcv:.3f}")
print(f"MAEdcv  : {mae_dcv:.3f}")

plot_path = os.path.join(output_folder_name, 'y_vs_estimated_y.png')
plot_y_vs_estimated_y(y, estimated_y_all, plot_path)

metrics_text = f"""Model Evaluation Metrics (DCV):
R²cv: {r2_dcv}
RMSEcv: {rmse_dcv}
MAEcv: {mae_dcv}
"""
metrics_path = os.path.join(output_folder_name, 'metrics.txt')
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write(metrics_text)
print(f"\nメトリクスのテキストファイルを保存しました: {metrics_path}")

estimated_y_df = pd.DataFrame(estimated_y_all, index=y.index, columns=['estimated_y'])
actual_y_df = pd.DataFrame(y, index=y.index)
actual_y_df.columns = ['actual_y']
error_df = actual_y_df['actual_y'] - estimated_y_df['estimated_y']
error_df = error_df.to_frame(name='error_of_y')
results_df = pd.concat([estimated_y_df, actual_y_df, error_df], axis=1)
save_path = os.path.join(output_folder_name, 'estimated_y_with_actual_error.csv')
results_df.to_csv(save_path, index=True)
print(f"推定値・実測値・誤差をまとめて保存しました: {save_path}")

if exe_time:
    process_time = time.time() - start
    print(f"\n処理時間: {process_time:.2f} 秒\n")


'''
GBDT_default ------------------------------------------------------------------------------------------------
'''
if exe_time:
    start = time.time()

output_folder_name = './GBDT_default'
os.makedirs(output_folder_name, exist_ok=True)

kfold = KFold(n_splits=outer_fold_number, shuffle=False, random_state=0)
estimated_y_all = np.zeros(len(y))

fold_id = 1
for train_idx, test_idx in kfold.split(x):
    print(f"\n========== Fold {fold_id} ==========")
    x_train_cv = autoscaled_x.iloc[train_idx, :]
    y_train_cv = y.iloc[train_idx]
    x_test_cv = autoscaled_x.iloc[test_idx, :]
    y_test_cv = y.iloc[test_idx]

    model_cv = GradientBoostingRegressor(random_state=0)
    model_cv.fit(x_train_cv, y_train_cv)
    y_pred_test_cv = model_cv.predict(x_test_cv)
    estimated_y_all[test_idx] = y_pred_test_cv

    r2_fold = metrics.r2_score(y_test_cv, y_pred_test_cv)
    rmse_fold = np.sqrt(metrics.mean_squared_error(y_test_cv, y_pred_test_cv))
    mae_fold = metrics.mean_absolute_error(y_test_cv, y_pred_test_cv)
    print(f"Fold {fold_id} R2 = {r2_fold:.3f}, RMSE = {rmse_fold:.3f}, MAE = {mae_fold:.3f}")
    fold_id += 1

r2_dcv = metrics.r2_score(y, estimated_y_all)
rmse_dcv = np.sqrt(metrics.mean_squared_error(y, estimated_y_all))
mae_dcv = metrics.mean_absolute_error(y, estimated_y_all)
print("\n=== Overall CV Results ===")
print(f"R²dcv   : {r2_dcv:.3f}")
print(f"RMSEdcv : {rmse_dcv:.3f}")
print(f"MAEdcv  : {mae_dcv:.3f}")

plot_path = os.path.join(output_folder_name, 'y_vs_estimated_y.png')
plot_y_vs_estimated_y(y, estimated_y_all, plot_path)

metrics_text = f"""Model Evaluation Metrics (DCV):
R²cv: {r2_dcv}
RMSEcv: {rmse_dcv}
MAEcv: {mae_dcv}
"""
metrics_path = os.path.join(output_folder_name, 'metrics.txt')
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write(metrics_text)
print(f"\nメトリクスのテキストファイルを保存しました: {metrics_path}")

estimated_y_df = pd.DataFrame(estimated_y_all, index=y.index, columns=['estimated_y'])
actual_y_df = pd.DataFrame(y, index=y.index)
actual_y_df.columns = ['actual_y']
error_df = actual_y_df['actual_y'] - estimated_y_df['estimated_y']
error_df = error_df.to_frame(name='error_of_y')
results_df = pd.concat([estimated_y_df, actual_y_df, error_df], axis=1)
save_path = os.path.join(output_folder_name, 'estimated_y_with_actual_error.csv')
results_df.to_csv(save_path, index=True)
print(f"推定値・実測値・誤差をまとめて保存しました: {save_path}")

if exe_time:
    process_time = time.time() - start
    print(f"\n処理時間: {process_time:.2f} 秒\n")


'''
CatBoost_default --------------------------------------------------------------------------------------------
'''
if exe_time:
    start = time.time()

output_folder_name = './CatBoost_default'
os.makedirs(output_folder_name, exist_ok=True)

kfold = KFold(n_splits=outer_fold_number, shuffle=False, random_state=0)
estimated_y_all = np.zeros(len(y))

fold_id = 1
for train_idx, test_idx in kfold.split(x):
    print(f"\n========== Fold {fold_id} ==========")
    x_train_cv = autoscaled_x.iloc[train_idx, :]
    y_train_cv = y.iloc[train_idx]
    x_test_cv = autoscaled_x.iloc[test_idx, :]
    y_test_cv = y.iloc[test_idx]

    model_cv = cb.CatBoostRegressor(random_seed=0, verbose=False)
    model_cv.fit(x_train_cv, y_train_cv)
    y_pred_test_cv = model_cv.predict(x_test_cv)
    estimated_y_all[test_idx] = y_pred_test_cv

    r2_fold = metrics.r2_score(y_test_cv, y_pred_test_cv)
    rmse_fold = np.sqrt(metrics.mean_squared_error(y_test_cv, y_pred_test_cv))
    mae_fold = metrics.mean_absolute_error(y_test_cv, y_pred_test_cv)
    print(f"Fold {fold_id} R2 = {r2_fold:.3f}, RMSE = {rmse_fold:.3f}, MAE = {mae_fold:.3f}")
    fold_id += 1

r2_dcv = metrics.r2_score(y, estimated_y_all)
rmse_dcv = np.sqrt(metrics.mean_squared_error(y, estimated_y_all))
mae_dcv = metrics.mean_absolute_error(y, estimated_y_all)
print("\n=== Overall CV Results ===")
print(f"R²dcv   : {r2_dcv:.3f}")
print(f"RMSEdcv : {rmse_dcv:.3f}")
print(f"MAEdcv  : {mae_dcv:.3f}")

plot_path = os.path.join(output_folder_name, 'y_vs_estimated_y.png')
plot_y_vs_estimated_y(y, estimated_y_all, plot_path)

metrics_text = f"""Model Evaluation Metrics (DCV):
R²cv: {r2_dcv}
RMSEcv: {rmse_dcv}
MAEcv: {mae_dcv}
"""
metrics_path = os.path.join(output_folder_name, 'metrics.txt')
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write(metrics_text)
print(f"\nメトリクスのテキストファイルを保存しました: {metrics_path}")

estimated_y_df = pd.DataFrame(estimated_y_all, index=y.index, columns=['estimated_y'])
actual_y_df = pd.DataFrame(y, index=y.index)
actual_y_df.columns = ['actual_y']
error_df = actual_y_df['actual_y'] - estimated_y_df['estimated_y']
error_df = error_df.to_frame(name='error_of_y')
results_df = pd.concat([estimated_y_df, actual_y_df, error_df], axis=1)
save_path = os.path.join(output_folder_name, 'estimated_y_with_actual_error.csv')
results_df.to_csv(save_path, index=True)
print(f"推定値・実測値・誤差をまとめて保存しました: {save_path}")

if exe_time:
    process_time = time.time() - start
    print(f"\n処理時間: {process_time:.2f} 秒\n")


'''
DNN (scikit-learn MLP) --------------------------------------------------------------------------------------
'''
if exe_time:
    start = time.time()

output_folder_name = './DNN_default'
os.makedirs(output_folder_name, exist_ok=True)

from sklearn.neural_network import MLPRegressor

kfold = KFold(n_splits=outer_fold_number, shuffle=False, random_state=0)
estimated_y_all = np.zeros(len(y))

fold_id = 1
for train_idx, test_idx in kfold.split(x):
    print(f"\n========== Fold {fold_id} ==========")
    x_train_cv = autoscaled_x.iloc[train_idx, :]
    y_train_cv = y.iloc[train_idx]
    x_test_cv = autoscaled_x.iloc[test_idx, :]
    y_test_cv = y.iloc[test_idx]

    model_cv = MLPRegressor(
        hidden_layer_sizes=(100, 100),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=0
    )
    model_cv.fit(x_train_cv, y_train_cv)
    y_pred_test_cv = model_cv.predict(x_test_cv)
    estimated_y_all[test_idx] = y_pred_test_cv

    r2_fold = metrics.r2_score(y_test_cv, y_pred_test_cv)
    rmse_fold = np.sqrt(metrics.mean_squared_error(y_test_cv, y_pred_test_cv))
    mae_fold = metrics.mean_absolute_error(y_test_cv, y_pred_test_cv)
    print(f"Fold {fold_id} R2 = {r2_fold:.3f}, RMSE = {rmse_fold:.3f}, MAE = {mae_fold:.3f}")
    fold_id += 1

r2_dcv = metrics.r2_score(y, estimated_y_all)
rmse_dcv = np.sqrt(metrics.mean_squared_error(y, estimated_y_all))
mae_dcv = metrics.mean_absolute_error(y, estimated_y_all)
print("\n=== Overall CV Results ===")
print(f"R²dcv   : {r2_dcv:.3f}")
print(f"RMSEdcv : {rmse_dcv:.3f}")
print(f"MAEdcv  : {mae_dcv:.3f}")

plot_path = os.path.join(output_folder_name, 'y_vs_estimated_y.png')
plot_y_vs_estimated_y(y, estimated_y_all, plot_path)

metrics_text = f"""Model Evaluation Metrics (DCV):
R²cv: {r2_dcv}
RMSEcv: {rmse_dcv}
MAEcv: {mae_dcv}
"""
metrics_path = os.path.join(output_folder_name, 'metrics.txt')
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write(metrics_text)
print(f"\nメトリクスのテキストファイルを保存しました: {metrics_path}")

estimated_y_df = pd.DataFrame(estimated_y_all, index=y.index, columns=['estimated_y'])
actual_y_df = pd.DataFrame(y, index=y.index)
actual_y_df.columns = ['actual_y']
error_df = actual_y_df['actual_y'] - estimated_y_df['estimated_y']
error_df = error_df.to_frame(name='error_of_y')
results_df = pd.concat([estimated_y_df, actual_y_df, error_df], axis=1)
save_path = os.path.join(output_folder_name, 'estimated_y_with_actual_error.csv')
results_df.to_csv(save_path, index=True)
print(f"推定値・実測値・誤差をまとめて保存しました: {save_path}")

if exe_time:
    process_time = time.time() - start
    print(f"\n処理時間: {process_time:.2f} 秒\n")

# 実行時間の一覧を保存
df_exe_time = pd.DataFrame(model_exe_time.values(), index=model_exe_time.keys())
df_exe_time = df_exe_time.rename(columns={0: 'exe_time_sec'})
df_exe_time.to_csv('exe_time.csv')
