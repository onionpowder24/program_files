# -*- coding: utf-8 -*- 
# %reset -f

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from rdkit.Chem import Draw
from rdkit import Chem 
from rdkit.Chem.Descriptors import descList
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline 
import sklearn.gaussian_process as gp
from scipy.stats import norm 
from sklearn.gaussian_process import kernels
from sklearn.gaussian_process.kernels import ConstantKernel as C
import matplotlib as mpl 
import matplotlib.pyplot as plt
mpl.rcParams['font.size'] = 25.
mpl.rcParams['figure.figsize'] = [12., 8.]
from warnings import filterwarnings
filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)
import os
import cv2
import img2pdf
from PIL import Image
import shutil
from natsort import natsorted
import re

'''
手書き入力-------------------------------------------------
'''

df_train = pd.read_csv('./inputs/molecules_with_boiling_point.csv' )# SMILESという名前の列名を用意  
df_test = pd.read_csv('./inputs/experimental_candidate.csv')# SMILESという名前の列名を用意  
df_test_name = './inputs/experimental_candidate .csv'
df_test_name = re.split('[/.]', df_test_name)[-2]
calc_desc = True
npy_file =  './np_Save_X_test.npy'
max_TARGETs = [] 
min_TARGETs = ['BoilingPoint'] 
all_img_pdf = True
num_img_pdf = 5000

'''
獲得関数の定義-------------------------------------------------------------------
'''
#PI
def probability_improvement(x_new, gaussian_process, evaluated_loss,
                         greater_is_better=False):
    assert type(evaluated_loss)is np.ndarray, "evaluated_loss must be np.array"
    if len(x_new.shape)==1:
        x_new = x_new.reshape(1, -1)
    mu, sigma = gaussian_process.predict(x_new, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (greater_is_better) 
    with np.errstate(divide='ignore'):
        Z = loss_optimum / sigma
        proba_improvement = norm.cdf(scaling_factor*Z) 
        proba_improvement[sigma == 0.0] == 0.0
    if len(proba_improvement.shape)==1:
        proba_improvement = proba_improvement.reshape(-1, 1)
    return proba_improvement

#EI
def expected_improvement(x_new, gaussian_process, evaluated_loss,
                         greater_is_better=False):
    assert type(evaluated_loss)is np.ndarray, "evaluated_loss must be np.array"
    if len(x_new.shape)==1:
        x_new = x_new.reshape(1, -1)
    mu, sigma = gaussian_process.predict(x_new, return_std=True)
    mu = mu.reshape(-1,)
    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)
    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        ei = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] == 0.0
    return ei

#UCB(MI)
def confidence_bound(x_new, gaussian_process, evaluated_loss,
                         greater_is_better=False):
    """LCB (default) / UCB for maximization"""
    assert type(evaluated_loss)is np.ndarray, "evaluated_loss must be np.array"
    if len(x_new.shape)==1:
        x_new = x_new.reshape(1, -1)
    mu, sigma = gaussian_process.predict(x_new, return_std=True)
    n_samples = evaluated_loss.shape[0]
    with np.errstate(divide='ignore'):
        kappa = np.sqrt(np.log( n_samples )/n_samples)
        if greater_is_better:
            # upper confindence bound
            cb = mu + kappa * sigma
        else:
            # lower confindence bound
            cb = mu - kappa * sigma
    return cb 

'''
その他の関数の定義----------------------------------------------------------------
'''
def plot_samples(x_obs, y_obs, x_test, y_m, y_s):
    """
    Plot for sampled and estimated response
    """
    if len(y_obs.shape)==2:
        y_obs = y_obs.copy().reshape(-1,)
    n_samples = x_obs.shape[0] + x_test.shape[0]
    n_trains = x_obs.shape[0]
    plt.plot(range(n_trains), y_obs, 'o')
    plt.plot(range(n_trains, n_samples), y_m)
    plt.fill_between(range(n_trains, n_samples), y_m-y_s,y_m+y_s,
                    alpha=.3, color='b')
    plt.xlim([0, n_samples])
    plt.ylim(np.array([np.amin(y_m-y_s) + np.amin(y_m-y_s)/5, np.amax(y_m+y_s) + np.amax(y_m+y_s) / 5]))
    plt.xlabel('sample index'); plt.ylabel('$y$')    
    
def img_to_pdf(img_list, pdf_file_path, pdf_file_name):
    '''
    画像20枚を連結して一枚の画像(縦4枚横5枚)にして、それをpdf1ページ分としたpdfを作成
    画像の枚数が20で割り切れないとき、白画像で足りない分を埋める
    
    Parameters
    ----------
    img_list : list
        画像の配列が格納されたリスト
    pdf_file_path : str
        pdfファイルを保存するディレクトリの相対パス
    pdf_file_name : str
        保存するpdfファイル名

    Returns
    -------
    None.

    '''
    
    mod = len(img_list) % 20  
    if mod != 0:
        lack_img_num = 20 - mod
        for i in range(lack_img_num):
            img_list.append(np.ones((img_list[0].shape[0], img_list[0].shape[1], img_list[0].shape[2]), np.uint8)*255)
    
    # 連結した20枚の画像を一時的に保存するディレクトリ
    os.mkdir('./Tmp') 
    
    # 連結した画像を保存するときのインデックス
    a = 0
    
    # 画像の連結
    concat_img_list = []
    for i in range(0, len(img_list[:-1]), 20):
        # 画像を5枚ずつ横に連結
        img_1 = cv2.hconcat([*img_list[i:i+5]])
        img_2 = cv2.hconcat([*img_list[i+5:i+10]])
        img_3 = cv2.hconcat([*img_list[i+10:i+15]])
        img_4 = cv2.hconcat([*img_list[i+15:i+20]])
        im_v = cv2.vconcat([img_1, img_2, img_3, img_4]) # 横に連結した5枚の画像×4を縦に連結
        cv2.imwrite('./Tmp/' + str(a) + '.png',im_v) # 連結した画像の保存
        a += 1        
    
    conca_png_folder = './Tmp/' # 連結した画像フォルダ
    concat_img_list = os.listdir(conca_png_folder) #ファイル名の取得
    concat_img_list = natsorted(concat_img_list) #自然順でソート
    
    # 連結した画像のpdf化    
    with open(os.path.join(pdf_file_path, pdf_file_name),"wb") as f:
        # 画像フォルダの中にあるPNGファイルを取得し配列に追加、バイナリ形式でファイルに書き込む
        f.write(img2pdf.convert([Image.open(conca_png_folder+j).filename for j in concat_img_list]))
    shutil.rmtree('./Tmp/') # 一時フォルダの削除
    
    
'''   
カーネル関数の定義-----------------------------------------------------------------
'''
#周辺尤度最大化の過程で不要なカーネル関数は重みが小さくなり、適切なカーネルの組み合わせが選ばれる。
k1 = kernels.Sum(C()*kernels.RBF(), C()*kernels.RationalQuadratic())
k2 = kernels.Sum(C()*kernels.Matern(), C()*kernels.ExpSineSquared())
k3 = kernels.Sum(C()*kernels.DotProduct(), kernels.ConstantKernel())
ks = kernels.Sum(k1, k2)
ks = kernels.Sum(ks, k3)
sum_kernel = kernels.Sum(ks, kernels.WhiteKernel())

'''
データ前処理----------------------------------------------------------------------
'''

#学習データの作成
df_train['mol'] = df_train['SMILES'].apply(Chem.MolFromSmiles) #SMILESからmol列を作成
#学習データの記述子抜き出し
X_train = np.array([list(map(lambda f: f[1](m), descList))for m in df_train['mol']])

#テストデータの作成
#一度記述子を計算している場合はnpyファイルを読み込むだけで終わる
if calc_desc == True: 
    df_test['mol'] = df_test['SMILES'].apply(Chem.MolFromSmiles) #SMILESからmol列を作成
    #テストデータの記述子抜き出し
    X_test = np.array([list(map(lambda f: f[1](m), descList))for m in df_test['mol']])
    #記述子データの保存(テストデータ名.npy)
    np.save(df_test_name,X_test)
else:
    #testデータのロード
    X_test = np.load(npy_file)

#テストデータ数の読み込み
number_of_test_samples = len(df_test)

#トレーニングデータとテストデータを結合しデータフレーム化
x = (np.concatenate([X_train, X_test], 0)) #トレーニングデータとテストデータを結合
x = pd.DataFrame(x) #データフレーム化
x = x.replace(np.inf, np.nan).fillna(np.nan)  # inf を NaN に置き換え
nan_variable_flags = x.isnull().any()  # NaN を含む変数
x = x.drop(x.columns[nan_variable_flags], axis=1)  # NaN を含む変数を削除

#トレーニングデータとテストデータに分割 (上記のNaN削除、データフレーム化を行ってから分割したかった)
x_train, x_test = train_test_split(x, test_size=number_of_test_samples, shuffle=False,
                                                    random_state=0)

#標準偏差が0の説明変数を削除(トレーニングとテスト)
std_0_variable_flags = x_train.std() == 0
x_train = x_train.drop(x_train.columns[std_0_variable_flags], axis=1)
x_test = x_test.drop(x_test.columns[std_0_variable_flags], axis=1)

x = np.concatenate([x_train, x_test], axis=0) #x_trainとx_testを結合
df_smile = df_test.SMILES #元の候補化合物のSMILES列の作成


#データフレームをマトリックスに変換
Xsc_all = x.copy()
Xsc_initial = x_train.copy()
Xsc_test = x_test.copy()

#乱数の固定
np.random.seed(42)


'''
機械学習(ベイズ最適化)------------------------------------------------------------
'''

#index の取得
X=Xsc_all
index_initial = df_train.index.tolist()
index_test = [num for num in np.arange(X.shape[0]) if num not in index_initial]

Xtrain = Xsc_all[index_initial,:].copy()
Xtest = Xsc_all[index_test,:].copy()

#目的変数の数を取得
num_of_obj = len(max_TARGETs) + len(min_TARGETs)


'''
#目的変数を最大化するベイズ最適化
for max_TARGET in max_TARGETs: 
    
    #出力フォルダ作成
    output_folder_name = './outputs/EI_' + max_TARGET[:]
    os.makedirs(output_folder_name, exist_ok=True)
        
    #目的変数を設定
    ytrain = df_train[max_TARGET].values.copy()    
    
    #GPRに標準化とカーネルの選択を追加しgprとする
    gpr = make_pipeline(StandardScaler(), 
                            gp.GaussianProcessRegressor(kernel=sum_kernel, normalize_y=True))      
    #ベイズ最適化の実行
    gpr.fit(Xtrain, ytrain)
    
    #テストデータのEIの計算
    test_EI = [expected_improvement(Xsc_all[idx,:], gpr, ytrain, greater_is_better = True).reshape(-1,)for idx in index_test]
    test_EI = np.array(test_EI)
    test_EI = test_EI.reshape(-1,)
    
    #推定値と推定値の標準偏差が入ったデータフレームを作成
    y_m, y_s = gpr.predict(x_test, return_std=True)
    y_m = y_m.reshape(-1,)
    y_m_df = pd.Series(y_m, name='Estimated_value')
    y_s_df = pd.Series(y_s, name='std_of_Estimated_value')
    #マトリックスのEIをデータフレーム化し、推定値の平均値、標準偏差、実験パラメータ、SMILES列を追加。
    #その後、更新幅の期待値順にソートする   
    EI_df = pd.Series(test_EI, name='Expected_value_update_width') #データフレーム化
    EI_df = pd.concat([EI_df, y_m_df, y_s_df, df_smile], axis = 1) #EIにSMILES列(実験パラメータ列含む)を追加
    EI_df = EI_df.sort_values(by='Expected_value_update_width', ascending=False) #更新幅の期待値順にソート
    EI_df = EI_df.reset_index(drop=True)
    EI_df.to_csv(output_folder_name + '/' + 'EI_' + max_TARGET[] + '.csv')
    
    #グラフで可視化とそのグラフの保存
    plot_samples(Xtrain, ytrain, Xsc_all[index_test,:], y_m, y_s)
    plt.savefig(output_folder_name + '/' + max_TARGET[2:] + '_fig.jpg')
    plt.show()
    
'''

#目的変数を最小化するベイズ最適化
for min_TARGET in min_TARGETs:
    
    #出力フォルダ作成
    output_folder_name = './outputs/EI_' + min_TARGET[:]
    os.makedirs(output_folder_name, exist_ok=True)
    
    #目的変数を設定
    ytrain = df_train[min_TARGET].values.copy()    
    
    #GPRに標準化とカーネルの選択を追加しgprとする
    gpr = make_pipeline(StandardScaler(), 
                            gp.GaussianProcessRegressor(kernel=sum_kernel, normalize_y=True))      
    #ベイズ最適化の実行
    gpr.fit(Xtrain, ytrain)
    
    #テストデータのEIの計算
    test_EI = [expected_improvement(Xsc_all[idx,:], gpr, ytrain, greater_is_better = False).reshape(-1,)for idx in index_test]
    test_EI = np.array(test_EI)
    test_EI = test_EI.reshape(-1,)
    
    #推定値と推定値の標準偏差が入ったデータフレームを作成
    y_m, y_s = gpr.predict(x_test, return_std=True)
    y_m = y_m.reshape(-1,)
    y_m_df = pd.Series(y_m, name='Estimated_value')
    y_s_df = pd.Series(y_s, name='std_of_Estimated_value')
    #マトリックスのEIをデータフレーム化し、推定値の平均値、標準偏差、実験パラメータ、SMILES列を追加。
    #その後、更新幅の期待値順にソートする   
    EI_df = pd.Series(test_EI, name='Expected_value_update_width') #データフレーム化
    EI_df = pd.concat([EI_df, y_m_df, y_s_df, df_smile], axis = 1) #EIにSMILES列(実験パラメータ列含む)を追加
    EI_df = EI_df.sort_values(by='Expected_value_update_width', ascending=False) #更新幅の期待値順にソート
    EI_df = EI_df.reset_index(drop=True)
    EI_df.to_csv(output_folder_name + '/' + 'EI_' + min_TARGET[:] + '.csv')
    
    #グラフで可視化とそのグラフの保存
    plot_samples(Xtrain, ytrain, Xsc_all[index_test,:], y_m, y_s)
    plt.savefig(output_folder_name + '/' + min_TARGET[:] + '_fig.jpg')
    plt.show()
    

'''
更新幅の期待値が大きい順に構造式の画像をpdf化-----------------------------------------
'''

pdf_file_path = './outputs'
pdf_file_name = 'Results_Compound.pdf'

#何枚の画像をpdf化するか
if all_img_pdf == True:
    num_img = len(EI_df)
else:
    num_img = num_img_pdf
img_list = []
for i in range(num_img):
    molecule = Chem.MolFromSmiles(EI_df.loc[i,'SMILES']) #SMILESからmolの読み込み
    img_file_name = str(i) +'.png'
    img_pil = Draw.MolToImage(molecule, size=(250, 250)) #molからPILイメージに変換
    img_numpy = np.asarray(img_pil)
    img_numpy_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGBA2BGR)
    cv2.putText(img_numpy_bgr, str(i), (3, 25), cv2.FONT_HERSHEY_PLAIN,
                1.5, (0, 0, 0), 1, cv2.LINE_AA) #読み込んだ画像にテキスト書き込み
    img_list.append(img_numpy_bgr)
   
img_to_pdf(img_list, pdf_file_path, pdf_file_name) #読み込んだ画像のpdf化


