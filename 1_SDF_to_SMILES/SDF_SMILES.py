# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
import img2pdf
from PIL import Image
import cv2
import shutil


"""
SDF -> SMILES変換 ＆ 画像ファイル連結PDF化スクリプト
"""

# ============================
#  1. SDFからSMILESへ一括変換
# ============================

# ◆1-1. ファイルパスの設定
input_folder = './input'
output_folder = './output'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

output_smiles_wide = os.path.join(output_folder, 'SMILES_WIDE.csv')
output_smiles_long = os.path.join(output_folder, 'SMILES_LONG.csv')

# ◆1-2. inputフォルダ内のSDFファイルのみを取得
sdf_list = [f for f in os.listdir(input_folder) if f.endswith('.sdf')]
print("【SDFファイル一覧】", sdf_list)

# ◆1-3. RDKitでSDFを読み込み -> SMILESに変換
sdf_dict = {}
for sdf_file_name in sdf_list:
    sdf_path = os.path.join(input_folder, sdf_file_name)
    
    try:
        mol_supplier = Chem.SDMolSupplier(sdf_path, sanitize=True)
        if mol_supplier is None:
            print(f"読み込み失敗: {sdf_path}")
            continue
    except Exception as e:
        print(f"SDMolSupplier読み込み時にエラー: {sdf_path}\n{e}")
        continue
    
    sdf_name = sdf_file_name.split('.')[0]
    # ここで辞書に格納
    sdf_dict[sdf_name] = mol_supplier

# ◆1-4. SMILES_WIDEのDataFrame作成
for sdf_name, mol_supplier in sdf_dict.items():
    mol_list = []
    for mol in mol_supplier:
        if mol is None:
            continue
        smiles_str = Chem.MolToSmiles(mol)
        mol_list.append(smiles_str)
    sdf_dict[sdf_name] = mol_list

SMILES_WIDE = pd.DataFrame(sdf_dict.values(), index=sdf_dict.keys()).T
SMILES_WIDE.to_csv(output_smiles_wide, index=False)
print(f"SMILES_WIDE.csv 出力完了: {output_smiles_wide}")

# ◆1-5. SMILES_LONGのDataFrame作成
smiles_all = []
for sdf_name, smiles_list in sdf_dict.items():
    smiles_all.extend(smiles_list)

SMILES_LONG = pd.DataFrame({'SMILES': smiles_all})
SMILES_LONG.to_csv(output_smiles_long, index=False)
print(f"SMILES_LONG.csv 出力完了: {output_smiles_long}")


# =========================================
#  2. SDFから構造式の画像を一括保存・PDF化
# =========================================

# ◆2-1. SDFをDataFrameに読み込み
sdf_dict_img = {}
for sdf_file_name in sdf_list:
    sdf_path = os.path.join(input_folder, sdf_file_name)
    try:
        sdf_df = PandasTools.LoadSDF(sdf_path)
        sdf_name = sdf_file_name.split('.')[0]
        sdf_dict_img[sdf_name] = sdf_df
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {sdf_path}")
    except Exception as e:
        print(f"LoadSDFでエラー: {sdf_path}\n{e}")

# ◆2-2. SDF名ごとのフォルダを output フォルダ配下に作成
# 例: ./output/SDF名/ に画像を格納
folder_dict = {}
for k in sdf_dict_img.keys():
    sub_folder = os.path.join(output_folder, k)
    if not os.path.exists(sub_folder):
        os.mkdir(sub_folder)
    folder_dict[k] = sub_folder

# ◆2-3. 画像ファイルを出力

# SMILES文字列を基にRDKitのMolオブジェクトを生成し、不正な化合物はスキップする処理
name_error_dict = {}
num = 1

for i, k in enumerate(folder_dict.keys()):
    sdf_df = sdf_dict_img[k]
    for j in range(len(sdf_df)):
        if j < SMILES_WIDE.shape[0]:
            smiles_str = SMILES_WIDE.iloc[j, i]
        else:
            continue
        
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            continue
        
        # すべての化合物名を "Compound_j" に統一
        compound_name = f"Compound_{j}"
        
        out_path = os.path.join(folder_dict[k], compound_name + '.png')
        try:
            Draw.MolToFile(mol, out_path)
        except (FileNotFoundError, OSError):
            new_compound_name = f"Compound_{num}"
            name_error_dict[new_compound_name] = k + '_' + compound_name
            out_path = os.path.join(folder_dict[k], new_compound_name + '.png')
            Draw.MolToFile(mol, out_path)
            num += 1
            
# ◆2-4. 画像以外のファイルは削除 & 画像をリサイズ
for k, sub_folder in folder_dict.items():
    for file_name in os.listdir(sub_folder):
        if not file_name.endswith('.png'):
            os.remove(os.path.join(sub_folder, file_name))
    folder_png = os.listdir(sub_folder)
    
    # 大規模データセットの場合、リサイズのループがメモリ・時間を消費する場合があるため、
    # 必要に応じてバッチ処理や並列化を検討
    for png_file in folder_png:
        img_path = os.path.join(sub_folder, png_file)
        img = Image.open(img_path)
        img_resize = img.resize((120, 120), Image.LANCZOS)
        img_resize.save(img_path)
    print('resize completed :', k)

# ◆2-5. PDF出力用フォルダ
pdf_folder = os.path.join(output_folder, 'Compound_pdf')
if not os.path.exists(pdf_folder):
    os.mkdir(pdf_folder)

# ◆2-6. 画像の連結 -> PDF化
for k, sub_folder in folder_dict.items():
    folder_k = os.listdir(sub_folder)
    img_list = []
    
    # メモリ使用量に注意しながら読み込み
    for img_file in folder_k:
        img_path = os.path.join(sub_folder, img_file)
        img_cv = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_cv is not None:
            img_list.append(img_cv)
    
    # 20枚ごとに1枚の画像に連結
    mod = len(img_list) % 20
    if mod != 0:
        lack_img_num = 20 - mod
        # 白画像を追加して整形
        for _ in range(lack_img_num):
            img_list.append(np.ones((120, 120, 3), dtype=np.uint8) * 255)
    
    # 一時フォルダも output 以下に作成
    temp_folder = os.path.join(output_folder, 'Temp')
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)
    
    concat_idx = 0
    for i in range(0, len(img_list), 20):
        batch = img_list[i:i+20]
        row1 = cv2.hconcat(batch[0:5])
        row2 = cv2.hconcat(batch[5:10])
        row3 = cv2.hconcat(batch[10:15])
        row4 = cv2.hconcat(batch[15:20])
        im_v = cv2.vconcat([row1, row2, row3, row4])
        
        out_img_path = os.path.join(temp_folder, f"img_{concat_idx}.png")
        cv2.imwrite(out_img_path, im_v)
        concat_idx += 1
    
    # PDF作成
    pdf_file_name = os.path.join(pdf_folder, k + '.pdf')
    with open(pdf_file_name, "wb") as f:
        png_list = [os.path.join(temp_folder, j) for j in os.listdir(temp_folder) if j.endswith('.png')]
        png_list.sort()
        # 大規模データセットの場合、画像結合やPDF化でメモリ使用が増大しやすい
        f.write(img2pdf.convert([Image.open(png).filename for png in png_list]))
    
    # 一時フォルダを空に
    for png_file in os.listdir(temp_folder):
        os.remove(os.path.join(temp_folder, png_file))

print("すべての処理が完了しました。")
print("エラー化合物名の置換リスト:", name_error_dict)