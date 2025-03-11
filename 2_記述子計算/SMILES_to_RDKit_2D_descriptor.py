# -*- coding: utf-8 -*-

import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


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
smiles = dataset.iloc[:, 0]  # 分子の SMILES
y = dataset.iloc[:, 1]  # 物性・活性などの目的変数

# 計算する記述子名の取得
descriptor_names = []
for descriptor_information in Descriptors.descList:
    descriptor_names.append(descriptor_information[0])
print('計算する記述子の数 :', len(descriptor_names))

# 記述子の計算
descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
descriptors = []  # ここに計算された記述子の値を追加
print('分子の数 :', len(smiles))
for index, smiles_i in enumerate(smiles):
    print(index + 1, '/', len(smiles))
    molecule = Chem.MolFromSmiles(smiles_i)
    descriptors.append(descriptor_calculator.CalcDescriptors(molecule))
descriptors = pd.DataFrame(descriptors, index=dataset.index, columns=descriptor_names)

# 保存
descriptors_with_y = pd.concat([y, descriptors], axis=1)  # y と記述子を結合
descriptors_with_y.to_csv('descriptors_with_y.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
