# 必要なライブラリを読み込む
library(tidyverse)  # データ操作と可視化のための包括的なパッケージ群
library(readr)      # 高速で柔軟なデータ読み込みのためのパッケージ
library(stringi)    # 文字列処理のための高性能パッケージ
library(jsonlite)   # JSONデータの処理用
library(PerformanceAnalytics)  # chart.Correlation関数の提供パッケージ


# outputフォルダが存在しない場合は作成
# showWarnings = FALSEは、フォルダが既に存在する場合の警告を抑制する
dir.create("output", showWarnings = FALSE)

# 文字コードを自動検出してCSVファイルを読み込む関数
read_csv_auto_encoding <- function(file_path) {
  # ファイルの生データを読み込む
  raw_data <- read_file_raw(file_path)
  # stringiパッケージを使用して文字エンコーディングを検出
  encoding <- stri_enc_detect(raw_data)[[1]]$Encoding[1]
  # 検出されたエンコーディングを使用してCSVを読み込む
  data <- read_csv(file_path, locale = locale(encoding = encoding))
  return(data)
}

# CSVファイルを読み込み、処理する関数
read_and_process_csv <- function(file_path) {
  # 上で定義した関数を使用してCSVを読み込む
  data <- read_csv_auto_encoding(file_path)
  # データフレームに元のファイル名を属性として追加
  attr(data, "filename") <- basename(file_path)
  return(data)
}

# inputフォルダから全てのCSVファイルを読み込む
input_files <- list.files(path = "input", pattern = "*.csv", full.names = TRUE)
# purrr::mapを使用して各ファイルに読み込み関数を適用
data_list <- map(input_files, read_and_process_csv)

# ファイル名のリストを作成
file_names <- map_chr(data_list, ~attr(., "filename"))

# =============================
#   相関図のPNG出力のメイン処理
# =============================
print("相関図をPNGファイルで出力します...")

for (i in seq_along(data_list)) {
  # データを取得
  df <- data_list[[i]]
  
  # ファイル名から拡張子を除去してPNGファイル名を生成
  png_filename <- paste0(gsub("\\.csv$", "", file_names[i]), "_correlation.png")
  
  # 出力先をPNGに設定
  png(file.path("output", png_filename), width = 800, height = 800)
  
  # 2列目以降のみ抽出し、相関図を描画
  # histogram = TRUE, pch=19 などは好みに合わせて調整可能
  # 2列目以降に数値以外の列が含まれているとエラーになるため、適宜ご注意ください
  chart.Correlation(df[, -1], histogram = TRUE, pch = 19)
  
  # 出力デバイスを閉じる
  dev.off()
  
  message(paste("相関図の出力が完了:", png_filename))
}

print("すべての処理が完了しました")