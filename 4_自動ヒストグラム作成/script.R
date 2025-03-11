# 必要なライブラリを読み込む
library(tidyverse)  # データ操作と可視化のための包括的なパッケージ群
library(readr)      # 高速で柔軟なデータ読み込みのためのパッケージ
library(stringi)    # 文字列処理のための高性能パッケージ
library(jsonlite)   # JSONデータの処理用


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

# スタージェスの公式でビン数を計算する関数
# この公式は、ヒストグラムの適切なビン数を推定するための一般的な方法
calculate_sturges_bins <- function(x) {
  n <- length(x)
  ceiling(log2(n) + 1)
}

# 静的なヒストグラムをPDFに保存
print("ヒストグラムをPDFに保存中...")
pdf_filename <- paste0(gsub("\\.csv$", "", file_names[1]), "_csv_ヒストグラム.pdf")
pdf(file.path("output", pdf_filename), width = 11, height = 8, family = "Japan1")

# 各データフレームの各列に対してヒストグラムを生成
for (data in data_list) {
  for (var in names(data)) {
    tryCatch({
      x <- data[[var]]
      if (is.null(x)) {
        # 列が見つからない場合のエラーメッセージを表示
        print(ggplot() + 
                annotate("text", x = 0.5, y = 0.5, label = paste(var, "列が見つかりません")) +
                theme_void())
        next
      }
      
      # データの基本統計量を計算
      na_count <- sum(is.na(x))
      inf_count <- sum(!is.finite(x) & !is.na(x))
      valid_count <- sum(!is.na(x) & is.finite(x)) #有効値のカウント
      unique_count <- length(unique(x[!is.na(x) & is.finite(x)])) #ユニークな値のカウント
      
      if (valid_count < 2) {
        # 有効なデータが不足している場合のエラーメッセージを表示
        print(ggplot() + 
                annotate("text", x = 0.5, y = 0.5, 
                         label = paste(var, "\n有効なデータが不足しています\n",
                                       "欠損値:", na_count, 
                                       "無限大/NaN(非数):", inf_count,
                                       "有効値:", valid_count)) +
                theme_void())
        next
      }
      
      # 有効なデータのみを抽出
      x_valid <- x[!is.na(x) & is.finite(x)]
      
      if (unique_count <= 10 || is.character(x) || is.factor(x)) {
        # ユニークな値が10以下または文字列/因子型の場合は棒グラフを使用
        df <- data.frame(x = factor(x_valid))
        p <- ggplot(df, aes(x = x)) +
          geom_bar(fill = "skyblue", color = "black") +
          labs(x = var, y = "頻度", 
               title = paste(var, "のヒストグラム"),
               caption = paste("参照CSVファイル:", paste(file_names, collapse = ", "))) +
          theme(axis.text.x = element_text(angle = 45, hjust = 1))  # x軸ラベルを斜めに
      } else {
        # それ以外はヒストグラムを使用
        bins <- min(calculate_sturges_bins(x_valid), unique_count)
        
        p <- ggplot(data.frame(x = x_valid), aes(x = x)) +
          geom_histogram(bins = bins, fill = "skyblue", color = "black") +
          labs(x = var, y = "頻度", 
               title = paste(var, "のヒストグラム"),
               caption = paste("参照CSVファイル:", paste(file_names, collapse = ", ")))
      }
      
      # グラフの見た目を調整
      p <- p + theme_grey() +
        theme(
          plot.title = element_text(hjust = 0.5, size = 20),
          plot.subtitle = element_text(hjust = 0.5, size = 16),
          axis.title = element_text(size = 16),
          axis.text = element_text(size = 14),
          plot.caption = element_text(hjust = 1, size = 12)
        )
      
      # 欠損値や無限大がある場合、それらの情報をサブタイトルに追加
      if (na_count > 0 || inf_count > 0) {
        p <- p + labs(subtitle = paste("欠損値:", na_count, 
                                       "無限大/NaN(非数):", inf_count,
                                       "有効値:", valid_count))
      }
      
      # グラフをPDFに出力
      print(p)
    }, error = function(e) {
      # エラーが発生した場合、エラーメッセージを表示
      print(ggplot() + 
              annotate("text", x = 0.5, y = 0.5, label = paste("エラー:", e$message)) +
              theme_void())
    })
  }
}

dev.off()
print("ヒストグラムのPDF保存が完了しました")






# データをJSON形式に変換する関数を定義
convert_data_to_json <- function(data_list) {
  # データリストの各要素（データフレーム）に対して処理を行う
  json_data <- lapply(data_list, function(df) {
    # データフレームの各列に対して処理を行う
    lapply(df, function(col) {
      if(is.numeric(col)) {
        # 数値型の列の場合、有限の値のみを抽出
        col[is.finite(col)]
      } else {
        # 非数値型の列の場合、文字列に変換
        as.character(col)
      }
    })
  })
  # 処理したデータをJSON形式に変換（自動的に単一要素をアンボックス）
  toJSON(json_data, auto_unbox = TRUE)
}

# 対話的なヒストグラムを表示するHTMLを生成する関数を定義
create_interactive_histogram_html <- function(data_list, file_names) {
  # データリストをJSON形式に変換
  json_data <- convert_data_to_json(data_list)
  
  # HTMLコンテンツを文字列として生成
  html_content <- sprintf('
  <!DOCTYPE html>
  <html>
  <head>
    <title>動的ヒストグラム</title>
    <!-- Plotly.jsライブラリを読み込み -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
      /* ページ全体のスタイル設定 */
      body { font-family: Arial, sans-serif; }
      /* プロット領域のサイズ設定 */
      #plot { width: 100%%; height: 600px; }
      /* コントロールパネルの下部マージン設定 */
      .control-panel { margin-bottom: 20px; }
      /* チェックボックスリストのレイアウト設定 */
      .checkbox-list { 
        display: flex; 
        flex-wrap: wrap; 
        max-height: 200px; 
        overflow-y: auto; 
      }
      /* 各チェックボックス項目のスタイル設定 */
      .checkbox-item { 
        width: 200px; 
        margin-right: 10px; 
      }
    </style>
  </head>
  <body>
    <h1>動的ヒストグラム分析</h1>
    <div class="control-panel">
      <!-- チェックボックスを配置する領域 -->
      <div id="var-checkboxes" class="checkbox-list"></div>
      <!-- プロット更新ボタン -->
      <button onclick="updatePlot()">プロット更新</button>
    </div>
    <!-- プロットを表示する領域 -->
    <div id="plot"></div>
    <script>
      // JSONデータをJavaScriptオブジェクトに変換
      const data = %s;
      // チェックボックスを配置するコンテナ要素を取得
      const checkboxContainer = document.getElementById("var-checkboxes");
      
      // データの全変数に対してチェックボックスを作成
      const allVars = Object.keys(data[0]);
      allVars.forEach(v => {
        // 各変数に対してチェックボックスアイテムを作成
        const checkboxItem = document.createElement("div");
        checkboxItem.className = "checkbox-item";
        // チェックボックス要素を作成
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.id = v;
        checkbox.value = v;
        // ラベル要素を作成
        const label = document.createElement("label");
        label.htmlFor = v;
        label.appendChild(document.createTextNode(v));
        // チェックボックスとラベルをアイテムに追加
        checkboxItem.appendChild(checkbox);
        checkboxItem.appendChild(label);
        // アイテムをコンテナに追加
        checkboxContainer.appendChild(checkboxItem);
      });
      
      // プロットを更新する関数
      function updatePlot() {
        // チェックされた変数を取得
        const selectedVars = Array.from(document.querySelectorAll("input[type=checkbox]:checked")).map(cb => cb.value);
        // 選択された各変数に対してトレースを作成
        const traces = selectedVars.map(v => {
          // 全データセットから現在の変数の値を取得
          const allValues = [].concat(...Object.values(data).map(d => d[v]));
          // 数値データのみをフィルタリング
          const numericValues = allValues.filter(val => typeof val === "number" && isFinite(val));
          
          if (numericValues.length > 0) {
            // 数値データの場合はヒストグラムを作成
            return {
              x: numericValues,
              type: "histogram",
              name: v,
              opacity: 0.7
            };
          } else {
            // カテゴリカルデータの場合は棒グラフを作成
            const valueCounts = allValues.reduce((acc, val) => {
              acc[val] = (acc[val] || 0) + 1;
              return acc;
            }, {});
            return {
              x: Object.keys(valueCounts),
              y: Object.values(valueCounts),
              type: "bar",
              name: v
            };
          }
        });
        
        // Plotlyを使用してグラフを描画
        Plotly.newPlot("plot", traces, {
          title: "選択された変数のヒストグラム",
          barmode: "overlay",
          xaxis: { title: "値" },
          yaxis: { title: "頻度" }
        });
      }
      
      // 初期プロット（最初の変数を選択）
      document.querySelector("input[type=checkbox]").checked = true;
      updatePlot();
    </script>
  </body>
  </html>
  ', json_data)
  
  # 生成したHTMLコンテンツを返す
  return(html_content)
}

# HTMLファイルを保存する処理を開始
print("動的ヒストグラムをhtmlで保存中...")
# 対話的ヒストグラムのHTMLコンテンツを生成
html_content <- create_interactive_histogram_html(data_list, file_names)
# 出力ファイル名を生成（最初のCSVファイル名をベースに）
html_filename <- paste0(gsub("\\.csv$", "", file_names[1]), "_csv_動的ヒストグラム.html")
# HTMLコンテンツをファイルに書き込み
writeLines(html_content, file.path("output", html_filename))
# 保存完了メッセージを表示
print("動的ヒストグラムをhtmlで保存完了しました")

# 全処理完了メッセージを表示
print("すべての処理が完了しました")
