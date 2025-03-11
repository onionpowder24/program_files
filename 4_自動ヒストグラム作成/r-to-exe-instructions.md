# RコードをEXEファイルに変換する手順

1. R Portableのダウンロードとセットアップ:
   - [R Portable](https://sourceforge.net/projects/rportable/)をダウンロードします。
   - ダウンロードしたファイルを解凍し、適当なフォルダに配置します。

2. 必要なパッケージのインストール:
   - R Portableのフォルダ内にある「R-Portable.exe」を実行します。
   - 以下のコマンドを実行して、必要なパッケージをインストールします：
     ```r
     install.packages(c("tidyverse", "ggrepel", "readr", "stringi"))
     ```

3. Rスクリプトの準備:
   - 元のRコードを「script.R」という名前で保存します。
   - スクリプトの冒頭に以下の行を追加して、作業ディレクトリを設定します：
     ```r
     setwd(dirname(sys.frame(1)$ofile))
     ```

4. バッチファイルの作成:
   - 以下の内容で「run_script.bat」というバッチファイルを作成します：
     ```batch
     @echo off
     set R_HOME=%~dp0R-Portable\App\R-Portable
     set PATH=%R_HOME%\bin;%PATH%
     Rscript.exe "%~dp0script.R"
     pause
     ```

5. ディレクトリ構造の設定:
   - 以下のようなディレクトリ構造を作成します：
     ```
     MyRProject/
     ├── R-Portable/
     ├── input/
     ├── script.R
     └── run_script.bat
     ```

6. EXEファイルの作成:
   - [Bat To Exe Converter](https://bat-to-exe-converter-x64.en.softonic.com/)などのツールを使用して、「run_script.bat」をEXEファイルに変換します。

7. 配布とユーザーへの指示:
   - 作成したEXEファイル、R-Portableフォルダ、およびinputフォルダを含むパッケージを配布します。
   - ユーザーに以下の指示を提供します：
     1. パッケージを解凍する
     2. inputフォルダにCSVファイルを配置する
     3. EXEファイルをダブルクリックして実行する

注意: このアプローチはWindowsシステムでのみ機能します。他のOSの場合は異なるアプローチが必要になります。
