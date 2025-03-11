```batch
@echo off
set R_HOME=%~dp0R-Portable\App\R-Portable
set PATH=%R_HOME%\bin;%PATH%
Rscript.exe "%~dp0script.R"
pause
```