@echo off
setlocal
cd /d %~dp0

if not exist requirements.txt (
  echo [Loi] Khong tim thay requirements.txt trong thu muc hien tai.
  echo Hay dat file requirements.txt cung cap voi script vao cung thu muc nay.
  pause
  exit /b 1
)

if not exist data mkdir data

echo === Cai thu vien can thiet ===
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo [Loi] Cai thu vien that bai.
  pause
  exit /b 1
)

echo.
echo === Kiem tra dataset ===
for %%f in (data\*.csv) do set HASCSV=1
if not defined HASCSV (
  echo Khong tim thay file CSV trong thu muc data\
  echo.
  echo Cach 1: dung Kaggle API
  echo   kaggle datasets download -d patelris/crop-yield-prediction-dataset -p data --unzip
  echo.
  echo Cach 2: tai file CSV thu cong tu Kaggle roi copy vao thu muc data\
  echo.
  pause
  exit /b 1
)

echo.
echo === Chay phan tich ===
python crop_yield_pipeline.py

echo.
pause
