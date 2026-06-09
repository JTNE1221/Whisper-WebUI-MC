@echo off

if not exist "%~dp0\venv\Scripts" (
    echo Creating venv...
    python -m venv venv
)
echo checked the venv folder. now installing requirements..

:: 1. 激活虚拟环境
call "%~dp0\venv\scripts\activate"

:: 2. 升级基础工具并解决 setuptools 兼容性
python -m pip install -U pip
pip install "setuptools<70.0.0" wheel

:: ---------------------------------------------------------
:: 3. 拦截式安装：先在本地寻找并安装 3.5GB 的 Torch 巨无霸
:: ---------------------------------------------------------
echo Checking for local PyTorch package...
if exist "%~dp0\torch-2.8.0+cu128-cp312-cp312-win_amd64.whl" (
    echo Local PyTorch found! Installing from local file...
    pip install "%~dp0\torch-2.8.0+cu128-cp312-cp312-win_amd64.whl"
) else (
    echo WARNING: Local PyTorch file NOT found! The script will try to download it online.
    echo Make sure your proxy is stable.
)

:: ---------------------------------------------------------
:: 4. 安装剩余的依赖包
:: ---------------------------------------------------------
echo Installing other dependencies...
pip install --no-build-isolation -r requirements.txt --proxy http://127.0.0.1:10808 --default-timeout=600

if errorlevel 1 (
    echo.
    echo Requirements installation failed. please remove venv folder and run install.bat again.
) else (
    echo.
    echo Requirements installed successfully.
)
pause
