@echo off
REM ========================================
REM Python Vehicle Simulator 完整修复脚本
REM 包含安装和lib/__init__.py修复
REM ========================================

echo ========================================
echo Python Vehicle Simulator 完整修复工具
echo ========================================
echo.

REM 检查是否在项目根目录
if not exist "setup.cfg" (
    echo [错误] 找不到 setup.cfg 文件！
    echo 请确保在项目根目录运行此脚本
    echo 当前目录: %CD%
    pause
    exit /b 1
)

echo [1/5] 检测项目结构...
if not exist "src\python_vehicle_simulator" (
    echo [错误] 找不到 src\python_vehicle_simulator 目录！
    pause
    exit /b 1
)
echo       ✓ 项目结构正确

echo.
echo [2/5] 检测Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 找不到Python！请先安装Python 3.10+
    pause
    exit /b 1
)
python --version
echo       ✓ Python已安装

echo.
echo [3/5] 安装项目（可编辑模式）...
echo 运行: pip install -e .
pip install -e .
if errorlevel 1 (
    echo [错误] 安装失败！
    pause
    exit /b 1
)
echo       ✓ 安装成功

echo.
echo [4/5] 修复 lib/__init__.py...

REM 备份原文件
if exist "src\python_vehicle_simulator\lib\__init__.py" (
    copy /Y "src\python_vehicle_simulator\lib\__init__.py" "src\python_vehicle_simulator\lib\__init__.py.backup" >nul 2>&1
    echo       ✓ 已备份原文件
)

REM 创建修复后的内容
(
echo #!/usr/bin/env python3
echo # -*- coding: utf-8 -*-
echo.
echo # 只导入存在的模块
echo from .gnc import *
echo from .guidance import *
echo from .control import *
echo.
echo # 以下模块在当前版本中不存在，已注释
echo # from .mainLoop import *
echo # from .plotTimeSeries import *
echo # from .models import *
echo # from .actuator import *
) > "src\python_vehicle_simulator\lib\__init__.py"

echo       ✓ 已修复 lib/__init__.py

echo.
echo [5/5] 验证修复...
python -c "from python_vehicle_simulator.lib import gnc, control, guidance; print('模块导入成功！')" >nul 2>&1
if errorlevel 1 (
    echo       ⚠ 导入验证失败，但可以尝试运行程序
) else (
    echo       ✓ 模块可以正常导入
)

echo.
echo ========================================
echo 修复完成！
echo ========================================
echo.
echo 现在可以运行程序：
echo    python src\python_vehicle_simulator\main_station_keeping.py
echo.
echo 如果还有问题，请查看 第二个错误修复说明.md
echo.
pause
