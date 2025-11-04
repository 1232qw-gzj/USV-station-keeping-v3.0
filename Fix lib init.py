#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_lib_init.py - 自动修复 lib/__init__.py 导入错误

修复 ModuleNotFoundError: No module named 'python_vehicle_simulator.lib.mainLoop'

使用方法:
    python fix_lib_init.py
"""

import os
from pathlib import Path

# 修复后的内容
FIXED_CONTENT = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 只导入存在的模块
from .gnc import *
from .guidance import *
from .control import *

# 以下模块在当前版本中不存在，已注释
# from .mainLoop import *
# from .plotTimeSeries import *
# from .models import *
# from .actuator import *
"""


def main():
    print("=" * 80)
    print("修复 lib/__init__.py 导入错误")
    print("=" * 80)

    # 获取项目根目录
    script_dir = Path(__file__).parent.resolve()

    # 如果脚本在 outputs 目录，需要调整路径
    if script_dir.name == 'outputs':
        project_root = script_dir.parent
    else:
        project_root = script_dir

    # lib/__init__.py 的路径
    lib_init_path = project_root / 'src' / 'python_vehicle_simulator' / 'lib' / '__init__.py'

    print(f"\n项目根目录: {project_root}")
    print(f"目标文件: {lib_init_path}")

    # 检查文件是否存在
    if not lib_init_path.exists():
        print(f"\n❌ 错误: 找不到文件 {lib_init_path}")
        print("请确保脚本在项目根目录中运行")
        return

    # 备份原文件
    backup_path = lib_init_path.with_suffix('.py.backup')
    try:
        with open(lib_init_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)

        print(f"✅ 已备份原文件到: {backup_path}")
    except Exception as e:
        print(f"⚠️  备份失败: {e}")
        print("继续修复...")

    # 写入修复后的内容
    try:
        with open(lib_init_path, 'w', encoding='utf-8') as f:
            f.write(FIXED_CONTENT)

        print(f"✅ 已修复: {lib_init_path}")
        print("\n修改内容:")
        print("  - ✅ 保留: from .gnc import *")
        print("  - ✅ 保留: from .guidance import *")
        print("  - ✅ 保留: from .control import *")
        print("  - ❌ 注释: from .mainLoop import *")
        print("  - ❌ 注释: from .plotTimeSeries import *")
        print("  - ❌ 注释: from .models import *")
        print("  - ❌ 注释: from .actuator import *")

    except Exception as e:
        print(f"❌ 修复失败: {e}")
        return

    print("\n" + "=" * 80)
    print("修复完成！")
    print("=" * 80)
    print("\n现在可以运行程序了：")
    print(f"   python {project_root / 'src' / 'python_vehicle_simulator' / 'main_station_keeping.py'}")
    print("\n或验证导入：")
    print("   python -c \"from python_vehicle_simulator.lib import gnc; print('导入成功！')\"")


if __name__ == '__main__':
    main()