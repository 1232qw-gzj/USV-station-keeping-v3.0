#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config_manager.py - 配置文件管理器

统一管理所有配置参数，避免硬编码
支持YAML配置文件加载

作者: [您的名字]
日期: 2025-11-04
"""

import yaml
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List


class ConfigManager:
    """
    配置管理器类
    
    从YAML文件加载配置，提供统一的参数访问接口
    """
    
    # 默认配置（如果没有配置文件时使用）
    DEFAULT_CONFIG = {
        'mission': {
            'waypoints': [[0, 0], [40, 0], [40, 30], [0, 30], [0, 0]],
            'arrival_radius': 5.0,
            'station_duration': 25.0,
            'station_radius': 2.5
        },
        'los_guidance': {
            'delta': 10.0,
            'r_los': 0,
            'k_psi': 100.0,
            'k_r': 50.0
        },
        'station_keeping': {
            'k_p': {'surge': 80.0, 'sway': 80.0, 'yaw': 30.0},
            'k_d': {'surge': 40.0, 'sway': 40.0, 'yaw': 20.0},
            'gamma': {'surge': 0.15, 'sway': 0.15, 'yaw': 0.08},
            'theta_max': 100.0
        },
        'emergency_guidance': {
            'k_psi': 120.0,
            'k_r': 60.0,
            'thrust_multiplier': 1.2,
            'return_distance_ratio': 0.8
        },
        'environment': {
            'current_speed': 0.3,
            'current_direction': 30
        },
        'vehicle': {
            'tau_x': 120,
            'wn': 1.5,
            'control_system': 'LOS_STATION_KEEPING'
        },
        'simulation': {
            'total_time': 300,
            'dt': 0.02,
            'integration_method': 'euler',
            'skip_frames': 20
        },
        'logging': {
            'level': 'INFO',
            'to_file': True,
            'filename': 'usv_simulation.log',
            'to_console': True,
            'print_interval': {
                'guidance': 100,
                'station_keeping': 100,
                'emergency': 50
            }
        },
        'output': {
            'save_matlab': True,
            'matlab_filename': 'simulation_data.mat',
            'save_animation': False,
            'animation_filename': 'usv_animation.gif',
            'generate_report': True
        },
        'visualization': {
            'figure_size': [18, 10],
            'ship_length': 4.0,
            'ship_width': 1.5,
            'waypoint_marker_size': 14,
            'update_interval': 0.1
        }
    }
    
    def __init__(self, config_path: str = None):
        """
        初始化配置管理器
        
        参数:
            config_path: 配置文件路径（可选）
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logging.info(f"✅ 配置文件加载成功: {self.config_path}")
                return self._merge_config(self.DEFAULT_CONFIG, config)
            except Exception as e:
                logging.warning(f"⚠️  配置文件加载失败: {e}，使用默认配置")
                return self.DEFAULT_CONFIG.copy()
        else:
            logging.info("ℹ️  未指定配置文件，使用默认配置")
            return self.DEFAULT_CONFIG.copy()
    
    def _merge_config(self, default: Dict, custom: Dict) -> Dict:
        """合并默认配置和自定义配置"""
        result = default.copy()
        for key, value in custom.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, *keys, default=None):
        """
        获取配置值
        
        示例:
            config.get('mission', 'waypoints')
            config.get('los_guidance', 'delta')
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def get_waypoints(self) -> List[List[float]]:
        """获取航点列表"""
        return self.get('mission', 'waypoints')
    
    def get_k_p_matrix(self) -> np.ndarray:
        """获取位置反馈增益矩阵"""
        k_p = self.get('station_keeping', 'k_p')
        return np.diag([k_p['surge'], k_p['sway'], k_p['yaw']])
    
    def get_k_d_matrix(self) -> np.ndarray:
        """获取速度阻尼增益矩阵"""
        k_d = self.get('station_keeping', 'k_d')
        return np.diag([k_d['surge'], k_d['sway'], k_d['yaw']])
    
    def get_gamma_matrix(self) -> np.ndarray:
        """获取自适应学习率矩阵"""
        gamma = self.get('station_keeping', 'gamma')
        return np.diag([gamma['surge'], gamma['sway'], gamma['yaw']])
    
    def print_summary(self):
        """打印配置摘要"""
        print("\n" + "=" * 70)
        print(" " * 20 + "📋 配置摘要")
        print("=" * 70)
        
        print("\n🎯 任务配置:")
        print(f"   航点数量: {len(self.get_waypoints())}")
        print(f"   镇定时长: {self.get('mission', 'station_duration')} 秒")
        print(f"   误差半径: {self.get('mission', 'station_radius')} m")
        
        print("\n⚙️  控制参数:")
        print(f"   LOS前视距离: {self.get('los_guidance', 'delta')} m")
        print(f"   基础推力: {self.get('vehicle', 'tau_x')} N")
        
        print("\n🌊 环境参数:")
        print(f"   洋流速度: {self.get('environment', 'current_speed')} m/s")
        print(f"   洋流方向: {self.get('environment', 'current_direction')}°")
        
        print("\n🎮 仿真参数:")
        print(f"   总时长: {self.get('simulation', 'total_time')} 秒")
        print(f"   时间步长: {self.get('simulation', 'dt')} 秒")
        print(f"   加速倍数: {self.get('simulation', 'skip_frames')}x")
        print(f"   积分方法: {self.get('simulation', 'integration_method')}")
        
        print("=" * 70 + "\n")


def setup_logging(config: ConfigManager):
    """
    配置日志系统
    
    参数:
        config: 配置管理器实例
    """
    log_level = getattr(logging, config.get('logging', 'level', default='INFO'))
    log_filename = config.get('logging', 'filename', default='usv_simulation.log')
    to_file = config.get('logging', 'to_file', default=True)
    to_console = config.get('logging', 'to_console', default=True)
    
    # 创建日志格式
    log_format = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置根日志器
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 添加文件处理器
    if to_file:
        file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        logger.addHandler(file_handler)
    
    # 添加控制台处理器
    if to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format, date_format))
        logger.addHandler(console_handler)
    
    logging.info("=" * 70)
    logging.info("🚀 USV Station Keeping System - Logging Started")
    logging.info("=" * 70)


if __name__ == "__main__":
    # 测试配置管理器
    print("\n" + "=" * 70)
    print("测试配置管理器")
    print("=" * 70 + "\n")
    
    # 使用默认配置
    config = ConfigManager()
    config.print_summary()
    
    # 测试参数获取
    print("测试参数获取:")
    print(f"  Delta: {config.get('los_guidance', 'delta')}")
    print(f"  K_p 矩阵: \n{config.get_k_p_matrix()}")
    print(f"  航点: {config.get_waypoints()}")
