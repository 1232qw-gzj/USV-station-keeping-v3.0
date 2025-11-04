#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
otter_station_keeping.py - Otter无人船扩展类（优化版）

添加LOS制导和定点保持控制功能

改进:
- 使用配置管理器，消除硬编码
- 使用logging替代print
- 统一属性命名
- 支持RK4积分（可选）

作者: [您的名字]
日期: 2025-11-04
"""

import numpy as np
import logging
from typing import Optional, List
# ✅ 正确：只导入父类 otter
from .otter import otter

# 获取模块日志器
logger = logging.getLogger(__name__)


class OtterStationKeeping(otter):
    """
    Otter无人船扩展类 - 添加制导-镇定切换控制（优化版）

    继承自 otter 类，新增：
    - LOS制导算法
    - 自适应镇定控制
    - 模式切换逻辑
    - 配置化参数管理
    - 日志系统
    """

    def __init__(self,
                 config=None,
                 control_system: str = "LOS_STATION_KEEPING",
                 waypoints: Optional[List[List[float]]] = None,
                 **kwargs):
        """
        初始化扩展类

        参数：
            config: ConfigManager配置对象（推荐）
            control_system: 控制系统类型
            waypoints: 航点列表 [[x1,y1], [x2,y2], ...] （会覆盖config中的配置）
            **kwargs: 传递给父类的其他参数
        """

        # ========================================
        # 1. 加载配置
        # ========================================
        if config is not None:
            self.config = config
        else:
            # 如果没有提供配置，使用默认值
            from config_manager import ConfigManager
            self.config = ConfigManager()
            logger.warning("未提供配置对象，使用默认配置")

        # ========================================
        # 2. 调用父类初始化
        # ========================================
        # 从配置获取父类需要的参数
        parent_kwargs = {
            'controlSystem': control_system,
            'V_current': self.config.get('environment', 'current_speed', default=0.3),
            'beta_current': self.config.get('environment', 'current_direction', default=30),
            'tau_X': self.config.get('vehicle', 'tau_x', default=120)
        }

        # ⚠️ 关键修复：从kwargs中移除config参数，避免传递给父类
        kwargs_for_parent = {k: v for k, v in kwargs.items() if k != 'config'}
        parent_kwargs.update(kwargs_for_parent)  # 允许其他kwargs覆盖

        super().__init__(**parent_kwargs)

        # ✅ 统一属性命名（同时保持对父类的兼容性）
        self._control_system = control_system
        self.controlsystem = control_system  # 兼容父类可能的命名
        self.controlSystem = control_system  # 兼容父类可能的命名

        logger.info("=" * 60)
        logger.info("🚢 OtterStationKeeping 初始化")
        logger.info("=" * 60)

        # ========================================
        # 3. 航点配置
        # ========================================
        if waypoints is not None:
            self.waypoints = waypoints
        else:
            self.waypoints = self.config.get_waypoints()

        self.current_wp = 0
        self.arrival_radius = self.config.get('mission', 'arrival_radius', default=5.0)

        # ========================================
        # 4. LOS制导参数
        # ========================================
        self.delta = self.config.get('los_guidance', 'delta', default=10.0)
        r_los = self.config.get('los_guidance', 'r_los', default=0)
        self.los_psi_d = r_los * np.pi / 180

        # LOS控制增益
        self.k_psi_guidance = self.config.get('los_guidance', 'k_psi', default=100.0)
        self.k_r_guidance = self.config.get('los_guidance', 'k_r', default=50.0)

        # ========================================
        # 5. 镇定控制参数
        # ========================================
        self.station_duration = self.config.get('mission', 'station_duration', default=25.0)
        self.station_radius = self.config.get('mission', 'station_radius', default=2.5)
        self.station_timer = 0.0
        self.station_point = None

        # 自适应控制参数（从配置加载）
        self.k_p = self.config.get_k_p_matrix()
        self.k_d = self.config.get_k_d_matrix()
        self.gamma = self.config.get_gamma_matrix()
        self.theta_hat = np.zeros(3)
        self.theta_max = self.config.get('station_keeping', 'theta_max', default=100.0)

        # ========================================
        # 6. 紧急制导参数
        # ========================================
        self.k_psi_emergency = self.config.get('emergency_guidance', 'k_psi', default=120.0)
        self.k_r_emergency = self.config.get('emergency_guidance', 'k_r', default=60.0)
        self.thrust_multiplier = self.config.get('emergency_guidance', 'thrust_multiplier', default=1.2)
        self.return_distance_ratio = self.config.get('emergency_guidance', 'return_distance_ratio', default=0.8)

        # ========================================
        # 7. 控制模式
        # ========================================
        self.control_mode = "GUIDANCE"
        self.mode_history = []
        self.call_count = 0

        # ✅ 使用 simTime 而不是 sim_time（匹配父类）
        self.simTime = 0.0

        # ========================================
        # 8. 日志打印间隔（从配置加载）
        # ========================================
        self.print_interval = {
            'guidance': self.config.get('logging', 'print_interval', 'guidance', default=100),
            'station_keeping': self.config.get('logging', 'print_interval', 'station_keeping', default=100),
            'emergency': self.config.get('logging', 'print_interval', 'emergency', default=50)
        }

        # ========================================
        # 9. 打印初始化信息
        # ========================================
        logger.info("")
        logger.info("📍 任务配置:")
        logger.info(f"   航点数量: {len(self.waypoints)}")
        logger.info(f"   镇定时长: {self.station_duration}秒/点")
        logger.info(f"   误差圆半径: {self.station_radius}米")
        logger.info(f"   到达半径: {self.arrival_radius}米")
        logger.info("")
        logger.info("⚙️  控制参数:")
        logger.info(f"   LOS前视距离: {self.delta}米")
        logger.info(f"   制导增益: K_psi={self.k_psi_guidance}, K_r={self.k_r_guidance}")
        logger.info(f"   镇定增益: K_p=diag{tuple(np.diag(self.k_p))}")
        logger.info(f"   速度阻尼: K_d=diag{tuple(np.diag(self.k_d))}")
        logger.info(f"   自适应率: Γ=diag{tuple(np.diag(self.gamma))}")
        logger.info("=" * 60)
        logger.info("")

    def los_guidance(self, eta, nu, sample_time):
        """
        LOS制导算法

        参数:
            eta: 位置/姿态 [x, y, z, φ, θ, ψ]
            nu: 速度 [u, v, w, p, q, r]
            sample_time: 采样时间

        返回:
            u_control: 控制输入 [n1, n2]
        """
        x = eta[0]
        y = eta[1]
        psi = eta[5]

        if self.current_wp >= len(self.waypoints) - 1:
            return np.array([0, 0], float)

        # 当前路径段
        wp_current = self.waypoints[self.current_wp]
        wp_next = self.waypoints[self.current_wp + 1]

        dx = wp_next[0] - wp_current[0]
        dy = wp_next[1] - wp_current[1]
        path_length = np.sqrt(dx ** 2 + dy ** 2)

        if path_length < 0.1:
            self.current_wp += 1
            return self.los_guidance(eta, nu, sample_time)

        # 路径方向角
        alpha_path = np.arctan2(dy, dx)

        # 横向偏差
        dx_ship = x - wp_current[0]
        dy_ship = y - wp_current[1]
        cross_track_error = -dx_ship * np.sin(alpha_path) + dy_ship * np.cos(alpha_path)

        # LOS期望航向
        los_angle = np.arctan2(-cross_track_error, self.delta)
        self.los_psi_d = alpha_path + los_angle
        self.los_psi_d = np.arctan2(np.sin(self.los_psi_d), np.cos(self.los_psi_d))

        # 到达判定
        dist_to_next = np.sqrt((x - wp_next[0]) ** 2 + (y - wp_next[1]) ** 2)

        if dist_to_next < self.arrival_radius:
            logger.info(f"✅ 到达航点 {self.current_wp + 1}")
            self.current_wp += 1

            if self.current_wp < len(self.waypoints) - 1:
                self._switch_to_station_keeping(eta)
                return self.adaptive_station_keeping(eta, nu, sample_time)
            else:
                logger.info("🎉 完成所有航点！")
                return np.array([0, 0], float)

        # 航向控制（PD控制器）
        e_psi = self.los_psi_d - psi
        e_psi = np.arctan2(np.sin(e_psi), np.cos(e_psi))

        r_d = self.wn * e_psi
        r = nu[5]
        e_r = r_d - r

        tau_N = self.k_psi_guidance * e_psi + self.k_r_guidance * e_r
        tau_X = self.tauX

        [n1, n2] = self.controlAllocation(tau_X, tau_N)
        u_control = np.array([n1, n2], float)

        self.call_count += 1
        if self.call_count % self.print_interval['guidance'] == 0:
            logger.debug(
                f"🔵 制导 | 位置:({x:5.1f},{y:5.1f}) | "
                f"航向:{psi * 180 / np.pi:5.1f}° | "
                f"距WP{self.current_wp + 1}:{dist_to_next:5.1f}m | "
                f"横偏:{cross_track_error:5.2f}m"
            )

        return u_control

    def adaptive_station_keeping(self, eta, nu, sample_time):
        """
        自适应镇定控制

        参数:
            eta: 位置/姿态
            nu: 速度
            sample_time: 采样时间

        返回:
            u_control: 控制输入 [n1, n2]
        """
        x = eta[0]
        y = eta[1]
        psi = eta[5]
        u = nu[0]
        v = nu[1]
        r = nu[5]

        if self.station_point is None:
            return np.array([0, 0], float)

        # 位置误差
        e_x = x - self.station_point[0]
        e_y = y - self.station_point[1]
        e_psi = psi - self.los_psi_d
        e_psi = np.arctan2(np.sin(e_psi), np.cos(e_psi))

        position_error = np.sqrt(e_x ** 2 + e_y ** 2)

        # 超出误差圆检查
        if position_error > self.station_radius:
            logger.warning(
                f"⚠️  偏差{position_error:.2f}m > 阈值{self.station_radius:.2f}m，"
                f"切换到紧急制导"
            )
            self.control_mode = "EMERGENCY_GUIDANCE"
            self.mode_history.append(("EMERGENCY_GUIDANCE", self.simTime))
            return self._emergency_guidance(eta, nu, sample_time)

        # 坐标变换到船体坐标系
        R = np.array([
            [np.cos(psi), np.sin(psi), 0],
            [-np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        e_eta = np.array([e_x, e_y, e_psi])
        e_body = R @ e_eta
        nu_simple = np.array([u, v, r])

        # 自适应PD控制律
        tau_p = -self.k_p @ e_body
        tau_d = -self.k_d @ nu_simple
        tau_adaptive = -self.theta_hat

        tau_body = tau_p + tau_d + tau_adaptive

        # 自适应律（积分型）
        theta_dot = self.gamma @ e_body
        self.theta_hat = self.theta_hat + theta_dot * sample_time
        self.theta_hat = np.clip(self.theta_hat, -self.theta_max, self.theta_max)

        # 控制分配
        tau_X = tau_body[0]
        tau_N = tau_body[2]

        [n1, n2] = self.controlAllocation(tau_X, tau_N)
        u_control = np.array([n1, n2], float)

        # 更新计时器
        self.station_timer += sample_time

        if self.station_timer >= self.station_duration:
            logger.info(f"✅ 镇定完成（{self.station_timer:.1f}秒）")
            self.control_mode = "GUIDANCE"
            self.mode_history.append(("GUIDANCE", self.simTime))

        self.call_count += 1
        if self.call_count % self.print_interval['station_keeping'] == 0:
            logger.debug(
                f"🟢 镇定 | 误差:{position_error:4.2f}m | "
                f"时间:{self.station_timer:4.1f}/{self.station_duration:.0f}s | "
                f"θ̂:[{self.theta_hat[0]:.1f},{self.theta_hat[1]:.1f},{self.theta_hat[2]:.1f}]"
            )

        return u_control

    def _emergency_guidance(self, eta, nu, sample_time):
        """
        紧急制导（返回镇定区域）

        参数:
            eta: 位置/姿态
            nu: 速度
            sample_time: 采样时间

        返回:
            u_control: 控制输入 [n1, n2]
        """
        x = eta[0]
        y = eta[1]
        psi = eta[5]

        dx = self.station_point[0] - x
        dy = self.station_point[1] - y
        psi_d = np.arctan2(dy, dx)
        dist = np.sqrt(dx ** 2 + dy ** 2)

        # 返回阈值
        return_threshold = self.station_radius * self.return_distance_ratio

        if dist < return_threshold:
            logger.info(f"✅ 返回镇定区域（距离{dist:.2f}m）")
            self.control_mode = "STATION_KEEPING"
            self.mode_history.append(("STATION_KEEPING", self.simTime))
            return self.adaptive_station_keeping(eta, nu, sample_time)

        e_psi = psi_d - psi
        e_psi = np.arctan2(np.sin(e_psi), np.cos(e_psi))

        r = nu[5]
        tau_N = self.k_psi_emergency * e_psi - self.k_r_emergency * r
        tau_X = self.tauX * self.thrust_multiplier

        [n1, n2] = self.controlAllocation(tau_X, tau_N)

        self.call_count += 1
        if self.call_count % self.print_interval['emergency'] == 0:
            logger.warning(f"🔴 紧急 | 距离:{dist:5.2f}m | 航向误差:{e_psi * 180 / np.pi:5.1f}°")

        return np.array([n1, n2], float)

    def _switch_to_station_keeping(self, eta):
        """切换到镇定模式"""
        self.control_mode = "STATION_KEEPING"
        self.station_point = [eta[0], eta[1]]
        self.station_timer = 0.0
        self.theta_hat = np.zeros(3)
        self.mode_history.append(("STATION_KEEPING", self.simTime))

        logger.info("🔄 切换到镇定模式")
        logger.info(f"   目标点: ({eta[0]:.2f}, {eta[1]:.2f})")

    def headingAutopilot(self, eta, nu, sample_time):
        """
        控制器入口（重写父类方法）

        根据当前控制模式分发到相应的控制器
        """
        # 如果不是新控制模式，使用原始控制器
        if self._control_system != "LOS_STATION_KEEPING":
            return super().headingAutopilot(eta, nu, sample_time)

        # 使用新控制器
        if self.control_mode == "GUIDANCE":
            return self.los_guidance(eta, nu, sample_time)
        elif self.control_mode == "STATION_KEEPING":
            return self.adaptive_station_keeping(eta, nu, sample_time)
        elif self.control_mode == "EMERGENCY_GUIDANCE":
            return self._emergency_guidance(eta, nu, sample_time)
        else:
            # 默认使用父类控制器
            return super().headingAutopilot(eta, nu, sample_time)


if __name__ == "__main__":
    # 测试代码
    import sys

    sys.path.insert(0, '.')
    from config_manager import ConfigManager, setup_logging

    print("\n" + "=" * 60)
    print("测试 OtterStationKeeping (优化版)")
    print("=" * 60 + "\n")

    # 创建配置
    config = ConfigManager('config.yaml')
    setup_logging(config)

    # 创建船舶对象
    ship = OtterStationKeeping(config=config)

    logger.info("✅ 对象创建成功")
    logger.info(f"   继承自: {ship.__class__.__bases__}")
    logger.info(f"   控制系统: {ship._control_system}")
    logger.info(f"   航点数量: {len(ship.waypoints)}")