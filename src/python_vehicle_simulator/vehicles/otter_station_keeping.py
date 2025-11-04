#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
otter_station_keeping.py - Otter无人船扩展类

添加LOS制导和定点保持控制功能

作者: [您的名字]
日期: 2025-11-03
"""

import numpy as np
# ✅ 正确：只导入父类 otter
from .otter import otter


class OtterStationKeeping(otter):
    """
    Otter无人船扩展类 - 添加制导-镇定切换控制

    继承自 otter 类，新增：
    - LOS制导算法
    - 自适应镇定控制
    - 模式切换逻辑
    """

    def __init__(self,
                 controlSystem="LOS_STATION_KEEPING",
                 r_los=0,
                 waypoints=None,
                 station_duration=20.0,
                 station_radius=2.5,
                 **kwargs):
        """
        初始化扩展类

        参数：
            controlSystem: 控制系统类型
            waypoints: 航点列表 [[x1,y1], [x2,y2], ...]
            station_duration: 定点保持时长(秒)
            station_radius: 误差圆半径(米)
            **kwargs: 传递给父类的其他参数
        """

        # ✅ 调用父类初始化
        super().__init__(controlSystem=controlSystem, **kwargs)
        
        # ✅ 强制设置控制系统名称（修复属性名不一致问题）
        self.controlsystem = controlSystem
        self.controlSystem = controlSystem

        print("\n" + "=" * 60)
        print("🚢 OtterStationKeeping 初始化")
        print("=" * 60)

        # 航点配置
        if waypoints is None:
            self.waypoints = [
                [0, 0],
                [30, 20],
                [60, 30],
                [0, 0]
            ]
        else:
            self.waypoints = waypoints

        self.current_wp = 0
        self.arrival_radius = 5.0

        # LOS制导参数
        self.Delta = 10.0
        self.los_psi_d = r_los * np.pi / 180

        # 镇定控制参数
        self.station_duration = station_duration
        self.station_radius = station_radius
        self.station_timer = 0.0
        self.station_point = None

        # 自适应参数
        self.K_p = np.diag([80.0, 80.0, 30.0])
        self.K_d = np.diag([40.0, 40.0, 20.0])
        self.theta_hat = np.zeros(3)
        self.Gamma = np.diag([0.15, 0.15, 0.08])

        # 控制模式
        self.control_mode = "GUIDANCE"
        self.mode_history = []
        self.call_count = 0

        # ✅ 使用 simTime 而不是 sim_time（匹配父类）
        self.simTime = 0.0

        print(f"\n📍 任务配置:")
        print(f"   航点数量: {len(self.waypoints)}")
        print(f"   镇定时长: {self.station_duration}秒/点")
        print(f"   误差圆半径: {self.station_radius}米")
        print("=" * 60 + "\n")

    def los_guidance(self, eta, nu, sampleTime):
        """LOS制导算法"""

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
            return self.los_guidance(eta, nu, sampleTime)

        # 路径方向角
        alpha_path = np.arctan2(dy, dx)

        # 横向偏差
        dx_ship = x - wp_current[0]
        dy_ship = y - wp_current[1]
        e = -dx_ship * np.sin(alpha_path) + dy_ship * np.cos(alpha_path)

        # LOS期望航向
        los_angle = np.arctan2(-e, self.Delta)
        self.los_psi_d = alpha_path + los_angle
        self.los_psi_d = np.arctan2(np.sin(self.los_psi_d), np.cos(self.los_psi_d))

        # 到达判定
        dist_to_next = np.sqrt((x - wp_next[0]) ** 2 + (y - wp_next[1]) ** 2)

        if dist_to_next < self.arrival_radius:
            print(f"\n✅ 到达航点 {self.current_wp + 1}")
            self.current_wp += 1

            if self.current_wp < len(self.waypoints) - 1:
                self._switch_to_station_keeping(eta)
                return self.adaptive_station_keeping(eta, nu, sampleTime)
            else:
                print(f"\n🎉 完成所有航点！")
                return np.array([0, 0], float)

        # 航向控制
        e_psi = self.los_psi_d - psi
        e_psi = np.arctan2(np.sin(e_psi), np.cos(e_psi))

        r_d = self.wn * e_psi
        r = nu[5]
        e_r = r_d - r

        tau_N = 100.0 * e_psi + 50.0 * e_r
        tau_X = self.tauX

        [n1, n2] = self.controlAllocation(tau_X, tau_N)
        u_control = np.array([n1, n2], float)

        self.call_count += 1
        if self.call_count % 100 == 0:
            print(f"🔵 制导 | 位置:({x:5.1f},{y:5.1f}) | "
                  f"航向:{psi * 180 / np.pi:5.1f}° | "
                  f"距WP{self.current_wp + 1}:{dist_to_next:5.1f}m")

        return u_control

    def adaptive_station_keeping(self, eta, nu, sampleTime):
        """自适应镇定控制"""

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
            print(f"\n⚠️  偏差{position_error:.2f}m > 阈值{self.station_radius:.2f}m")
            self.control_mode = "EMERGENCY_GUIDANCE"
            self.mode_history.append(("EMERGENCY_GUIDANCE", self.simTime))
            return self._emergency_guidance(eta, nu, sampleTime)

        # 坐标变换
        R = np.array([
            [np.cos(psi), np.sin(psi), 0],
            [-np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        e_eta = np.array([e_x, e_y, e_psi])
        e_body = R @ e_eta
        nu_simple = np.array([u, v, r])

        # 控制律
        tau_p = -self.K_p @ e_body
        tau_d = -self.K_d @ nu_simple
        tau_adaptive = -self.theta_hat

        tau_body = tau_p + tau_d + tau_adaptive

        # 参数自适应
        theta_dot = self.Gamma @ e_body
        self.theta_hat = self.theta_hat + theta_dot * sampleTime
        self.theta_hat = np.clip(self.theta_hat, -100, 100)

        # 控制分配
        tau_X = tau_body[0]
        tau_N = tau_body[2]

        [n1, n2] = self.controlAllocation(tau_X, tau_N)
        u_control = np.array([n1, n2], float)

        # 更新计时器
        self.station_timer += sampleTime

        if self.station_timer >= self.station_duration:
            print(f"\n✅ 镇定完成（{self.station_timer:.1f}秒）")
            self.control_mode = "GUIDANCE"
            self.mode_history.append(("GUIDANCE", self.simTime))

        self.call_count += 1
        if self.call_count % 100 == 0:
            print(f"🟢 镇定 | 误差:{position_error:4.2f}m | "
                  f"时间:{self.station_timer:4.1f}/{self.station_duration:.0f}s")

        return u_control

    def _emergency_guidance(self, eta, nu, sampleTime):
        """紧急制导"""

        x = eta[0]
        y = eta[1]
        psi = eta[5]

        dx = self.station_point[0] - x
        dy = self.station_point[1] - y
        psi_d = np.arctan2(dy, dx)
        dist = np.sqrt(dx ** 2 + dy ** 2)

        if dist < self.station_radius * 0.8:
            print(f"\n✅ 返回镇定区域（距离{dist:.2f}m）")
            self.control_mode = "STATION_KEEPING"
            self.mode_history.append(("STATION_KEEPING", self.simTime))
            return self.adaptive_station_keeping(eta, nu, sampleTime)

        e_psi = psi_d - psi
        e_psi = np.arctan2(np.sin(e_psi), np.cos(e_psi))

        r = nu[5]
        tau_N = 120.0 * e_psi - 60.0 * r
        tau_X = self.tauX * 1.2

        [n1, n2] = self.controlAllocation(tau_X, tau_N)

        self.call_count += 1
        if self.call_count % 50 == 0:
            print(f"🔴 紧急 | 距离:{dist:5.2f}m")

        return np.array([n1, n2], float)

    def _switch_to_station_keeping(self, eta):
        """切换到镇定模式"""

        self.control_mode = "STATION_KEEPING"
        self.station_point = [eta[0], eta[1]]
        self.station_timer = 0.0
        self.theta_hat = np.zeros(3)
        self.mode_history.append(("STATION_KEEPING", self.simTime))

        print(f"🔄 切换到镇定模式")
        print(f"   目标点: ({eta[0]:.2f}, {eta[1]:.2f})\n")

    def headingAutopilot(self, eta, nu, sampleTime):
        """
        ✅ 重写控制器入口
        """

        # 如果不是新控制模式，使用原始控制器
        control_sys = getattr(self, 'controlsystem', getattr(self, 'controlSystem', ''))
        if control_sys != "LOS_STATION_KEEPING":
            return super().headingAutopilot(eta, nu, sampleTime)

        # 使用新控制器
        if self.control_mode == "GUIDANCE":
            return self.los_guidance(eta, nu, sampleTime)
        elif self.control_mode == "STATION_KEEPING":
            return self.adaptive_station_keeping(eta, nu, sampleTime)
        elif self.control_mode == "EMERGENCY_GUIDANCE":
            return self._emergency_guidance(eta, nu, sampleTime)
        else:
            return super().headingAutopilot(eta, nu, sampleTime)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("测试 OtterStationKeeping")
    print("=" * 60 + "\n")

    ship = OtterStationKeeping(
        controlSystem="LOS_STATION_KEEPING",
        r_los=0,
        V_current=0.3,
        beta_current=30,
        tau_X=120
    )

    print("✅ 对象创建成功")
    print(f"   继承自: {ship.__class__.__bases__}")
    print(f"   控制系统: {ship.controlSystem}")
    print(f"   航点数量: {len(ship.waypoints)}")