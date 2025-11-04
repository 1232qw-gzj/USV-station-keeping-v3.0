#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_station_keeping.py - 实时动画主程序（优化版）

改进:
- 使用配置管理器
- 使用logging系统
- 支持RK4积分（可选）
- 代码结构优化

基于OtterStationKeeping类的实时动画系统

作者: [您的名字]
日期: 2025-11-04
"""

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import sys
import os
import logging
import warnings
warnings.filterwarnings('ignore')

# ========================================
# 路径设置
# ========================================
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录（main_station_keeping.py 的上两级目录）
project_root = os.path.dirname(os.path.dirname(current_dir))

# 添加到搜索路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ========================================
# 调试：打印路径
# ========================================
print("=" * 80)
print("Python 搜索路径:")
for i, path in enumerate(sys.path[:5]):  # 只打印前5个
    print(f"{i}: {path}")
print("=" * 80)
print(f"当前目录: {current_dir}")
print(f"项目根目录: {project_root}")
print("=" * 80)

# ========================================
# 导入模块
# ========================================
try:
    # 导入配置管理器（在项目根目录）
    sys.path.insert(0, project_root)
    from config_manager import ConfigManager, setup_logging

    # 导入 USV 相关类（使用相对导入）
    from vehicles.otter_station_keeping import OtterStationKeeping
    from lib.gnc import Rzyx

    print("✅ 所有模块导入成功")
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print(f"\n请检查以下文件是否存在：")
    print(f"1. {os.path.join(project_root, 'config_manager.py')}")
    print(f"2. {os.path.join(current_dir, 'vehicles', 'otter_station_keeping.py')}")
    print(f"3. {os.path.join(current_dir, 'lib', 'gnc.py')}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# 获取模块日志器
logger = logging.getLogger(__name__)
# 获取模块日志器
logger = logging.getLogger(__name__)


class RealtimeAnimationOtter:
    """
    基于 OtterStationKeeping 的实时动画（优化版）

    改进：
    - 使用配置管理器
    - 使用logging
    - 支持RK4积分
    """

    def __init__(self, ship, config: ConfigManager):
        """
        参数:
            ship: OtterStationKeeping 对象
            config: ConfigManager 配置对象
        """
        self.ship = ship
        self.config = config

        # 从配置加载仿真参数
        self.t_final = config.get('simulation', 'total_time', default=300)
        self.dt = config.get('simulation', 'dt', default=0.02)
        self.skip_frames = config.get('simulation', 'skip_frames', default=10)
        self.integration_method = config.get('simulation', 'integration_method', default='euler')

        self.n_steps = int(self.t_final / self.dt)

        # 初始状态
        self.eta = np.array([0, 0, 0, 0, 0, 0], float)
        self.nu = np.array([0, 0, 0, 0, 0, 0], float)
        self.u_actual = np.array([0, 0], float)

        # 历史数据
        self.history = {
            'time': [],
            'x': [], 'y': [], 'psi': [],
            'u': [], 'v': [], 'r': [],
            'n1': [], 'n2': [],
            'mode': [],
            'error': [],
            'theta_hat': [],
            'los_psi_d': []
        }

        self.current_step = 0
        self.paused = False

        # 创建图形界面
        self._setup_figure()

        logger.info("=" * 70)
        logger.info("🎬 实时动画系统初始化完成")
        logger.info("=" * 70)
        logger.info(f"📊 使用类: {self.ship.__class__.__name__}")
        logger.info(f"⚙️  控制系统: {self.ship._control_system}")
        logger.info(f"📍 航点数量: {len(self.ship.waypoints)}")
        logger.info(f"⏱️  仿真时长: {self.t_final}秒")
        logger.info(f"🔄 时间步长: {self.dt}秒")
        logger.info(f"⚡ 播放加速: {self.skip_frames}x")
        logger.info(f"🧮 积分方法: {self.integration_method.upper()}")
        logger.info("=" * 70)

    def _setup_figure(self):
        """设置图形界面"""
        fig_size = self.config.get('visualization', 'figure_size', default=[18, 10])
        self.fig = plt.figure(figsize=fig_size)
        self.fig.suptitle('🚢 Otter Station Keeping - Real-time Animation (Press SPACE to pause)',
                          fontsize=16, fontweight='bold')

        # 使用GridSpec布局
        gs = GridSpec(3, 4, figure=self.fig, hspace=0.35, wspace=0.35)

        # 主轨迹图（占据左侧2列）
        self.ax_traj = self.fig.add_subplot(gs[:, 0:2])

        # 右侧子图
        self.ax_error = self.fig.add_subplot(gs[0, 2])
        self.ax_mode = self.fig.add_subplot(gs[1, 2])
        self.ax_speed = self.fig.add_subplot(gs[2, 2])
        self.ax_heading = self.fig.add_subplot(gs[0, 3])
        self.ax_control = self.fig.add_subplot(gs[1, 3])
        self.ax_info = self.fig.add_subplot(gs[2, 3])

        # 设置各子图
        self._setup_trajectory_plot()
        self._setup_error_plot()
        self._setup_mode_plot()
        self._setup_speed_plot()
        self._setup_heading_plot()
        self._setup_control_plot()
        self._setup_info_plot()

        # 键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _setup_trajectory_plot(self):
        """设置轨迹图"""
        ax = self.ax_traj
        ax.set_xlabel('Y - East (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('X - North (m)', fontsize=11, fontweight='bold')
        ax.set_title('Mission Trajectory', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axis('equal')

        # 绘制航点和路径
        waypoints = self.ship.waypoints
        wp_marker_size = self.config.get('visualization', 'waypoint_marker_size', default=14)

        for i, wp in enumerate(waypoints):
            # 航点标记
            ax.plot(wp[1], wp[0], 'gs', markersize=wp_marker_size,
                    markeredgecolor='darkgreen', markeredgewidth=2.5, zorder=10)
            ax.text(wp[1] + 1.5, wp[0] + 1.5, f'WP{i}',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4',
                              facecolor='yellow', alpha=0.8, edgecolor='black'))

            # 误差圆（只在任务点）
            if i > 0 and i < len(waypoints) - 1:
                circle = Circle((wp[1], wp[0]), self.ship.station_radius,
                                fill=False, edgecolor='green', linestyle='--',
                                linewidth=2.5, alpha=0.7, zorder=5)
                ax.add_patch(circle)

        # 路径连线
        for i in range(len(waypoints) - 1):
            wp1, wp2 = waypoints[i], waypoints[i + 1]
            ax.plot([wp1[1], wp2[1]], [wp1[0], wp2[0]],
                    'g--', linewidth=2, alpha=0.4, zorder=1)

        # 轨迹线（分模式着色）
        self.traj_guidance, = ax.plot([], [], 'b-', linewidth=2.5,
                                      alpha=0.8, label='Guidance', zorder=3)
        self.traj_station, = ax.plot([], [], 'g-', linewidth=2.5,
                                     alpha=0.8, label='Station Keeping', zorder=3)
        self.traj_emergency, = ax.plot([], [], 'r-', linewidth=2.5,
                                       alpha=0.8, label='Emergency', zorder=3)

        # 船体箭头
        self.ship_arrow = ax.add_patch(
            FancyArrow(0, 0, 0, 0, width=1.0, head_width=2.5, head_length=2.0,
                       color='red', alpha=0.95, zorder=20,
                       edgecolor='darkred', linewidth=2.5)
        )

        # 当前位置点
        self.ship_pos, = ax.plot([], [], 'r*', markersize=28,
                                 zorder=25, label='Current Position',
                                 markeredgecolor='darkred', markeredgewidth=1.5)

        # 镇定散点
        self.station_scatter = ax.scatter([], [], c='lime', s=20,
                                          alpha=0.5, zorder=2, label='Station Points')

        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    def _setup_error_plot(self):
        """误差图"""
        ax = self.ax_error
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Position Error (m)', fontsize=10)
        ax.set_title('Station Keeping Error', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 阈值线
        ax.axhline(y=self.ship.station_radius, color='red',
                   linestyle='--', linewidth=2, alpha=0.7, label='Threshold')
        ax.fill_between([0, self.t_final], 0, self.ship.station_radius,
                        alpha=0.15, color='green', label='Safe Zone')

        self.error_line, = ax.plot([], [], 'r-', linewidth=2)
        ax.set_xlim(0, self.t_final)
        ax.set_ylim(0, self.ship.station_radius * 3)
        ax.legend(fontsize=8, loc='upper right')

    def _setup_mode_plot(self):
        """模式图"""
        ax = self.ax_mode
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Control Mode', fontsize=10)
        ax.set_title('Mode Switching', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Guidance', 'Station\nKeeping', 'Emergency'], fontsize=9)

        self.mode_line, = ax.plot([], [], 'b-', linewidth=2.5, drawstyle='steps-post')
        ax.set_xlim(0, self.t_final)
        ax.set_ylim(-0.5, 2.5)

        # 模式颜色背景
        ax.axhspan(-0.5, 0.5, alpha=0.1, color='blue')
        ax.axhspan(0.5, 1.5, alpha=0.1, color='green')
        ax.axhspan(1.5, 2.5, alpha=0.1, color='red')

    def _setup_speed_plot(self):
        """速度图"""
        ax = self.ax_speed
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Velocity (m/s, rad/s)', fontsize=10)
        ax.set_title('Velocity Components', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        self.u_line, = ax.plot([], [], 'b-', linewidth=1.8, label='u (surge)')
        self.v_line, = ax.plot([], [], 'r-', linewidth=1.8, label='v (sway)')
        self.r_line, = ax.plot([], [], 'g-', linewidth=1.8, label='r (yaw)')

        ax.set_xlim(0, self.t_final)
        ax.set_ylim(-2, 3)
        ax.legend(fontsize=8, loc='upper right')

    def _setup_heading_plot(self):
        """航向图"""
        ax = self.ax_heading
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Heading (deg)', fontsize=10)
        ax.set_title('Heading Tracking', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        self.psi_line, = ax.plot([], [], 'b-', linewidth=2, label='Actual ψ')
        self.psi_d_line, = ax.plot([], [], 'r--', linewidth=2, label='Desired ψ_d (LOS)')

        ax.set_xlim(0, self.t_final)
        ax.set_ylim(-180, 180)
        ax.legend(fontsize=8, loc='upper right')

    def _setup_control_plot(self):
        """控制输入图"""
        ax = self.ax_control
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Propeller Speed (rad/s)', fontsize=10)
        ax.set_title('Control Inputs', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        self.n1_line, = ax.plot([], [], 'b-', linewidth=1.8, label='n₁ (left)')
        self.n2_line, = ax.plot([], [], 'r-', linewidth=1.8, label='n₂ (right)')

        ax.set_xlim(0, self.t_final)
        ax.set_ylim(-50, 200)
        ax.legend(fontsize=8, loc='upper right')

    def _setup_info_plot(self):
        """信息显示"""
        self.ax_info.axis('off')
        self.info_text = self.ax_info.text(
            0.05, 0.95, '', transform=self.ax_info.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                      alpha=0.9, edgecolor='black', linewidth=1.5)
        )

    def _on_key(self, event):
        """键盘事件处理"""
        if event.key == ' ':
            self.paused = not self.paused
            if self.paused:
                self.fig.suptitle('🚢 Otter Station Keeping - ⏸️  PAUSED (Press SPACE to resume)',
                                  fontsize=16, fontweight='bold', color='red')
                logger.info("⏸️  仿真暂停")
            else:
                self.fig.suptitle('🚢 Otter Station Keeping - ▶️  RUNNING (Press SPACE to pause)',
                                  fontsize=16, fontweight='bold', color='green')
                logger.info("▶️  仿真继续")

    def _update_ship_arrow(self, x, y, psi):
        """更新船体箭头"""
        self.ship_arrow.remove()

        # 从配置获取船体显示尺寸
        length = self.config.get('visualization', 'ship_length', default=4.0)
        width = self.config.get('visualization', 'ship_width', default=1.5)

        # 箭头方向
        dx = length * np.sin(psi)
        dy = length * np.cos(psi)

        self.ship_arrow = self.ax_traj.add_patch(
            FancyArrow(y, x, dy, dx,
                       width=width * 0.4, head_width=width, head_length=length * 0.35,
                       color='red', alpha=0.95, zorder=20,
                       edgecolor='darkred', linewidth=2.5)
        )

    def _dynamics_rk4(self, eta, nu, u_actual, u_control, dt):
        """
        RK4积分方法（4阶Runge-Kutta）

        提供比Euler方法更高的精度
        """

        def derivative(eta_curr, nu_curr, u_act_curr, u_ctrl):
            """计算导数"""
            # 调用ship的dynamics获取下一步状态（使用很小的dt）
            [nu_next, u_act_next] = self.ship.dynamics(
                eta_curr, nu_curr, u_act_curr, u_ctrl, dt
            )

            # 计算速度导数
            nu_dot = (nu_next - nu_curr) / dt

            # 运动学导数
            R = Rzyx(eta_curr[3], eta_curr[4], eta_curr[5])
            eta_dot = np.concatenate([R @ nu_curr[0:3], nu_curr[3:6]])

            # 推进器导数
            u_act_dot = (u_act_next - u_act_curr) / dt

            return eta_dot, nu_dot, u_act_dot

        # RK4 四个斜率
        k1_eta, k1_nu, k1_u = derivative(eta, nu, u_actual, u_control)

        eta_2 = eta + 0.5 * dt * k1_eta
        nu_2 = nu + 0.5 * dt * k1_nu
        u_2 = u_actual + 0.5 * dt * k1_u
        k2_eta, k2_nu, k2_u = derivative(eta_2, nu_2, u_2, u_control)

        eta_3 = eta + 0.5 * dt * k2_eta
        nu_3 = nu + 0.5 * dt * k2_nu
        u_3 = u_actual + 0.5 * dt * k2_u
        k3_eta, k3_nu, k3_u = derivative(eta_3, nu_3, u_3, u_control)

        eta_4 = eta + dt * k3_eta
        nu_4 = nu + dt * k3_nu
        u_4 = u_actual + dt * k3_u
        k4_eta, k4_nu, k4_u = derivative(eta_4, nu_4, u_4, u_control)

        # 组合
        eta_next = eta + (dt / 6.0) * (k1_eta + 2 * k2_eta + 2 * k3_eta + k4_eta)
        nu_next = nu + (dt / 6.0) * (k1_nu + 2 * k2_nu + 2 * k3_nu + k4_nu)
        u_actual_next = u_actual + (dt / 6.0) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)

        return nu_next, u_actual_next, eta_next

    def _integrate_step(self, u_control):
        """
        执行一步积分（根据配置选择方法）
        """
        if self.integration_method == 'rk4':
            # 使用RK4方法
            nu_next, u_actual_next, eta_next = self._dynamics_rk4(
                self.eta, self.nu, self.u_actual, u_control, self.dt
            )
            self.nu = nu_next
            self.u_actual = u_actual_next
            self.eta = eta_next
        else:
            # 使用默认Euler方法
            [self.nu, self.u_actual] = self.ship.dynamics(
                self.eta, self.nu, self.u_actual, u_control, self.dt
            )

            # 运动学更新
            R = Rzyx(self.eta[3], self.eta[4], self.eta[5])
            eta_dot = np.concatenate([R @ self.nu[0:3], self.nu[3:6]])
            self.eta = self.eta + eta_dot * self.dt

    def _init_animation(self):
        """初始化动画"""
        return (self.traj_guidance, self.traj_station, self.traj_emergency,
                self.ship_pos, self.error_line, self.mode_line,
                self.u_line, self.v_line, self.r_line,
                self.psi_line, self.psi_d_line,
                self.n1_line, self.n2_line)

    def _update_frame(self, frame):
        """更新每一帧"""
        if self.paused:
            return self._init_animation()

        if self.current_step >= self.n_steps:
            return self._init_animation()

        # 检查任务完成
        if self.ship.current_wp >= len(self.ship.waypoints) - 1:
            if self.ship.control_mode == "GUIDANCE":
                logger.info(f"✅✅✅ 任务完成！(t={self.ship.simTime:.1f}秒) ✅✅✅")
                return self._init_animation()

        # 每帧执行多个仿真步（加速显示）
        for _ in range(self.skip_frames):
            if self.current_step >= self.n_steps:
                break

            # 仿真步进
            t = self.current_step * self.dt
            self.ship.simTime = t

            # 调用控制器
            u_control = self.ship.headingAutopilot(self.eta, self.nu, self.dt)

            # 积分（根据配置选择方法）
            self._integrate_step(u_control)

            # 计算误差
            if self.ship.control_mode == "STATION_KEEPING" and self.ship.station_point:
                error = np.sqrt((self.eta[0] - self.ship.station_point[0]) ** 2 +
                                (self.eta[1] - self.ship.station_point[1]) ** 2)
            else:
                error = 0

            # 记录数据
            self.history['time'].append(t)
            self.history['x'].append(self.eta[0])
            self.history['y'].append(self.eta[1])
            self.history['psi'].append(self.eta[5] * 180 / np.pi)
            self.history['u'].append(self.nu[0])
            self.history['v'].append(self.nu[1])
            self.history['r'].append(self.nu[5])
            self.history['n1'].append(self.u_actual[0])
            self.history['n2'].append(self.u_actual[1])
            self.history['error'].append(error)
            self.history['los_psi_d'].append(self.ship.los_psi_d * 180 / np.pi)
            self.history['theta_hat'].append(self.ship.theta_hat.copy())

            # 模式编码
            mode_map = {'GUIDANCE': 0, 'STATION_KEEPING': 1, 'EMERGENCY_GUIDANCE': 2}
            self.history['mode'].append(mode_map.get(self.ship.control_mode, 0))

            self.current_step += 1

        # 更新图形
        self._update_plots()

        return self._init_animation()

    def _update_plots(self):
        """更新所有图表"""
        t = self.history['time']

        # 轨迹图 - 按模式分段着色
        x_all = np.array(self.history['x'])
        y_all = np.array(self.history['y'])
        mode_all = np.array(self.history['mode'])

        # 提取不同模式的轨迹
        guidance_mask = (mode_all == 0)
        station_mask = (mode_all == 1)
        emergency_mask = (mode_all == 2)

        if np.any(guidance_mask):
            self.traj_guidance.set_data(y_all[guidance_mask], x_all[guidance_mask])
        if np.any(station_mask):
            self.traj_station.set_data(y_all[station_mask], x_all[station_mask])
        if np.any(emergency_mask):
            self.traj_emergency.set_data(y_all[emergency_mask], x_all[emergency_mask])

        # 镇定散点
        if np.any(station_mask):
            self.station_scatter.set_offsets(np.c_[y_all[station_mask], x_all[station_mask]])

        # 当前位置
        self.ship_pos.set_data([self.eta[1]], [self.eta[0]])
        self._update_ship_arrow(self.eta[0], self.eta[1], self.eta[5])

        # 误差图
        self.error_line.set_data(t, self.history['error'])

        # 模式图
        self.mode_line.set_data(t, self.history['mode'])

        # 速度图
        self.u_line.set_data(t, self.history['u'])
        self.v_line.set_data(t, self.history['v'])
        self.r_line.set_data(t, self.history['r'])

        # 航向图
        self.psi_line.set_data(t, self.history['psi'])
        self.psi_d_line.set_data(t, self.history['los_psi_d'])

        # 控制输入图
        self.n1_line.set_data(t, self.history['n1'])
        self.n2_line.set_data(t, self.history['n2'])

        # 信息文本
        speed = np.sqrt(self.nu[0] ** 2 + self.nu[1] ** 2)
        info = f"""
╔════════════════════════════════╗
║      SIMULATION STATUS         ║
╚════════════════════════════════╝

⏱️  Time:        {self.ship.simTime:7.2f} s

📍 Position:
   North (X):    {self.eta[0]:7.2f} m
   East (Y):     {self.eta[1]:7.2f} m

🧭 Heading:      {self.eta[5] * 180 / np.pi:7.2f}°
🚢 Speed:        {speed:7.3f} m/s

🎯 Mission:
   Mode:         {self.ship.control_mode}
   Waypoint:     {self.ship.current_wp + 1}/{len(self.ship.waypoints)}

📊 Performance:
   Position Error: {self.history['error'][-1]:6.3f} m
   Threshold:      {self.ship.station_radius:.2f} m

⚙️  Adaptive Params:
   θ̂_x:   {self.ship.theta_hat[0]:6.2f}
   θ̂_y:   {self.ship.theta_hat[1]:6.2f}
   θ̂_ψ:   {self.ship.theta_hat[2]:6.2f}
        """
        self.info_text.set_text(info)

        # 自动调整显示范围
        if len(t) > 10:
            curr_t = t[-1]

            # 时间轴自适应
            if curr_t > 20:
                self.ax_error.set_xlim(0, curr_t + 10)
                self.ax_mode.set_xlim(0, curr_t + 10)
                self.ax_speed.set_xlim(0, curr_t + 10)
                self.ax_heading.set_xlim(0, curr_t + 10)
                self.ax_control.set_xlim(0, curr_t + 10)

            # 轨迹图自适应
            if len(x_all) > 1:
                margin = 10
                self.ax_traj.set_xlim(min(y_all) - margin, max(y_all) + margin)
                self.ax_traj.set_ylim(min(x_all) - margin, max(x_all) + margin)

    def run(self):
        """运行动画"""
        logger.info("=" * 70)
        logger.info("🎬 启动实时动画")
        logger.info("=" * 70)

        # 任务配置详情
        logger.info("")
        logger.info("📋 任务配置:")
        logger.info(f"   航点数量: {len(self.ship.waypoints)}")
        logger.info(f"   镇定时长: {self.ship.station_duration}秒/点")
        logger.info(f"   误差圆半径: {self.ship.station_radius}米")
        logger.info(f"   前视距离: {self.ship.delta}米")
        logger.info(f"   洋流速度: {getattr(self.ship, 'V_c', 'N/A')} m/s")
        logger.info(
            f"   洋流方向: {getattr(self.ship, 'beta_c', 'N/A') * 180 / np.pi if hasattr(self.ship, 'beta_c') else 'N/A'}°")

        logger.info("")
        logger.info("🗺️  航点列表:")
        for i, wp in enumerate(self.ship.waypoints):
            wp_type = ""
            if i == 0:
                wp_type = " (起点)"
            elif i == len(self.ship.waypoints) - 1:
                wp_type = " (终点)"
            else:
                wp_type = f" (任务点{i})"
            logger.info(f"   WP{i}: [{wp[0]:6.2f}, {wp[1]:6.2f}]{wp_type}")

        # 仿真参数
        logger.info("")
        logger.info("⚙️  仿真参数:")
        logger.info(f"   控制系统: {self.ship._control_system}")
        logger.info(f"   总时长: {self.t_final}秒")
        logger.info(f"   时间步长: {self.dt}秒")
        logger.info(f"   总步数: {self.n_steps}")
        logger.info(f"   播放加速: {self.skip_frames}x")
        logger.info(f"   积分方法: {self.integration_method.upper()}")
        logger.info(f"   预计播放时长: ~{self.t_final / self.skip_frames:.1f}秒")

        # 操作说明
        logger.info("")
        logger.info("🎮 操作说明:")
        logger.info("   ▶️  按 SPACE 键 - 暂停/继续")
        logger.info("   ❌ 关闭窗口 - 停止仿真")
        logger.info("=" * 70)

        # 创建动画
        update_interval = self.config.get('visualization', 'update_interval', default=0.1)
        self.anim = FuncAnimation(
            self.fig,
            self._update_frame,
            init_func=self._init_animation,
            frames=self.n_steps,
            interval=update_interval,
            blit=False,
            repeat=False
        )

        plt.show()

        # 打印统计
        self._print_statistics()

    def save_to_matlab(self, filename=None):
        """保存数据到MATLAB格式"""
        if filename is None:
            filename = self.config.get('output', 'matlab_filename', default='simulation_data.mat')

        try:
            from scipy.io import savemat

            # 转换模式为数字
            mode_numeric = np.array(self.history['mode'])

            # 准备数据字典
            matlab_data = {
                'time': np.array(self.history['time']),
                'x': np.array(self.history['x']),
                'y': np.array(self.history['y']),
                'psi': np.array(self.history['psi']),
                'u': np.array(self.history['u']),
                'v': np.array(self.history['v']),
                'r': np.array(self.history['r']),
                'n1': np.array(self.history['n1']),
                'n2': np.array(self.history['n2']),
                'error': np.array(self.history['error']),
                'los_psi_d': np.array(self.history['los_psi_d']),
                'mode': mode_numeric,
                'waypoints': np.array(self.ship.waypoints),
                'station_radius': self.ship.station_radius,
                'station_duration': self.ship.station_duration,
            }

            # 添加自适应参数
            if len(self.history['theta_hat']) > 0:
                theta_hat_array = np.array(self.history['theta_hat'])
                matlab_data['theta_hat_x'] = theta_hat_array[:, 0]
                matlab_data['theta_hat_y'] = theta_hat_array[:, 1]
                matlab_data['theta_hat_psi'] = theta_hat_array[:, 2]

            # 保存文件
            savemat(filename, matlab_data)
            logger.info(f"✅ 数据已保存到: {filename}")
            return True

        except ImportError:
            logger.error("❌ scipy未安装，无法保存MAT文件")
            logger.error("   安装方法: pip install scipy")
            return False
        except Exception as e:
            logger.error(f"❌ 保存失败: {e}")
            return False

    def _print_statistics(self):
        """打印仿真统计并保存MATLAB数据"""
        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 仿真统计")
        logger.info("=" * 70)

        # 基本信息
        logger.info("")
        logger.info(f"⏱️  总仿真时间: {self.ship.simTime:.2f} 秒")
        logger.info(f"📍 最终位置: ({self.eta[0]:.2f}, {self.eta[1]:.2f}) m")
        logger.info(f"🧭 最终航向: {self.eta[5] * 180 / np.pi:.1f}°")
        logger.info(f"🚢 最终速度: {np.sqrt(self.nu[0] ** 2 + self.nu[1] ** 2):.3f} m/s")

        # 任务配置信息
        logger.info("")
        logger.info("📋 任务配置:")
        logger.info(f"   航点数量: {len(self.ship.waypoints)}")
        logger.info(f"   完成航点: {self.ship.current_wp}/{len(self.ship.waypoints)}")
        logger.info(f"   镇定时长: {self.ship.station_duration} 秒/点")
        logger.info(f"   误差圆半径: {self.ship.station_radius} 米")
        logger.info(f"   前视距离: {self.ship.delta} 米")

        # 模式切换信息
        logger.info("")
        logger.info("🔄 模式切换:")
        logger.info(f"   总切换次数: {len(self.ship.mode_history)}")

        if len(self.ship.mode_history) > 0:
            station_count = sum(1 for m, t in self.ship.mode_history if m == "STATION_KEEPING")
            emergency_count = sum(1 for m, t in self.ship.mode_history if m == "EMERGENCY_GUIDANCE")
            logger.info(f"   镇定次数: {station_count}")
            logger.info(f"   紧急制导次数: {emergency_count}")

        # 镇定性能统计
        errors = [e for e in self.history['error'] if e > 0]
        if len(errors) > 0:
            errors_array = np.array(errors)
            logger.info("")
            logger.info("📊 镇定性能:")
            logger.info(f"   数据点数: {len(errors)}")
            logger.info(f"   最大误差: {np.max(errors_array):.4f} m")
            logger.info(f"   最小误差: {np.min(errors_array):.4f} m")
            logger.info(f"   平均误差: {np.mean(errors_array):.4f} m")
            logger.info(f"   标准差:   {np.std(errors_array):.4f} m")
            logger.info(f"   RMS误差:  {np.sqrt(np.mean(errors_array ** 2)):.4f} m")

            violations = sum(1 for e in errors if e > self.ship.station_radius)
            violation_rate = violations / len(errors) * 100
            logger.info(f"   阈值: {self.ship.station_radius} m")
            logger.info(f"   超出次数: {violations}/{len(errors)}")
            logger.info(f"   超出率: {violation_rate:.2f}%")

        # 控制性能统计
        if len(self.history['n1']) > 0:
            logger.info("")
            logger.info("⚙️  控制输入统计:")
            n1_array = np.array(self.history['n1'])
            n2_array = np.array(self.history['n2'])
            logger.info(f"   左推进器 (n1):")
            logger.info(f"     平均: {np.mean(n1_array):.2f} rad/s")
            logger.info(f"     最大: {np.max(n1_array):.2f} rad/s")
            logger.info(f"     最小: {np.min(n1_array):.2f} rad/s")
            logger.info(f"   右推进器 (n2):")
            logger.info(f"     平均: {np.mean(n2_array):.2f} rad/s")
            logger.info(f"     最大: {np.max(n2_array):.2f} rad/s")
            logger.info(f"     最小: {np.min(n2_array):.2f} rad/s")

        # 自适应参数统计
        if len(self.history['theta_hat']) > 0:
            theta_array = np.array(self.history['theta_hat'])
            logger.info("")
            logger.info("🔧 自适应参数 (最终值):")
            logger.info(f"   θ̂_x (surge):  {theta_array[-1, 0]:7.3f}")
            logger.info(f"   θ̂_y (sway):   {theta_array[-1, 1]:7.3f}")
            logger.info(f"   θ̂_ψ (yaw):    {theta_array[-1, 2]:7.3f}")
            logger.info(f"   ||θ̂||:        {np.linalg.norm(theta_array[-1]):.3f}")

        # 保存MATLAB数据
        if self.config.get('output', 'save_matlab', default=True):
            logger.info("")
            logger.info("=" * 70)
            logger.info("💾 正在保存数据到MATLAB格式...")
            logger.info("=" * 70)

            if self.save_to_matlab():
                logger.info("")
                logger.info("📖 MATLAB使用说明:")
                logger.info("   >> data = load('simulation_data.mat');")
                logger.info("   >> figure; plot(data.y, data.x)")
                logger.info("   >> xlabel('East (m)'); ylabel('North (m)');")

        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ 仿真完成！")
        logger.info("=" * 70)


def main():
    """主函数"""

    # 解析命令行参数（可选）
    import argparse
    parser = argparse.ArgumentParser(description='USV Station Keeping Simulation')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径 (默认: config.yaml)')
    parser.add_argument('--log-level', type=str, default=None,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志级别 (覆盖配置文件)')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" " * 10 + "🚢 Otter Station Keeping System (优化版) 🚢")
    print("=" * 70 + "\n")

    # 加载配置
    config = ConfigManager(args.config)

    # 设置日志（允许命令行参数覆盖）
    if args.log_level:
        config.config['logging']['level'] = args.log_level
    setup_logging(config)

    logger.info("正在初始化系统...")

    # 打印配置摘要
    config.print_summary()

    # 创建 OtterStationKeeping 对象
    try:
        ship = OtterStationKeeping(config=config)
        logger.info("✅ 船舶对象创建成功")
    except Exception as e:
        logger.error(f"❌ 船舶对象创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 创建动画系统
    try:
        animation = RealtimeAnimationOtter(ship=ship, config=config)
        logger.info("✅ 动画系统创建成功")
    except Exception as e:
        logger.error(f"❌ 动画系统创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 运行动画
    try:
        animation.run()
    except KeyboardInterrupt:
        logger.warning("⚠️  用户中断仿真")
    except Exception as e:
        logger.error(f"❌ 仿真错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
