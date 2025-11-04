#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时动画版本 - 基于OtterStationKeeping类
Real-time Animation for Otter Station Keeping System

基于您现有的完整架构：
- 使用 OtterStationKeeping 类（继承自 otter）
- 保持完整的 6DOF 动力学模型
- 使用原始的 dynamics() 和控制分配
- 添加实时动画显示

作者: [您的名字]
日期: 2025-11-03
"""

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Rectangle
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import sys
import os

# 添加路径（根据您的项目结构调整）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 导入您的类
from python_vehicle_simulator.vehicles.otter_station_keeping import OtterStationKeeping
from python_vehicle_simulator.lib.gnc import Rzyx


class RealtimeAnimationOtter:
    """
    基于 OtterStationKeeping 的实时动画

    完全使用您现有的系统架构，只添加可视化功能
    """

    def __init__(self, ship, T_final=300, dt=0.02, skip_frames=5):
        """
        参数:
            ship: OtterStationKeeping 对象
            T_final: 仿真总时长
            dt: 时间步长
            skip_frames: 每次显示更新跳过的帧数（越大越快，但越不流畅）
                        建议值: 1-10
                        1 = 实时显示每一帧（最慢）
                        5 = 跳过4帧显示1帧（快5倍）
                        10 = 跳过9帧显示1帧（快10倍）
        """
        self.ship = ship
        self.T_final = T_final
        self.dt = dt
        self.N = int(T_final / dt)
        self.skip_frames = skip_frames  # 新增：跳帧参数

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

        print("\n" + "=" * 70)
        print("🎬 实时动画系统初始化完成")
        print("=" * 70)
        print(f"📊 使用类: {self.ship.__class__.__name__}")

        # 获取控制系统名称（兼容不同的属性名）
        control_sys = getattr(self.ship, 'controlSystem',
                              getattr(self.ship, 'controlsystem', 'Unknown'))
        print(f"⚙️  控制系统: {control_sys}")
        print(f"📍 航点数量: {len(self.ship.waypoints)}")
        print(f"⏱️  仿真时长: {T_final}秒")
        print(f"🔄 时间步长: {dt}秒")
        print(f"⚡ 播放加速: {self.skip_frames}x (每帧跳过{self.skip_frames - 1}步)")
        print("=" * 70 + "\n")

    def _setup_figure(self):
        """设置图形界面"""
        self.fig = plt.figure(figsize=(18, 10))
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
        for i, wp in enumerate(waypoints):
            # 航点标记
            ax.plot(wp[1], wp[0], 'gs', markersize=14,
                    markeredgecolor='darkgreen', markeredgewidth=2.5, zorder=10)
            ax.text(wp[1] + 1.5, wp[0] + 1.5, f'WP{i}',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4',
                              facecolor='yellow', alpha=0.8, edgecolor='black'))

            # 误差圆（只在任务点，不在起点和终点）
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

        # 船体表示
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
        ax.fill_between([0, self.T_final], 0, self.ship.station_radius,
                        alpha=0.15, color='green', label='Safe Zone')

        self.error_line, = ax.plot([], [], 'r-', linewidth=2)
        ax.set_xlim(0, self.T_final)
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
        ax.set_xlim(0, self.T_final)
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

        ax.set_xlim(0, self.T_final)
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

        ax.set_xlim(0, self.T_final)
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

        ax.set_xlim(0, self.T_final)
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
            else:
                self.fig.suptitle('🚢 Otter Station Keeping - ▶️  RUNNING (Press SPACE to pause)',
                                  fontsize=16, fontweight='bold', color='green')

    def _update_ship_arrow(self, x, y, psi):
        """更新船体箭头"""
        self.ship_arrow.remove()

        # 船体尺寸
        L = 4.0  # 显示长度
        W = 1.5  # 显示宽度

        # 箭头方向
        dx = L * np.sin(psi)
        dy = L * np.cos(psi)

        self.ship_arrow = self.ax_traj.add_patch(
            FancyArrow(y, x, dy, dx,
                       width=W * 0.4, head_width=W, head_length=L * 0.35,
                       color='red', alpha=0.95, zorder=20,
                       edgecolor='darkred', linewidth=2.5)
        )

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

        if self.current_step >= self.N:
            return self._init_animation()

        # 检查任务完成
        if self.ship.current_wp >= len(self.ship.waypoints) - 1:
            if self.ship.control_mode == "GUIDANCE":
                print(f"\n✅✅✅ 任务完成！(t={self.ship.simTime:.1f}秒) ✅✅✅\n")
                return self._init_animation()

        # ========================================
        # 每帧执行多个仿真步（加速显示）
        # ========================================
        for _ in range(self.skip_frames):
            if self.current_step >= self.N:
                break

            # 仿真步进（使用原始的dynamics）
            t = self.current_step * self.dt
            self.ship.simTime = t

            # ✅ 使用您的控制器
            u_control = self.ship.headingAutopilot(self.eta, self.nu, self.dt)

            # ✅ 使用原始的完整动力学模型
            [self.nu, self.u_actual] = self.ship.dynamics(
                self.eta, self.nu, self.u_actual, u_control, self.dt
            )

            # ✅ 运动学更新
            R = Rzyx(self.eta[3], self.eta[4], self.eta[5])
            eta_dot = np.concatenate([R @ self.nu[0:3], self.nu[3:6]])
            self.eta = self.eta + eta_dot * self.dt

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

        # 更新图形（只在跳帧后更新一次）
        self._update_plots()

        return self._init_animation()

    def _update_plots(self):
        """更新所有图表"""
        t = self.history['time']

        # ========================================
        # 1. 轨迹图 - 按模式分段着色
        # ========================================
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

        # ========================================
        # 2. 误差图
        # ========================================
        self.error_line.set_data(t, self.history['error'])

        # ========================================
        # 3. 模式图
        # ========================================
        self.mode_line.set_data(t, self.history['mode'])

        # ========================================
        # 4. 速度图
        # ========================================
        self.u_line.set_data(t, self.history['u'])
        self.v_line.set_data(t, self.history['v'])
        self.r_line.set_data(t, self.history['r'])

        # ========================================
        # 5. 航向图
        # ========================================
        self.psi_line.set_data(t, self.history['psi'])
        self.psi_d_line.set_data(t, self.history['los_psi_d'])

        # ========================================
        # 6. 控制输入图
        # ========================================
        self.n1_line.set_data(t, self.history['n1'])
        self.n2_line.set_data(t, self.history['n2'])

        # ========================================
        # 7. 信息文本
        # ========================================
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

        # ========================================
        # 自动调整显示范围
        # ========================================
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
        print("\n" + "=" * 70)
        print(" " * 15 + "🎬 启动实时动画 🎬")
        print("=" * 70)

        # ========================================
        # 任务配置详情
        # ========================================
        print("\n📋 任务配置:")
        print(f"   航点数量: {len(self.ship.waypoints)}")
        print(f"   镇定时长: {self.ship.station_duration}秒/点")
        print(f"   误差圆半径: {self.ship.station_radius}米")
        print(f"   前视距离: {self.ship.Delta}米")
        print(f"   洋流速度: {getattr(self.ship, 'V_c', 'N/A')} m/s")
        print(f"   洋流方向: {getattr(self.ship, 'beta_c', 'N/A')}°")

        print(f"\n🗺️  航点列表:")
        for i, wp in enumerate(self.ship.waypoints):
            wp_type = ""
            if i == 0:
                wp_type = " (起点)"
            elif i == len(self.ship.waypoints) - 1:
                wp_type = " (终点)"
            else:
                wp_type = f" (任务点{i})"
            print(f"   WP{i}: [{wp[0]:6.2f}, {wp[1]:6.2f}]{wp_type}")

        # ========================================
        # 仿真参数
        # ========================================
        print(f"\n⚙️  仿真参数:")
        # 获取控制系统名称（兼容不同的属性名）
        control_sys = getattr(self.ship, 'controlSystem',
                              getattr(self.ship, 'controlsystem', 'Unknown'))
        print(f"   控制系统: {control_sys}")
        print(f"   总时长: {self.T_final}秒")
        print(f"   时间步长: {self.dt}秒")
        print(f"   总步数: {self.N}")
        print(f"   播放加速: {self.skip_frames}x")
        print(f"   预计播放时长: ~{self.T_final / self.skip_frames:.1f}秒")

        # ========================================
        # 操作说明
        # ========================================
        print("\n🎮 操作说明:")
        print("   ▶️  按 SPACE 键 - 暂停/继续")
        print("   ❌ 关闭窗口 - 停止仿真")

        print("=" * 70 + "\n")

        # 创建动画
        self.anim = FuncAnimation(
            self.fig,
            self._update_frame,
            init_func=self._init_animation,
            frames=self.N,
            interval=0.1,  # 更新间隔(ms)，越小越快。0.1ms = 极快速度
            blit=False,
            repeat=False
        )

        plt.show()

        # 打印统计
        self._print_statistics()

    def save_to_matlab(self, filename='simulation_data.mat'):
        """
        手动保存数据到MATLAB格式

        参数:
            filename: 保存的文件名（默认: simulation_data.mat）

        返回:
            True: 保存成功
            False: 保存失败
        """
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
            print(f"✅ 数据已保存到: {filename}")
            return True

        except ImportError:
            print("❌ scipy未安装，无法保存MAT文件")
            print("   安装方法: pip install scipy")
            return False
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            return False

    def _print_statistics(self):
        """打印仿真统计并保存MATLAB数据"""
        print("\n" + "=" * 70)
        print(" " * 25 + "📊 仿真统计 📊")
        print("=" * 70)

        # ========================================
        # 1. 基本信息
        # ========================================
        print(f"\n⏱️  总仿真时间: {self.ship.simTime:.2f} 秒")
        print(f"📍 最终位置: ({self.eta[0]:.2f}, {self.eta[1]:.2f}) m")
        print(f"🧭 最终航向: {self.eta[5] * 180 / np.pi:.1f}°")
        print(f"🚢 最终速度: {np.sqrt(self.nu[0] ** 2 + self.nu[1] ** 2):.3f} m/s")

        # ========================================
        # 2. 任务配置信息
        # ========================================
        print(f"\n📋 任务配置:")
        print(f"   航点数量: {len(self.ship.waypoints)}")
        print(f"   完成航点: {self.ship.current_wp}/{len(self.ship.waypoints)}")
        print(f"   镇定时长: {self.ship.station_duration} 秒/点")
        print(f"   误差圆半径: {self.ship.station_radius} 米")
        print(f"   前视距离: {self.ship.Delta} 米")

        print(f"\n🗺️  航点坐标:")
        for i, wp in enumerate(self.ship.waypoints):
            wp_type = ""
            if i == 0:
                wp_type = " (起点)"
            elif i == len(self.ship.waypoints) - 1:
                wp_type = " (终点)"
            else:
                wp_type = f" (任务点{i})"
            print(f"   WP{i}: [{wp[0]:6.2f}, {wp[1]:6.2f}]{wp_type}")

        # ========================================
        # 3. 模式切换信息
        # ========================================
        print(f"\n🔄 模式切换:")
        print(f"   总切换次数: {len(self.ship.mode_history)}")

        if len(self.ship.mode_history) > 0:
            station_count = sum(1 for m, t in self.ship.mode_history if m == "STATION_KEEPING")
            emergency_count = sum(1 for m, t in self.ship.mode_history if m == "EMERGENCY_GUIDANCE")
            print(f"   镇定次数: {station_count}")
            print(f"   紧急制导次数: {emergency_count}")

            print(f"\n📝 模式切换历史:")
            for i, (mode, t) in enumerate(self.ship.mode_history):
                mode_symbol = {
                    'GUIDANCE': '🔵',
                    'STATION_KEEPING': '🟢',
                    'EMERGENCY_GUIDANCE': '🔴'
                }.get(mode, '⚪')
                print(f"   {i + 1}. t={t:7.2f}s  →  {mode_symbol} {mode}")

        # ========================================
        # 4. 镇定性能统计
        # ========================================
        errors = [e for e in self.history['error'] if e > 0]
        if len(errors) > 0:
            errors_array = np.array(errors)
            print(f"\n📊 镇定性能:")
            print(f"   数据点数: {len(errors)}")
            print(f"   最大误差: {np.max(errors_array):.4f} m")
            print(f"   最小误差: {np.min(errors_array):.4f} m")
            print(f"   平均误差: {np.mean(errors_array):.4f} m")
            print(f"   标准差:   {np.std(errors_array):.4f} m")
            print(f"   RMS误差:  {np.sqrt(np.mean(errors_array ** 2)):.4f} m")

            violations = sum(1 for e in errors if e > self.ship.station_radius)
            violation_rate = violations / len(errors) * 100
            print(f"\n   阈值: {self.ship.station_radius} m")
            print(f"   超出次数: {violations}/{len(errors)}")
            print(f"   超出率: {violation_rate:.2f}%")

            # 镇定时长统计
            station_mask = np.array([m == 1 for m in self.history['mode']])
            if np.any(station_mask):
                station_times = np.array(self.history['time'])[station_mask]
                if len(station_times) > 0:
                    total_station_time = station_times[-1] - station_times[0]
                    print(f"   总镇定时长: {total_station_time:.1f} 秒")

        # ========================================
        # 5. 控制性能统计
        # ========================================
        if len(self.history['n1']) > 0:
            print(f"\n⚙️  控制输入统计:")
            n1_array = np.array(self.history['n1'])
            n2_array = np.array(self.history['n2'])
            print(f"   左推进器 (n1):")
            print(f"     平均: {np.mean(n1_array):.2f} rad/s")
            print(f"     最大: {np.max(n1_array):.2f} rad/s")
            print(f"     最小: {np.min(n1_array):.2f} rad/s")
            print(f"   右推进器 (n2):")
            print(f"     平均: {np.mean(n2_array):.2f} rad/s")
            print(f"     最大: {np.max(n2_array):.2f} rad/s")
            print(f"     最小: {np.min(n2_array):.2f} rad/s")

        # ========================================
        # 6. 自适应参数统计
        # ========================================
        if len(self.history['theta_hat']) > 0:
            theta_array = np.array(self.history['theta_hat'])
            print(f"\n🔧 自适应参数 (最终值):")
            print(f"   θ̂_x (surge):  {theta_array[-1, 0]:7.3f}")
            print(f"   θ̂_y (sway):   {theta_array[-1, 1]:7.3f}")
            print(f"   θ̂_ψ (yaw):    {theta_array[-1, 2]:7.3f}")
            print(f"   ||θ̂||:        {np.linalg.norm(theta_array[-1]):.3f}")

        # ========================================
        # 7. 保存MATLAB数据
        # ========================================
        print("\n" + "=" * 70)
        print("💾 正在保存数据到MATLAB格式...")
        print("=" * 70)

        if self.save_to_matlab('simulation_data.mat'):
            print("\n📖 MATLAB使用说明:")
            print("   >> data = load('simulation_data.mat');")
            print("   >> figure; plot(data.y, data.x)")
            print("   >> xlabel('East (m)'); ylabel('North (m)');")
            print("   >> title('USV Trajectory');")
            print("\n📊 可用变量:")
            print("   - time, x, y, psi (位置和航向)")
            print("   - u, v, r (速度)")
            print("   - n1, n2 (推进器转速)")
            print("   - error (位置误差)")
            print("   - mode (控制模式: 0=制导, 1=镇定, 2=紧急)")
            print("   - theta_hat_x, theta_hat_y, theta_hat_psi (自适应参数)")
            print("   - waypoints, station_radius, station_duration (任务参数)")
        else:
            print("\n💡 备选方案：数据已保存在内存中")
            print("   可以使用 animation.history 访问所有数据")

        print("\n" + "=" * 70)
        print(" " * 20 + "✅ 仿真完成！✅")
        print("=" * 70 + "\n")


def main():
    """
    主函数
    """

    print("\n" + "=" * 70)
    print(" " * 10 + "🚢 Otter Station Keeping - Real-time Animation 🚢")
    print("=" * 70 + "\n")

    # ========================================
    # 1. 创建 OtterStationKeeping 对象
    # ========================================
    waypoints = [
        [0, 0],  # 起点
        [40, 0],  # 任务点1
        [40, 30],  # 任务点2
        [0, 30],  # 任务点3
        [0, 0]  # 返回起点
    ]

    print("正在创建 OtterStationKeeping 对象...")

    ship = OtterStationKeeping(
        controlSystem="LOS_STATION_KEEPING",  # 使用新的控制系统
        waypoints=waypoints,
        station_duration=25.0,  # 每个点镇定25秒
        station_radius=2.5,  # 误差圆半径2.5米
        V_current=0.3,  # 洋流速度
        beta_current=30,  # 洋流方向
        tau_X=120  # 基础推力
    )

    print("\n✅ 对象创建成功！\n")

    # ========================================
    # 2. 创建动画系统
    # ========================================
    animation = RealtimeAnimationOtter(
        ship=ship,
        T_final=300,  # 总仿真时长300秒
        dt=0.02,  # 时间步长0.02秒
        skip_frames=10  # ⚡ 播放速度：调整这个参数来控制动画速度
        #    1  = 正常速度（最慢，最流畅）
        #    5  = 5倍速（快，较流畅）
        #    10 = 10倍速（很快，推荐）
        #    20 = 20倍速（超快，可能不流畅）
    )

    # ========================================
    # 3. 运行动画
    # ========================================
    try:
        animation.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()