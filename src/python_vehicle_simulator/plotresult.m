%% plot_simulation_data.m
% 用MATLAB绘制Python仿真数据
%
% 使用方法:
%   1. 在Python中运行 main_detailed_plots.py
%   2. 会生成 simulation_data.mat 文件
%   3. 在MATLAB中运行此脚本

clear; clc; close all;

%% 加载数据
fprintf('正在加载仿真数据...\n');
data = load('simulation_data.mat');

% 提取数据并转换为double类型（避免类型错误）
time = double(data.time);
x = double(data.x);
y = double(data.y);
psi = double(data.psi);
u = double(data.u);
v = double(data.v);
r = double(data.r);
n1 = double(data.n1);
n2 = double(data.n2);
error = double(data.error);
theta_x = double(data.theta_hat_x);
theta_y = double(data.theta_hat_y);
theta_psi = double(data.theta_hat_psi);
los_psi_d = double(data.los_psi_d);
mode = double(data.mode);
waypoints = double(data.waypoints);
station_radius = double(data.station_radius);

fprintf('✅ 数据加载完成\n');
fprintf('   数据点数: %d\n', length(time));
fprintf('   仿真时间: %.1f秒\n', time(end));

%% 图1: 任务轨迹
figure('Name', '任务轨迹', 'NumberTitle', 'off', 'Position', [100 100 1200 800]);

% 1.1 轨迹图
subplot(2,2,1);
hold on; grid on; box on;

% 根据模式着色
guidance_idx = (mode == 0);
station_idx = (mode == 1);
emergency_idx = (mode == 2);

if sum(guidance_idx) > 0
    plot(y(guidance_idx), x(guidance_idx), 'b.', 'MarkerSize', 2);
end
if sum(station_idx) > 0
    plot(y(station_idx), x(station_idx), 'g.', 'MarkerSize', 2);
end
if sum(emergency_idx) > 0
    plot(y(emergency_idx), x(emergency_idx), 'r.', 'MarkerSize', 2);
end

% 起点和终点
plot(y(1), x(1), 'go', 'MarkerSize', 12, 'MarkerFaceColor', 'g');
plot(y(end), x(end), 'r*', 'MarkerSize', 15);

% 航点和误差圆
for i = 1:size(waypoints, 1)
    plot(waypoints(i,2), waypoints(i,1), 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    text(waypoints(i,2)+1, waypoints(i,1)+1, sprintf('WP%d', i-1), ...
        'FontSize', 10, 'FontWeight', 'bold');
    
    % 误差圆
    if i > 1 && i < size(waypoints, 1)
        theta = linspace(0, 2*pi, 100);
        xc = waypoints(i,2) + station_radius * cos(theta);
        yc = waypoints(i,1) + station_radius * sin(theta);
        plot(xc, yc, 'g--', 'LineWidth', 1.5);
    end
end

xlabel('East (m)', 'FontSize', 12);
ylabel('North (m)', 'FontSize', 12);
title('(a) 任务轨迹', 'FontSize', 13, 'FontWeight', 'bold');
legend('制导', '镇定', '紧急', '起点', '终点', 'Location', 'best');
axis equal;

% 1.2 位置误差
subplot(2,2,2);
hold on; grid on; box on;

plot(time, error, 'r-', 'LineWidth', 1.5);
plot([time(1) time(end)], [station_radius station_radius], 'g--', 'LineWidth', 1.5);

% 填充安全区域
patch([time(1) time(end) time(end) time(1)], ...
      [0 0 station_radius station_radius], ...
      'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

xlabel('时间 (s)', 'FontSize', 12);
ylabel('位置误差 (m)', 'FontSize', 12);
title('(b) 镇定误差', 'FontSize', 13, 'FontWeight', 'bold');
legend('误差', '阈值', 'Location', 'best');
ylim([0, max(station_radius*2, max(error)*1.1)]);

% 1.3 控制模式
subplot(2,2,3);
hold on; grid on; box on;

plot(time, mode, 'b-', 'LineWidth', 2);
xlabel('时间 (s)', 'FontSize', 12);
ylabel('控制模式', 'FontSize', 12);
title('(c) 模式切换', 'FontSize', 13, 'FontWeight', 'bold');
yticks([0 1 2]);
yticklabels({'制导', '镇定', '紧急'});
ylim([-0.5 2.5]);

% 1.4 控制输入
subplot(2,2,4);
hold on; grid on; box on;

plot(time, n1, 'b-', 'LineWidth', 1.5, 'DisplayName', 'n_1 (左)');
plot(time, n2, 'r-', 'LineWidth', 1.5, 'DisplayName', 'n_2 (右)');
xlabel('时间 (s)', 'FontSize', 12);
ylabel('螺旋桨转速 (rad/s)', 'FontSize', 12);
title('(d) 控制输入', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best');

%% 图2: 镇定性能分析
figure('Name', '镇定性能分析', 'NumberTitle', 'off', 'Position', [150 150 1200 900]);

% 提取镇定阶段数据
station_mask = (mode == 1);
station_time = time(station_mask);
station_x = x(station_mask);
station_y = y(station_mask);
station_error = error(station_mask);
station_psi = psi(station_mask);

if ~isempty(station_x)
    % 2.1 位置散点云
    subplot(3,2,1);
    hold on; grid on; box on;
    
    scatter(station_y, station_x, 5, station_time, 'filled', 'MarkerFaceAlpha', 0.6);
    colorbar;
    colormap('jet');
    xlabel('East (m)', 'FontSize', 11);
    ylabel('North (m)', 'FontSize', 11);
    title('(a) 镇定时的位置分布', 'FontSize', 12, 'FontWeight', 'bold');
    
    % 绘制误差圆
    for i = 2:size(waypoints,1)-1
        theta = linspace(0, 2*pi, 100);
        xc = waypoints(i,2) + station_radius * cos(theta);
        yc = waypoints(i,1) + station_radius * sin(theta);
        plot(xc, yc, 'r--', 'LineWidth', 2);
        plot(waypoints(i,2), waypoints(i,1), 'r*', 'MarkerSize', 12);
    end
    axis equal;
    
    % 2.2 X方向位置
    subplot(3,2,2);
    plot(station_time, station_x, 'b-', 'LineWidth', 1);
    grid on; box on;
    xlabel('时间 (s)', 'FontSize', 11);
    ylabel('北向位置 (m)', 'FontSize', 11);
    title('(b) 北向位置', 'FontSize', 12, 'FontWeight', 'bold');
    
    % 2.3 Y方向位置
    subplot(3,2,3);
    plot(station_time, station_y, 'r-', 'LineWidth', 1);
    grid on; box on;
    xlabel('时间 (s)', 'FontSize', 11);
    ylabel('东向位置 (m)', 'FontSize', 11);
    title('(c) 东向位置', 'FontSize', 12, 'FontWeight', 'bold');
    
    % 2.4 航向
    subplot(3,2,4);
    plot(station_time, station_psi, 'g-', 'LineWidth', 1);
    grid on; box on;
    xlabel('时间 (s)', 'FontSize', 11);
    ylabel('航向 (度)', 'FontSize', 11);
    title('(d) 航向', 'FontSize', 12, 'FontWeight', 'bold');
    
    % 2.5 误差直方图
    subplot(3,2,5);
    histogram(station_error, 30, 'FaceColor', [0.3 0.7 1], ...
        'EdgeColor', 'k', 'FaceAlpha', 0.7);
    hold on;
    xline(station_radius, 'r--', 'LineWidth', 2);
    grid on; box on;
    xlabel('位置误差 (m)', 'FontSize', 11);
    ylabel('频次', 'FontSize', 11);
    title('(e) 误差分布', 'FontSize', 12, 'FontWeight', 'bold');
    legend(sprintf('阈值 (%.1fm)', station_radius), 'Location', 'best');
    
    % 2.6 统计信息
    subplot(3,2,6);
    axis off;
    
    error_max = max(station_error);
    error_mean = mean(station_error);
    error_std = std(station_error);
    error_rms = sqrt(mean(station_error.^2));
    violation_rate = sum(station_error > station_radius) / length(station_error) * 100;
    
    stats_text = sprintf([...
        '镇定性能统计:\n\n' ...
        '最大误差:     %.3f m\n' ...
        '平均误差:     %.3f m\n' ...
        '标准差:       %.3f m\n' ...
        'RMS误差:      %.3f m\n\n' ...
        '阈值:         %.1f m\n' ...
        '超出率:       %.1f%%\n\n' ...
        '持续时间:     %.1f s'...
        ], error_max, error_mean, error_std, error_rms, ...
           station_radius, violation_rate, ...
           station_time(end) - station_time(1));
    
    text(0.1, 0.5, stats_text, 'FontSize', 11, ...
        'FontName', 'Courier', 'VerticalAlignment', 'middle');
    title('(f) 性能指标', 'FontSize', 12, 'FontWeight', 'bold');
end

%% 图3: 自适应参数演化
figure('Name', '自适应参数演化', 'NumberTitle', 'off', 'Position', [200 200 1200 800]);

% 3.1 自适应参数
subplot(2,2,1);
hold on; grid on; box on;

plot(time, theta_x, 'b-', 'LineWidth', 1.5, 'DisplayName', '\theta_x (surge)');
plot(time, theta_y, 'r-', 'LineWidth', 1.5, 'DisplayName', '\theta_y (sway)');
plot(time, theta_psi, 'g-', 'LineWidth', 1.5, 'DisplayName', '\theta_\psi (yaw)');
xlabel('时间 (s)', 'FontSize', 12);
ylabel('参数估计', 'FontSize', 12);
title('(a) 自适应参数 vs 时间', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best');

% 3.2 参数幅值
subplot(2,2,2);
param_magnitude = sqrt(theta_x.^2 + theta_y.^2 + theta_psi.^2);
plot(time, param_magnitude, 'k-', 'LineWidth', 1.5);
grid on; box on;
xlabel('时间 (s)', 'FontSize', 12);
ylabel('||\theta||', 'FontSize', 12);
title('(b) 参数向量幅值', 'FontSize', 13, 'FontWeight', 'bold');

% 3.3 航向跟踪
subplot(2,2,3);
hold on; grid on; box on;

plot(time, psi, 'b-', 'LineWidth', 1.5, 'DisplayName', '实际 \psi');
plot(time, los_psi_d, 'r--', 'LineWidth', 1.5, 'DisplayName', '期望 \psi_d (LOS)');
xlabel('时间 (s)', 'FontSize', 12);
ylabel('航向 (度)', 'FontSize', 12);
title('(c) 航向跟踪', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best');

% 3.4 速度
subplot(2,2,4);
hold on; grid on; box on;

U = sqrt(u.^2 + v.^2);
plot(time, u, 'b-', 'LineWidth', 1.5, 'DisplayName', 'u (surge)');
plot(time, v, 'r-', 'LineWidth', 1.5, 'DisplayName', 'v (sway)');
plot(time, U, 'g-', 'LineWidth', 2, 'DisplayName', 'U (速度)');
xlabel('时间 (s)', 'FontSize', 12);
ylabel('速度 (m/s)', 'FontSize', 12);
title('(d) 速度分量', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best');

%% 图4: 额外分析（可选）
figure('Name', '额外分析', 'NumberTitle', 'off', 'Position', [250 250 1200 600]);

% 4.1 转向速度
subplot(1,2,1);
plot(time, r, 'b-', 'LineWidth', 1.5);
grid on; box on;
xlabel('时间 (s)', 'FontSize', 12);
ylabel('转向速度 (度/s)', 'FontSize', 12);
title('转向速度', 'FontSize', 13, 'FontWeight', 'bold');

% 4.2 速度矢量幅值
subplot(1,2,2);
U = sqrt(u.^2 + v.^2);
plot(time, U, 'b-', 'LineWidth', 2);
grid on; box on;
xlabel('时间 (s)', 'FontSize', 12);
ylabel('速度 (m/s)', 'FontSize', 12);
title('速度幅值', 'FontSize', 13, 'FontWeight', 'bold');



%% 打印统计摘要
fprintf('\n=======================================================\n');
fprintf('任务统计摘要:\n');
fprintf('=======================================================\n');
fprintf('仿真时间:        %.1f 秒\n', time(end));
fprintf('数据点数:        %d\n', length(time));
fprintf('最终位置:        (%.2f, %.2f) m\n', x(end), y(end));
fprintf('最终航向:        %.1f 度\n', psi(end));

if ~isempty(station_error)
    fprintf('\n镇定性能:\n');
    fprintf('  最大误差:      %.3f m\n', max(station_error));
    fprintf('  平均误差:      %.3f m\n', mean(station_error));
    fprintf('  RMS误差:       %.3f m\n', sqrt(mean(station_error.^2)));
    fprintf('  超出阈值率:    %.1f%%\n', ...
        sum(station_error > station_radius) / length(station_error) * 100);
end

fprintf('=======================================================\n\n');