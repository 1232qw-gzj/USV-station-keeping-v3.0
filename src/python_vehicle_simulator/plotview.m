%% realtime_animation_pure.m
% MATLAB实时动画 - 纯动画版本（无子图）
%
% 功能：只显示船的运动轨迹，不显示误差、模式等子图
%
% 使用方法:
%   1. 确保有 simulation_data.mat 文件
%   2. 运行此脚本: realtime_animation_pure
%   3. 观看动画

clear; clc; close all;

%% 加载数据
fprintf('正在加载仿真数据...\n');
data = load('simulation_data.mat');

% 转换数据类型
time = double(data.time);
x = double(data.x);
y = double(data.y);
psi = double(data.psi);
mode = double(data.mode);
waypoints = double(data.waypoints);
station_radius = double(data.station_radius);

fprintf('✅ 数据加载完成 (%d 帧)\n', length(time));

%% 动画参数
speed_factor = 10;  % 播放速度（1=实时，10=10倍速）
skip_frames = 5;    % 每隔几帧显示一次

%% 创建图形窗口
fig = figure('Name', 'Otter Animation', 'NumberTitle', 'off', ...
    'Position', [200 100 1000 800], 'Color', 'w');

hold on; grid on; box on;
axis equal;
xlabel('East (m)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('North (m)', 'FontSize', 14, 'FontWeight', 'bold');
title('Otter Station Keeping - Real-time Animation', ...
    'FontSize', 16, 'FontWeight', 'bold');

%% 绘制航点和路径
for i = 1:size(waypoints, 1)
    % 航点标记
    plot(waypoints(i,2), waypoints(i,1), 'gs', ...
        'MarkerSize', 16, 'MarkerFaceColor', 'g', 'LineWidth', 2.5);
    
    % 航点标签
    text(waypoints(i,2)+2, waypoints(i,1)+2, sprintf('WP%d', i-1), ...
        'FontSize', 12, 'FontWeight', 'bold', ...
        'BackgroundColor', 'yellow', 'EdgeColor', 'black', 'Margin', 3);
    
    % 误差圆（只在任务点）
    if i > 1 && i < size(waypoints, 1)
        theta = linspace(0, 2*pi, 100);
        xc = waypoints(i,2) + station_radius * cos(theta);
        yc = waypoints(i,1) + station_radius * sin(theta);
        plot(xc, yc, 'g--', 'LineWidth', 2.5, 'Color', [0 0.7 0]);
    end
end

% 路径连线
for i = 1:size(waypoints,1)-1
    plot([waypoints(i,2) waypoints(i+1,2)], ...
         [waypoints(i,1) waypoints(i+1,1)], ...
         '--', 'LineWidth', 1.5, 'Color', [0.6 0.8 0.6]);
end

%% 初始化动态元素

% 轨迹线（分模式着色）
h_traj_guidance = plot(NaN, NaN, 'b-', 'LineWidth', 2.5, ...
    'DisplayName', 'Guidance');
h_traj_station = plot(NaN, NaN, 'g-', 'LineWidth', 2.5, ...
    'DisplayName', 'Station Keeping');
h_traj_emergency = plot(NaN, NaN, 'r-', 'LineWidth', 2.5, ...
    'DisplayName', 'Emergency');

% 镇定点散点
h_station = scatter(NaN, NaN, 30, [0 1 0], 'filled', ...
    'MarkerFaceAlpha', 0.4, 'DisplayName', 'Station Points');

% 当前位置
h_current = plot(NaN, NaN, 'r*', 'MarkerSize', 30, 'LineWidth', 3, ...
    'DisplayName', 'Current Position');

% 船体（圆圈）
h_ship = plot(NaN, NaN, 'ro', 'MarkerSize', 18, ...
    'MarkerFaceColor', 'r', 'LineWidth', 2.5);

% 航向线
h_heading = plot(NaN, NaN, 'r-', 'LineWidth', 4);

% 图例
legend('Location', 'best', 'FontSize', 11);

%% 动画循环
fprintf('\n开始播放动画...\n');
fprintf('播放速度: %dx\n', speed_factor);
fprintf('跳帧数: %d\n', skip_frames);
fprintf('按 Ctrl+C 停止\n\n');

% 存储不同模式的轨迹
guidance_x = [];
guidance_y = [];
station_x = [];
station_y = [];
emergency_x = [];
emergency_y = [];
station_pts_x = [];
station_pts_y = [];

for i = 1:skip_frames:length(time)
    % 根据模式分类存储轨迹点
    if mode(i) == 0  % Guidance
        guidance_x = [guidance_x; x(i)];
        guidance_y = [guidance_y; y(i)];
    elseif mode(i) == 1  % Station Keeping
        station_x = [station_x; x(i)];
        station_y = [station_y; y(i)];
        station_pts_x = [station_pts_x; x(i)];
        station_pts_y = [station_pts_y; y(i)];
    elseif mode(i) == 2  % Emergency
        emergency_x = [emergency_x; x(i)];
        emergency_y = [emergency_y; y(i)];
    end
    
    % 更新轨迹（分模式着色）
    if ~isempty(guidance_y)
        set(h_traj_guidance, 'XData', guidance_y, 'YData', guidance_x);
    end
    if ~isempty(station_y)
        set(h_traj_station, 'XData', station_y, 'YData', station_x);
    end
    if ~isempty(emergency_y)
        set(h_traj_emergency, 'XData', emergency_y, 'YData', emergency_x);
    end
    
    % 更新镇定点散点
    if ~isempty(station_pts_x)
        set(h_station, 'XData', station_pts_y, 'YData', station_pts_x);
    end
    
    % 更新当前位置
    set(h_current, 'XData', y(i), 'YData', x(i));
    
    % 更新船体位置
    set(h_ship, 'XData', y(i), 'YData', x(i));
    
    % 更新航向线
    psi_rad = psi(i) * pi / 180;
    L = 4.5;  % 航向线长度
    heading_x = [y(i), y(i) + L*sin(psi_rad)];
    heading_y = [x(i), x(i) + L*cos(psi_rad)];
    set(h_heading, 'XData', heading_x, 'YData', heading_y);
    
    % 更新标题显示时间和模式
    mode_str = {'Guidance', 'Station Keeping', 'Emergency'};
    mode_idx = mode(i) + 1;
    if mode_idx < 1 || mode_idx > 3
        mode_idx = 1;
    end
    
    title(sprintf('Time: %.1f / %.1f s  |  Mode: %s  |  Progress: %.0f%%', ...
        time(i), time(end), mode_str{mode_idx}, i/length(time)*100), ...
        'FontSize', 16, 'FontWeight', 'bold');
    
    % 刷新显示
    drawnow;
    
    % 控制播放速度
    if i > skip_frames
        dt = time(i) - time(i-skip_frames);
        pause(dt / speed_factor);
    end
    
    % 进度提示
    if mod(i, 200) == 0
        fprintf('进度: %.1f%% | 时间: %.1f/%.1f s | 模式: %s\n', ...
            i/length(time)*100, time(i), time(end), mode_str{mode_idx});
    end
end

fprintf('\n✅ 动画播放完成！\n');

%% 最终统计
fprintf('\n========================================\n');
fprintf('           仿真统计\n');
fprintf('========================================\n');
fprintf('总时间: %.2f 秒\n', time(end));
fprintf('总帧数: %d\n', length(time));
fprintf('最终位置: (%.2f, %.2f) m\n', x(end), y(end));
fprintf('最终航向: %.1f 度\n', psi(end));
fprintf('========================================\n\n');