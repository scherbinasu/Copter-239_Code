import time, cv2
import traceback
import numpy as np
from scipy.spatial import cKDTree
from robot.control.lidar.ms200k.oradar_lidar import LidarReader
from control.web.webGUI import WebGUI

# ---------- Настройки лидар-визуализации (ваши) ----------
WINDOW_NAME = "Lidar Scan"
IMAGE_SIZE = 800
MAP_SIZE_M = 20.0
PIXELS_PER_METER = IMAGE_SIZE / MAP_SIZE_M
CENTER = (IMAGE_SIZE // 2, IMAGE_SIZE // 2)
BACKGROUND_COLOR = (0, 0, 0)

# ---------- Настройки SLAM ----------
MAP_RESOLUTION = 0.05          # метр/пиксель
MAP_SIZE_METERS = 40.0         # 40x40 м
MAP_PIXELS = int(MAP_SIZE_METERS / MAP_RESOLUTION)
MAP_CENTER = MAP_PIXELS // 2   # начало глобальной системы координат в центре карты
OCCUPIED_THRESH = 128
FREE_DECAY = 3                 # уменьшение вероятности для свободных лучей
OCCUPIED_INC = 15              # увеличение для занятых точек
MAX_RANGE = 5.0               # максимальная дальность для обновления карты (м)

# ---------- Функции SLAM ----------
def world_to_map(x, y):
    """Перевод мировых координат (м) в индексы occupancy grid."""
    px = int(MAP_CENTER + x / MAP_RESOLUTION)
    py = int(MAP_CENTER - y / MAP_RESOLUTION)   # ось Y направлена вверх, но в изображении вниз
    return px, py

def map_to_world(px, py):
    """Обратное преобразование."""
    wx = (px - MAP_CENTER) * MAP_RESOLUTION
    wy = (MAP_CENTER - py) * MAP_RESOLUTION
    return wx, wy

def bresenham_line(x0, y0, x1, y1):
    """Алгоритм Брезенхема для получения всех пикселей на отрезке (целочисленные координаты)."""
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            if x0 == x1:
                break
            err += dy
            x0 += sx
        if e2 <= dx:
            if y0 == y1:
                break
            err += dx
            y0 += sy
    return points

def update_map(grid, robot_pose, scan):
    """
    Обновляет occupancy grid новым сканом.
    robot_pose: (x, y, yaw) в мировых координатах.
    scan: структурированный numpy массив с полями angle, distance.
    """
    # Координаты робота на карте
    rx_px, ry_px = world_to_map(robot_pose[0], robot_pose[1])
    angles = (scan['angle'] + robot_pose[2]) % 360.0   # глобальный угол луча
    dists = scan['distance']
    valid = (dists > 0.1) & (dists < MAX_RANGE)
    angles = angles[valid]
    dists = dists[valid]

    # Для каждого луча
    for i in range(len(angles)):
        angle_rad = np.deg2rad(angles[i])
        d = dists[i]
        # Глобальные координаты конечной точки
        ex = robot_pose[0] + d * np.cos(angle_rad)
        ey = robot_pose[1] + d * np.sin(angle_rad)
        ex_px, ey_px = world_to_map(ex, ey)

        # Рисуем линию от робота до точки (свободное пространство)
        for (px, py) in bresenham_line(rx_px, ry_px, ex_px, ey_px):
            if 0 <= px < MAP_PIXELS and 0 <= py < MAP_PIXELS:
                # Уменьшаем вероятность занятости (свободно)
                grid[py, px] = max(0, grid[py, px] - FREE_DECAY)

        # Отмечаем конечную точку как занятую
        if 0 <= ex_px < MAP_PIXELS and 0 <= ey_px < MAP_PIXELS:
            grid[ey_px, ex_px] = min(255, grid[ey_px, ex_px] + OCCUPIED_INC)

def icp_scan_matching(prev_scan, curr_scan, max_iterations=20, tolerance=1e-4):
    """
    Упрощённый ICP для двух облаков точек (массивы Nx2).
    Возвращает (dx, dy, dtheta) — приращение позы от prev к curr.
    """
    if len(prev_scan) < 10 or len(curr_scan) < 10:
        return 0.0, 0.0, 0.0

    # Преобразуем curr_scan в numpy массив [x, y]
    angles_curr = np.deg2rad(curr_scan['angle'])
    dists_curr = curr_scan['distance']
    valid_c = dists_curr > 0.1
    P = np.column_stack([
        dists_curr[valid_c] * np.cos(angles_curr[valid_c]),
        dists_curr[valid_c] * np.sin(angles_curr[valid_c])
    ])

    angles_prev = np.deg2rad(prev_scan['angle'])
    dists_prev = prev_scan['distance']
    valid_p = dists_prev > 0.1
    Q = np.column_stack([
        dists_prev[valid_p] * np.cos(angles_prev[valid_p]),
        dists_prev[valid_p] * np.sin(angles_prev[valid_p])
    ])

    if len(P) < 10 or len(Q) < 10:
        return 0.0, 0.0, 0.0

    # Начальное приближение: нулевое смещение
    R = np.eye(2)
    t = np.zeros(2)
    prev_error = float('inf')

    for it in range(max_iterations):
        # Поиск ближайших точек (Q -> P)
        tree = cKDTree(P)
        dists, idx = tree.query(Q)
        # Отбираем только хорошие соответствия (расстояние < 2 м)
        good = dists < 2.0
        if np.sum(good) < 5:
            break
        Q_good = Q[good]
        P_good = P[idx[good]]

        # Вычисляем оптимальное вращение и сдвиг (метод наименьших квадратов)
        centroid_Q = np.mean(Q_good, axis=0)
        centroid_P = np.mean(P_good, axis=0)
        Q_centered = Q_good - centroid_Q
        P_centered = P_good - centroid_P
        H = Q_centered.T @ P_centered
        U, _, Vt = np.linalg.svd(H)
        R_new = Vt.T @ U.T
        t_new = centroid_P - R_new @ centroid_Q

        # Обновляем Q
        Q = (R_new @ Q.T).T + t_new
        # Накапливаем трансформацию
        t = t + R @ t_new
        R = R_new @ R

        # Ошибка
        curr_error = np.mean(dists[good])
        if abs(prev_error - curr_error) < tolerance:
            break
        prev_error = curr_error

    dtheta = np.arctan2(R[1, 0], R[0, 0])
    return t[0], t[1], dtheta

# ---------- Вспомогательные функции отрисовки (ваши) ----------
def get_contour_points(scan):
    if len(scan) == 0:
        return np.empty((0, 2), dtype=np.int32)
    angles = scan['angle'] % 360.0
    dists = scan['distance']
    valid = dists >= 0.1
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.int32)
    angles = angles[valid]
    dists = dists[valid]
    sort_idx = np.argsort(angles)
    angles = angles[sort_idx]
    dists = dists[sort_idx]
    theta = np.deg2rad(angles)
    x_m = dists * np.cos(theta)
    y_m = dists * np.sin(theta)
    x_px = CENTER[0] + (x_m * PIXELS_PER_METER)
    y_px = CENTER[1] + (y_m * PIXELS_PER_METER)
    x_idx = np.round(x_px).astype(int)
    y_idx = np.round(y_px).astype(int)
    in_bounds = (x_idx >= 0) & (x_idx < IMAGE_SIZE) & (y_idx >= 0) & (y_idx < IMAGE_SIZE)
    x_idx = x_idx[in_bounds]
    y_idx = y_idx[in_bounds]
    return np.column_stack([x_idx, y_idx]).astype(np.int32)

def draw_scan_hsv(img, scan):
    if len(scan) == 0:
        return
    angles = (scan['angle'] + 0) % 360.0
    dists = scan['distance']
    intensities = scan['intensity']
    valid = dists >= 0.1
    angles = angles[valid]
    dists = dists[valid]
    intensities = intensities[valid]
    theta = np.deg2rad(angles)
    x_m = dists * np.cos(theta)
    y_m = dists * np.sin(theta)
    x_px = CENTER[0] + (x_m * PIXELS_PER_METER)
    y_px = CENTER[1] + (y_m * PIXELS_PER_METER)
    x_idx = np.round(x_px).astype(int)
    y_idx = np.round(y_px).astype(int)
    in_bounds = (x_idx >= 0) & (x_idx < IMAGE_SIZE) & (y_idx >= 0) & (y_idx < IMAGE_SIZE)
    x_idx = x_idx[in_bounds]
    y_idx = y_idx[in_bounds]
    angles_f = angles[in_bounds]
    intensities_f = intensities[in_bounds]
    h = (angles_f / 360.0 * 179).astype(np.uint8)
    max_int = np.max(intensities_f) if len(intensities_f) > 0 else 1
    s = np.clip((intensities_f / max_int) * 255, 0, 255).astype(np.uint8)
    v = np.full_like(h, 255, dtype=np.uint8)
    img[y_idx, x_idx] = np.stack([h, s, v], axis=1)

def main():
    PORT = '/dev/ttyUSB0'
    BAUDRATE = 230400

    lidar = LidarReader(port=PORT, baudrate=BAUDRATE, timeout_ms=500)
    try:
        lidar.start()
        print("Лидар запущен. Для выхода нажмите 'q' или ESC.")

        # Инициализация SLAM
        slam_map = np.zeros((MAP_PIXELS, MAP_PIXELS), dtype=np.uint8)
        robot_pose = np.array([0.0, 0.0, 0.0])   # x, y, yaw (в радианах)
        prev_scan = None

        while True:
            # Основное окно с точками и контуром
            frame_hsv = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            scan = lidar.get_scan()

            if scan is not None and len(scan) > 0:
                # Оценка смещения через ICP (если есть предыдущий скан)
                if prev_scan is not None:
                    dx, dy, dtheta = icp_scan_matching(prev_scan, scan)
                    # Обновляем позу (простая модель движения)
                    robot_pose[0] += dx * np.cos(robot_pose[2]) - dy * np.sin(robot_pose[2])
                    robot_pose[1] += dx * np.sin(robot_pose[2]) + dy * np.cos(robot_pose[2])
                    robot_pose[2] += dtheta
                prev_scan = scan.copy()

                # Обновление глобальной карты
                update_map(slam_map, robot_pose, scan)

                # Рисуем цветные точки и контур
                draw_scan_hsv(frame_hsv, scan)
                contour = get_contour_points(scan)

            # Конвертируем HSV -> BGR
            frame_bgr = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)

            if scan is not None and len(contour) >= 3:
                cv2.polylines(frame_bgr, [contour], isClosed=True,
                              color=(255, 255, 255), thickness=2)

            # Визуализация карты SLAM
            map_display = cv2.cvtColor(slam_map, cv2.COLOR_GRAY2BGR)
            # Рисуем робота на карте
            rx_px, ry_px = world_to_map(robot_pose[0], robot_pose[1])
            # Направление
            heading_x = rx_px + int(15 * np.cos(robot_pose[2]))
            heading_y = ry_px - int(15 * np.sin(robot_pose[2]))   # из-за инверсии Y
            cv2.circle(map_display, (rx_px, ry_px), 5, (0, 0, 255), -1)  # красная точка
            cv2.line(map_display, (rx_px, ry_px), (heading_x, heading_y), (0, 255, 0), 2)  # зелёная линия направления
            # Масштабируем карту для удобного просмотра (например, 600x600)
            map_display = cv2.resize(map_display, (600, 600))

            gui.imshow(WINDOW_NAME, frame_bgr)
            gui.imshow("SLAM Map", map_display)
            gui.imshow(WINDOW_NAME + ' test', frame_bgr)   # оставим, как было

            key = gui.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        print("\nОстановка по Ctrl+C.")
    except Exception as e:
        traceback.print_exc()
    finally:
        lidar.stop()
        gui.destroyAllWindows()

if __name__ == "__main__":
    gui = WebGUI(host='0.0.0.0', port=5000)
    gui.start()
    main()