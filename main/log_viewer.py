import base64
import pickle
import math
import cv2
import numpy as np

# ---------------------- Константы ----------------------
SCAN_WINDOW_NAME = "Lidar Scan"
SCAN_IMAGE_SIZE = 800
SCAN_MAP_SIZE_M = 20.0
SCAN_PIXELS_PER_METER = SCAN_IMAGE_SIZE / SCAN_MAP_SIZE_M
SCAN_CENTER = (SCAN_IMAGE_SIZE // 2, SCAN_IMAGE_SIZE // 2)

# ------------------ Функции преобразования координат ------------------
def polar_to_cartesian(r, angle_deg):
    """
    Преобразование полярных координат (r, α) в декартовы (x, y) в метрах.
    α – угол в градусах, измеренный от оси X против часовой стрелки
    (0° – вперёд, 90° – влево).
    Возвращает:
        x = r * cos(α)   (вперёд)
        y = -r * sin(α)  (вправо)
    """
    theta = np.deg2rad(angle_deg)
    x = r * np.cos(theta)
    y = -r * np.sin(theta)
    return x, y

def cartesian_to_pixel(x, y, center=SCAN_CENTER, scale=SCAN_PIXELS_PER_METER):
    """
    Преобразование декартовых координат (x, y) в метрах в пиксельные
    координаты изображения OpenCV.
    col = center_col + y * scale
    row = center_row - x * scale
    """
    col = center[0] + y * scale
    row = center[1] - x * scale
    return col, row

# ------------------ Вспомогательные геометрические функции ------------------
def groups_first_last(groups):
    """
    Принимает список массивов (каждый формы (k, D)) и возвращает
    np.ndarray формы (G, 2, D) из первых и последних точек каждой группы.
    """
    if not groups:
        return np.empty((0, 2, 0))

    D = next(g.shape[1] for g in groups if g.size > 0)
    G = len(groups)
    result = np.full((G, 2, D), np.nan)

    for i, g in enumerate(groups):
        if len(g) > 0:
            result[i, 0] = g[0]
            result[i, 1] = g[-1]
    return result

def nearest_from_other_group(groups):
    """
    Для каждой точки в groups (формы (G,2,D)) ищет ближайшую точку,
    принадлежащую другой группе.
    """
    G, _, D = groups.shape
    points = groups.reshape(-1, D)
    N = 2 * G

    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)

    group_ids = np.arange(N) // 2
    same_group = group_ids[:, np.newaxis] == group_ids[np.newaxis, :]
    dists_masked = np.where(same_group, np.inf, dists)

    nearest_idx = np.argmin(dists_masked, axis=1)
    distances = dists[np.arange(N), nearest_idx]

    return nearest_idx, distances

def farthest_within_limit(pts, a, n):
    """
    Возвращает точку из pts, максимально удалённую от a, но не дальше n.
    pts – (N, D), a – (D,), n – максимальное расстояние.
    """
    pts = np.asarray(pts, dtype=float)
    a = np.asarray(a, dtype=float)

    dists = np.linalg.norm(pts - a, axis=1)
    valid = dists <= n
    if not np.any(valid):
        return None, None, None

    valid_indices = np.where(valid)[0]
    idx_of_max = np.argmax(dists[valid_indices])
    best_idx = valid_indices[idx_of_max]
    return best_idx, pts[best_idx], dists[best_idx]

def solve_sas(b, c, alpha):
    """Решение треугольника по двум сторонам и углу (SAS)."""
    a = math.sqrt(b * b + c * c - 2 * b * c * math.cos(alpha))
    beta = math.acos((a * a + c * c - b * b) / (2 * a * c))
    gamma = math.pi - alpha - beta
    return a, beta, gamma

def intersection_of_regression_lines(points_arr1, points_arr2, tol=1e-10):
    """Точка пересечения двух прямых, построенных по МНК."""
    x1, y1 = points_arr1[:, 0], points_arr1[:, 1]
    x2, y2 = points_arr2[:, 0], points_arr2[:, 1]
    a1, b1 = np.polyfit(x1, y1, 1)
    a2, b2 = np.polyfit(x2, y2, 1)
    if np.isclose(a1, a2, atol=tol):
        return None
    x_int = (b2 - b1) / (a1 - a2)
    y_int = a1 * x_int + b1
    return np.array([x_int, y_int])

def find_triangle_angles(p1, p2):
    """Угол прямой (p1->p2) в диапазоне [0,180)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return (angle + 180) % 180

# ------------ Функции визуализации и обработки скана ------------
def get_contour_points(scan):
    """
    Возвращает массив (N,2) пиксельных координат замкнутого контура,
    упорядоченных по углу сканирования. Точки с dist < 0.1 отбрасываются.
    """
    if len(scan) == 0:
        return np.empty((0, 2), dtype=np.int32)

    angles = scan['angle']                # без сдвига!
    dists = scan['distance']
    valid = dists >= 0.1
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.int32)

    angles = angles[valid]
    dists = dists[valid]

    # сортировка по углу
    sort_idx = np.argsort(angles)
    angles = angles[sort_idx]
    dists = dists[sort_idx]

    # полярные → декартовы (метры)
    x_m, y_m = polar_to_cartesian(dists, angles)

    # декартовы → пиксели
    col, row = cartesian_to_pixel(x_m, y_m, SCAN_CENTER, SCAN_PIXELS_PER_METER)
    x_idx = np.round(col).astype(int)
    y_idx = np.round(row).astype(int)

    # оставляем точки внутри изображения
    in_bounds = (x_idx >= 0) & (x_idx < SCAN_IMAGE_SIZE) & (y_idx >= 0) & (y_idx < SCAN_IMAGE_SIZE)
    return np.column_stack([x_idx[in_bounds], y_idx[in_bounds]]).astype(np.int32)

def draw_scan_hsv(img, scan):
    """
    Векторизованное рисование скана в HSV (цветной круг).
    img – пустое HSV-изображение (np.uint8).
    """
    if len(scan) == 0:
        return

    angles = scan['angle']                # без сдвига
    dists = scan['distance']
    intensities = scan['intensity']
    valid = dists >= 0.1
    angles = angles[valid]
    dists = dists[valid]
    intensities = intensities[valid]

    # полярные → декартовы
    x_m, y_m = polar_to_cartesian(dists, angles)

    # декартовы → пиксели
    col, row = cartesian_to_pixel(x_m, y_m, SCAN_CENTER, SCAN_PIXELS_PER_METER)
    x_idx = np.round(col).astype(int)
    y_idx = np.round(row).astype(int)

    in_bounds = (x_idx >= 0) & (x_idx < SCAN_IMAGE_SIZE) & (y_idx >= 0) & (y_idx < SCAN_IMAGE_SIZE)
    x_idx = x_idx[in_bounds]
    y_idx = y_idx[in_bounds]
    angles_f = angles[in_bounds]
    intensities_f = intensities[in_bounds]

    # цвет: H – угол, S – нормированная интенсивность
    h = (angles_f / 360.0 * 179).astype(np.uint8)
    max_int = np.max(intensities_f) if len(intensities_f) > 0 else 1
    s = np.clip((intensities_f / max_int) * 255, 0, 255).astype(np.uint8)
    v = np.full_like(h, 255, dtype=np.uint8)

    img[y_idx, x_idx] = np.stack([h, s, v], axis=1)

    # отметка центра
    x_center, y_center = SCAN_CENTER
    half = int(SCAN_PIXELS_PER_METER * 0.2)
    img[y_center - half: y_center + half, x_center - half: x_center + half] = (255, 255, 255)

def get_scan(scan):
    img = np.zeros((SCAN_IMAGE_SIZE, SCAN_IMAGE_SIZE, 3), dtype=np.uint8)
    draw_scan_hsv(img, scan)
    cv2.imshow('scan', img)
    return scan, img

def parse_line(line):
    line = line.strip()
    data, t = line.split(b' ')
    t = float(t.decode('utf-8'))
    data = pickle.loads(base64.b64decode(data))
    return data, t

# ---------------------- Кластеризация ----------------------
def cluster_lidar_points_v2(scan, distance_threshold=0.5, min_cluster_size=3):
    """
    Кластеризует точки лидара. Возвращает список массивов (k,2) с декартовыми
    координатами в МЕТРАХ (x вперёд, y вправо) относительно центра сканирования.
    """
    angles = scan['angle']                # без сдвига!
    dists = scan['distance']
    valid = dists >= 0.01
    ang = angles[valid]
    dst = dists[valid]
    n = len(ang)

    if n < min_cluster_size:
        return []

    clusters = []
    start = 0
    sin_a = np.sin(np.deg2rad(ang))
    cos_a = np.cos(np.deg2rad(ang))

    for i in range(1, n):
        # расстояние между лучами по теореме косинусов
        d_sq = (dst[i] ** 2 + dst[i - 1] ** 2
                - 2 * dst[i] * dst[i - 1] * (sin_a[i] * sin_a[i - 1] + cos_a[i] * cos_a[i - 1]))
        if d_sq >= distance_threshold ** 2:
            if i - start >= min_cluster_size:
                # декартовы координаты: x = r·cos(α), y = -r·sin(α)
                pts = np.column_stack((
                    dst[start:i] * cos_a[start:i],
                    -dst[start:i] * sin_a[start:i]
                ))
                clusters.append(pts)
            start = i

    # последний кластер
    if n - start >= min_cluster_size:
        pts = np.column_stack((
            dst[start:n] * cos_a[start:n],
            -dst[start:n] * sin_a[start:n]
        ))
        clusters.append(pts)

    return clusters

def cluster_to_pixels(cluster):
    """
    Переводит кластер (N,2) в метрах (x,y) в пиксельные координаты изображения.
    """
    x_m = cluster[:, 0]
    y_m = cluster[:, 1]
    col, row = cartesian_to_pixel(x_m, y_m, SCAN_CENTER, SCAN_PIXELS_PER_METER)
    return np.column_stack((col, row))

def get_cluster_extremes(cluster):
    """
    Возвращает крайние точки кластера по углу с учётом перехода через 0°.
    cluster – (N,2) в метрах (x,y).
    """
    if len(cluster) == 0:
        return None, None
    if len(cluster) == 1:
        return cluster[0].copy(), cluster[0].copy()
    if len(cluster) == 2:
        return cluster[0].copy(), cluster[1].copy()

    angles = np.rad2deg(np.arctan2(cluster[:, 1], cluster[:, 0])) % 360.0
    min_ang, max_ang = angles.min(), angles.max()

    if max_ang - min_ang > 180.0:
        adjusted = np.where(angles < 180.0, angles + 360.0, angles)
    else:
        adjusted = angles

    idx_min = np.argmin(adjusted)
    idx_max = np.argmax(adjusted)
    return cluster[idx_min].copy(), cluster[idx_max].copy()

# ---------------------- Оценка угла стены ----------------------
def angle_wall(scan, angle, range_angle):
    """Вычисляет угол стены относительно заданного направления."""
    idx_center = np.abs(scan['angle'] - angle).argmin()
    nearest = scan['distance'][idx_center]
    nearest_angle = scan['angle'][idx_center]

    left_mask = (scan['angle'] >= angle - range_angle / 2) & (scan['angle'] < angle)
    left_dists = scan['distance'][left_mask]
    if len(left_dists) == 0:
        left_med, left_angle_med = np.nan, np.nan
    else:
        left_med = np.median(left_dists)
        left_angles = scan['angle'][left_mask]
        idx_left = np.argmin(np.abs(left_dists - left_med))
        left_angle_med = left_angles[idx_left]

    right_mask = (scan['angle'] > angle) & (scan['angle'] <= angle + range_angle / 2)
    right_dists = scan['distance'][right_mask]
    if len(right_dists) == 0:
        right_med, right_angle_med = np.nan, np.nan
    else:
        right_med = np.median(right_dists)
        right_angles = scan['angle'][right_mask]
        idx_right = np.argmin(np.abs(right_dists - right_med))
        right_angle_med = right_angles[idx_right]

    a, beta_rad, gamma_rad = solve_sas(
        float(left_med), float(right_med),
        float(math.radians(abs(left_angle_med - right_angle_med)))
    )
    wall_angle_raw = math.degrees(beta_rad) - math.degrees(gamma_rad)
    wall_angle = wall_angle_raw + angle - nearest_angle
    return wall_angle, wall_angle_raw, nearest


def point_line_distance(point, line_start, line_end):
    """
    Расстояние от точки до линии.
    """
    line = line_end - line_start
    line_len = np.linalg.norm(line)

    if line_len < 1e-8:
        return np.linalg.norm(point - line_start)

    return np.abs(np.cross(line, point - line_start)) / line_len


def split_and_merge(points, threshold=0.05, min_points=2):
    """
    Split-and-merge для 2D lidar points.

    Args:
        points: np.ndarray shape (N, 2)
        threshold: max distance to line before split
        min_points: minimum points in segment

    Returns:
        list of segments [(start_point, end_point), ...]
    """

    segments = []

    def recursive_split(pts):
        if len(pts) < min_points:
            return

        start = pts[0]
        end = pts[-1]

        # Ищем точку с максимальным отклонением
        max_dist = -1
        split_idx = -1

        for i in range(1, len(pts) - 1):
            dist = point_line_distance(pts[i], start, end)

            if dist > max_dist:
                max_dist = dist
                split_idx = i

        # Если линия плохая — делим
        if max_dist > threshold:
            recursive_split(pts[:split_idx + 1])
            recursive_split(pts[split_idx:])
        else:
            segments.append((start, end))

    recursive_split(points)

    return segments


def render_segments(img, segments, cartesian_to_pixel,
                    color=(0, 255, 0),
                    thickness=2,
                    draw_points=True):
    """
    Рисует сегменты split-and-merge на изображении.

    Args:
        img: OpenCV image
        segments: list[(start, end)]
        cartesian_to_pixel: function(x, y) -> (col, row)
        color: line color
        thickness: line thickness
        draw_points: draw segment endpoints
    """

    for start, end in segments:
        x1, y1 = start
        x2, y2 = end

        col1, row1 = cartesian_to_pixel(x1, y1)
        col2, row2 = cartesian_to_pixel(x2, y2)

        cv2.line(
            img,
            (int(col1), int(row1)),
            (int(col2), int(row2)),
            color,
            thickness
        )

        if draw_points:
            cv2.circle(img, (int(col1), int(row1)), 4, (0, 0, 255), -1)
            cv2.circle(img, (int(col2), int(row2)), 4, (255, 0, 0), -1)


# ---------------------- Главный цикл ----------------------
def main():
    with open('lidar_log.txt', 'rb') as f:
        for line in f.readlines():
            data, new_t = parse_line(line)
            key = cv2.waitKey(10)
            scan, img = get_scan(data)
            clusters = cluster_lidar_points_v2(scan, 0.2, 5)
            print(clusters)
            for i in clusters:
                for px in i:
                    col, row = cartesian_to_pixel(px[0], px[1])
                    row, col = round(row), round(col)
                    img[row, col] = (0, 0, 255)
                segments = split_and_merge(i, 0.2, 2)
                render_segments(img, segments, cartesian_to_pixel)
            cv2.imshow('scan_poly', img)
            if key == 27:
                return

if __name__ == '__main__':
    main()
