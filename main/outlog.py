import base64
import pickle
import math
import cv2
import numpy as np
from robot.control.web.webGUI import *
SCAN_WINDOW_NAME = "Lidar Scan"
SCAN_IMAGE_SIZE = 800
SCAN_MAP_SIZE_M = 20.0
SCAN_PIXELS_PER_METER = SCAN_IMAGE_SIZE / SCAN_MAP_SIZE_M
SCAN_CENTER = (SCAN_IMAGE_SIZE // 2, SCAN_IMAGE_SIZE // 2)

web = WebGUI(port=5000)
web.start()
def groups_first_last(groups):
    """
    Принимает список массивов (каждый формы (k, D)) и возвращает
    np.ndarray формы (G, 2, D) из первых и последних точек каждой группы.

    Параметры
    ----------
    groups : list of np.ndarray
        Каждый элемент — массив точек (k_i, D). Все D должны совпадать.

    Возвращает
    -------
    result : np.ndarray (G, 2, D)
        Для каждой группы: [первая_точка, последняя_точка].
        Если группа пуста, обе точки заполняются np.nan.
    """
    if not groups:
        return np.empty((0, 2, 0))

    # Определяем размерность точек по первой непустой группе
    D = next(g.shape[1] for g in groups if g.size > 0)
    G = len(groups)

    # Создаём выходной массив, заполненный NaN
    result = np.full((G, 2, D), np.nan)

    for i, g in enumerate(groups):
        if len(g) > 0:
            result[i, 0] = g[0]      # первая
            result[i, 1] = g[-1]     # последняя
        # если группа пуста — останутся nan

    return result
def nearest_from_other_group(groups):
    """
    Для каждой точки в groups (формы (G,2,D)) ищет ближайшую точку,
    принадлежащую другой группе.

    Параметры
    ----------
    groups : np.ndarray (G, 2, D)

    Возвращает
    -------
    nearest_idx : np.ndarray (2G,)
        Индексы ближайших точек в общем массиве points = groups.reshape(-1, D).
    distances : np.ndarray (2G,)
        Евклидовы расстояния до этих точек.
    """
    G, _, D = groups.shape
    points = groups.reshape(-1, D)       # (N, D), где N = 2G
    N = 2 * G

    # Попарные расстояния (векторизация через broadcasting)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]   # (N, N, D)
    dists = np.linalg.norm(diff, axis=2)                         # (N, N)

    # Маска: True, если точки i и j из одной группы
    group_ids = np.arange(N) // 2               # [0,0,1,1,2,2,...]
    same_group = group_ids[:, np.newaxis] == group_ids[np.newaxis, :]  # (N, N)

    # Исключаем себя и партнёра по группе, заменяя расстояние на ∞
    dists_masked = np.where(same_group, np.inf, dists)

    # Ближайший сосед
    nearest_idx = np.argmin(dists_masked, axis=1)  # (N,)
    distances = dists[np.arange(N), nearest_idx]   # из исходных dists

    return nearest_idx, distances
def farthest_within_limit(pts, a, n):
    """
    Возвращает точку из pts, максимально удалённую от a, но не дальше n.

    Параметры
    ----------
    pts : np.ndarray (N, D)
        Массив точек.
    a : array_like (D,)
        Опорная точка.
    n : float > 0
        Максимально допустимое расстояние.

    Возвращает
    -------
    index : int или None
    point : np.ndarray или None
    distance : float или None
        Если ни одна точка не удовлетворяет условию dist <= n, возвращаются None.
    """
    pts = np.asarray(pts, dtype=float)
    a = np.asarray(a, dtype=float)

    dists = np.linalg.norm(pts - a, axis=1)

    # Маска точек в пределах радиуса n
    valid = dists <= n
    if not np.any(valid):
        return None, None, None

    # Индексы подходящих точек
    valid_indices = np.where(valid)[0]
    # Ищем точку с максимальным расстоянием среди них
    idx_of_max = np.argmax(dists[valid_indices])
    best_idx = valid_indices[idx_of_max]

    return best_idx, pts[best_idx], dists[best_idx]

def get_contour_points(scan):
    """
    Возвращает массив (N,2) пиксельных координат, упорядоченных по углу,
    для построения замкнутого контура. Точки с dist < 0.1 отбрасываются.
    """
    if len(scan) == 0:
        return np.empty((0, 2), dtype=np.int32)

    angles = (scan['angle'] + 180) % 360.0
    dists = scan['distance']
    valid = dists >= 0.1
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.int32)

    angles = angles[valid]
    dists = dists[valid]

    # Сортировка по углу (как при сканировании)
    sort_idx = np.argsort(angles)
    angles = angles[sort_idx]
    dists = dists[sort_idx]

    # Преобразование в декартовы координаты (с зеркалированием, как в основном коде)
    theta = np.deg2rad(angles)
    x_m = dists * np.cos(theta)
    y_m = dists * np.sin(theta)

    x_px = SCAN_CENTER[0] + (x_m * SCAN_PIXELS_PER_METER)
    y_px = SCAN_CENTER[1] + (y_m * SCAN_PIXELS_PER_METER)
    x_idx = np.round(x_px).astype(int)
    y_idx = np.round(y_px).astype(int)

    # Оставляем только точки в пределах изображения
    in_bounds = (x_idx >= 0) & (x_idx < SCAN_IMAGE_SIZE) & (y_idx >= 0) & (y_idx < SCAN_IMAGE_SIZE)
    x_idx = x_idx[in_bounds]
    y_idx = y_idx[in_bounds]

    return np.column_stack([x_idx, y_idx]).astype(np.int32)


def draw_scan_hsv(img, scan):
    """
    Векторизованное рисование скана в HSV (цветной круг).
    img – пустое HSV-изображение (np.uint8).
    """
    if len(scan) == 0:
        return

    # --- фильтрация и преобразование координат ---
    angles = (scan['angle'] + 0) % 360.0  # поворот на 0° (оставили как было)
    dists = scan['distance']
    intensities = scan['intensity']
    valid = dists >= 0.1
    angles = angles[valid]
    dists = dists[valid]
    intensities = intensities[valid]

    # полярные -> декартовы (с уже имеющимся у вас зеркалированием)
    theta = np.deg2rad(angles)
    x_m = dists * np.cos(theta)
    y_m = dists * np.sin(theta)

    x_px = SCAN_CENTER[0] + (x_m * SCAN_PIXELS_PER_METER)
    y_px = SCAN_CENTER[1] + (y_m * SCAN_PIXELS_PER_METER)
    x_idx = np.round(x_px).astype(int)
    y_idx = np.round(y_px).astype(int)

    # точки в пределах изображения
    in_bounds = (x_idx >= 0) & (x_idx < SCAN_IMAGE_SIZE) & (y_idx >= 0) & (y_idx < SCAN_IMAGE_SIZE)
    x_idx = x_idx[in_bounds]
    y_idx = y_idx[in_bounds]
    angles_f = angles[in_bounds]
    intensities_f = intensities[in_bounds]

    # --- формирование цветов HSV ---
    # Hue: 0..179
    h = (angles_f / 360.0 * 179).astype(np.uint8)

    # Нормализация насыщенности по максимальной интенсивности текущего скана
    max_int = np.max(intensities_f) if len(intensities_f) > 0 else 1
    s = np.clip((intensities_f / max_int) * 255, 0, 255).astype(np.uint8)

    v = np.full_like(h, 255, dtype=np.uint8)  # постоянная яркость

    # назначаем пиксели в HSV-изображении
    img[y_idx, x_idx] = np.stack([h, s, v], axis=1)
    x_center, y_center = SCAN_CENTER
    half = int(SCAN_PIXELS_PER_METER * 0.2)
    img[y_center - half: y_center + half, x_center - half:x_center + half] = (255, 255, 255)


def get_scan(scan):
    img = np.zeros((SCAN_IMAGE_SIZE, SCAN_IMAGE_SIZE, 3), dtype=np.uint8)
    draw_scan_hsv(img, scan)
    if not img is None:
        web.imshow('scan', img)
    return scan, img


def parse_line(line):
    line = line.strip()
    data, t = line.split(b' ')
    t = float(t.decode('utf-8'))
    data = pickle.loads(base64.b64decode(data))
    return data, t


def intersection_of_regression_lines(points_arr1, points_arr2, tol=1e-10):
    """
    Вычисляет точку пересечения двух прямых, аппроксимирующих два набора точек
    методом наименьших квадратов (линейная регрессия).

    Параметры
    ----------
    points_arr1 : array
        Координаты точек первой линии.
    points_arr2 : array
        Координаты точек второй линии.
    tol : float, опционально
        Допуск для проверки равенства угловых коэффициентов (по умолчанию 1e-10).

    Возвращает
    -------
    result : tuple
        кортеж (x, y) – координаты пересечения.
        Если прямые параллельны и не совпадают, значение None.

    Пример
    -------
    > points1 = np.array([[1, 2.1], [2, 3.8], [3, 6.2], [4, 7.9], [5, 10.1]])
    > points2 = np.array([[1, 10.0], [2, 8.2], [3, 6.1], [4, 4.3], [5, 2.0]])
    > res = intersection_of_regression_lines(points1, points2)
    > print(res)
    """
    # Преобразуем входные данные в numpy массивы
    x1, y1 = points_arr1[:, 0], points_arr1[:, 1]
    x2, y2 = points_arr2[:, 0], points_arr2[:, 1]

    # --- 1. Линейная регрессия для каждой группы точек ---
    # np.polyfit(x, y, 1) возвращает [наклон, сдвиг]
    a1, b1 = np.polyfit(x1, y1, 1)
    a2, b2 = np.polyfit(x2, y2, 1)

    # --- 2. Проверка параллельности и совпадения ---
    if np.isclose(a1, a2, atol=tol):
        if np.isclose(b1, b2, atol=tol):
            # Прямые совпадают
            return None
        else:
            # Параллельны, но не совпадают
            return None

    # --- 3. Расчёт точки пересечения ---
    # Решаем уравнение a1*x + b1 = a2*x + b2
    x_int = (b2 - b1) / (a1 - a2)
    y_int = a1 * x_int + b1

    return np.array([x_int, y_int])


def cluster_lidar_points_v2(scan, distance_threshold=0.5, min_cluster_size=3):
    """
    Кластеризует точки лидара. Возвращает список массивов (k,2) с декартовыми
    координатами в МЕТРАХ (x, y) относительно центра сканирования.
    """
    angles = (scan['angle']+90) % 360.0
    dists = scan['distance']
    valid = dists >= 0.01
    ang = angles[valid]
    dst = dists[valid]
    n = len(ang)

    if n < min_cluster_size:
        return []

    clusters = []
    start = 0
    # Синусы и косинусы для всех углов
    sin_a = np.sin(np.deg2rad(ang))
    cos_a = np.cos(np.deg2rad(ang))

    for i in range(1, n):
        # Расстояние между лучами по теореме косинусов
        d_sq = (dst[i] ** 2 + dst[i - 1] ** 2
                - 2 * dst[i] * dst[i - 1] * (sin_a[i] * sin_a[i - 1] + cos_a[i] * cos_a[i - 1]))
        if d_sq >= distance_threshold ** 2:
            if i - start >= min_cluster_size:
                # Декартовы координаты в метрах (без SCAN_CENTER)
                pts = np.column_stack((
                    dst[start:i] * sin_a[start:i],
                    dst[start:i] * cos_a[start:i]
                ))
                clusters.append(pts)
            start = i

    # Последний кластер
    if n - start >= min_cluster_size:
        pts = np.column_stack((
            dst[start:n] * sin_a[start:n],
            dst[start:n] * cos_a[start:n]
        ))
        clusters.append(pts)

    return clusters


def cluster_to_pixels(cluster):
    """Переводит кластер (N,2) в метрах в пиксельные координаты изображения."""
    x_px = SCAN_CENTER[0] + cluster[:, 0] * SCAN_PIXELS_PER_METER
    y_px = SCAN_CENTER[1] - cluster[:, 1] * SCAN_PIXELS_PER_METER
    return np.column_stack((x_px, y_px))


def get_cluster_extremes(cluster):
    """
    Принимает кластер — np.array (N,2) декартовых координат (x, y) в метрах
    относительно лидара. Возвращает (leftmost_xy, rightmost_xy) — крайние
    точки по углу с корректной обработкой перехода через 0°.
    """
    if len(cluster) == 0:
        return None, None
    if len(cluster) == 1:
        return cluster[0].copy(), cluster[0].copy()
    if len(cluster) == 2:
        return cluster[0].copy(), cluster[1].copy()

    # Углы [0, 360) относительно лидара (x,y — метры)
    angles = np.rad2deg(np.arctan2(cluster[:, 1], cluster[:, 0])) % 360.0
    min_ang, max_ang = angles.min(), angles.max()

    # Циклический переход через 0°
    if max_ang - min_ang > 180.0:
        adjusted = np.where(angles < 180.0, angles + 360.0, angles)
    else:
        adjusted = angles

    idx_min = np.argmin(adjusted)
    idx_max = np.argmax(adjusted)
    return cluster[idx_min].copy(), cluster[idx_max].copy()


def solve_sas(b, c, alpha):
    """
    Решает треугольник по двум сторонам и углу между ними (SAS).

    Аргументы:
        b, c: длины известных сторон (float > 0)
        alpha: угол между b и c в радианах (0 < alpha < π)

    Возвращает:
        (a, beta, gamma): сторона a и углы beta (напротив b),
                          gamma (напротив c) в радианах.
    """
    # Сторона a по теореме косинусов
    a = math.sqrt(b * b + c * c - 2 * b * c * math.cos(alpha))

    # Угол beta по теореме косинусов (однозначно)
    beta = math.acos((a * a + c * c - b * b) / (2 * a * c))

    # Угол gamma из суммы углов треугольника
    gamma = math.pi - alpha - beta

    return a, beta, gamma


def angle_wall(scan, angle, range_angle):
    # центральный луч (ближайший к angle)
    idx_center = np.abs(scan['angle'] - angle).argmin()
    nearest = scan['distance'][idx_center]
    nearest_angle = scan['angle'][idx_center]  # фактический угол центра

    # левый сектор
    left_mask = (scan['angle'] >= angle - range_angle / 2) & (scan['angle'] < angle)
    left_dists = scan['distance'][left_mask]
    if len(left_dists) == 0:
        left_med, left_angle_med = np.nan, np.nan
    else:
        left_med = np.median(left_dists)
        # находим индекс в left_dists с расстоянием, ближайшим к медиане
        idx_left_in_masked = np.argmin(np.abs(left_dists - left_med))
        left_angles = scan['angle'][left_mask]
        left_angle_med = left_angles[idx_left_in_masked]

    # правый сектор
    right_mask = (scan['angle'] > angle) & (scan['angle'] <= angle + range_angle / 2)
    right_dists = scan['distance'][right_mask]
    if len(right_dists) == 0:
        right_med, right_angle_med = np.nan, np.nan
    else:
        right_med = np.median(right_dists)
        idx_right_in_masked = np.argmin(np.abs(right_dists - right_med))
        right_angles = scan['angle'][right_mask]
        right_angle_med = right_angles[idx_right_in_masked]
    # print(left_med, left_angle_med, right_med, right_angle_med)
    a, beta_rad, gamma_rad = solve_sas(float(left_med), float(right_med),
                                       float(math.radians(abs(left_angle_med - right_angle_med))))
    wall_angle_raw = math.degrees(beta_rad) - math.degrees(gamma_rad)
    wall_angle = wall_angle_raw + angle - nearest_angle
    return wall_angle, wall_angle_raw, nearest


def main():
    t = 0
    with open('lidar_log.txt', 'rb') as f:
        for i in f.readlines():
            data, new_t = parse_line(i)
            key = web.waitKey(100)
            scan, img = get_scan(data)
            wall_angle, wall_angle_raw, nearest = angle_wall(scan, 180, 45)
            clusters = cluster_lidar_points_v2(scan, 0.2, 5)

            edge = groups_first_last(clusters)
            nearest_idx, dists = nearest_from_other_group(edge)
            points = cluster_to_pixels(edge.reshape(-1, 2))

            for i in range(len(points)):
                if abs(dists[i]-0.95) < 0.1:
                    print(f"Точка {i} {points[i]} -> "
                          f"ближайшая {nearest_idx[i]} {points[nearest_idx[i]]}, "
                          f"расстояние = {dists[i]:.3f}")
                    cv2.line(img, points[i].astype(int), points[nearest_idx[i]].astype(int), (255, 0, 0), 2)
                    cv2.line(img, farthest_within_limit(cluster_to_pixels(clusters[i//2]), points[i], 0.9*SCAN_PIXELS_PER_METER)[1].astype(int), points[i].astype(int), (255, 0, 255), 2)
                    cv2.line(img, points[nearest_idx[i]].astype(int), farthest_within_limit(cluster_to_pixels(clusters[nearest_idx[i]//2]), points[nearest_idx[i]], 0.9*SCAN_PIXELS_PER_METER)[1].astype(int), (255, 0, 255), 2)
                # px_cl = cluster_to_pixels(cluster)
                # pts = np.array([px_cl[0], farthest_within_limit(px_cl, px_cl[0], 1*SCAN_PIXELS_PER_METER)[1], px_cl[-1]], dtype=np.int32)
                # cv2.polylines(bw, [pts], False, (255, 255, 255), 2)
            print(f'wall_angle: {wall_angle:.2f}, wall_angle_raw: {wall_angle_raw:.2f}, nearest: {nearest:.2f}')

            web.imshow('scan_poly', img)
            if key == 27:
                return
            print(f"FPS: {1 / (new_t - t+0.0000001):.1f}")
            t = new_t


if __name__ == '__main__':
    main()
