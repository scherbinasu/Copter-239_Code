import cv2
import numpy as np
import pickle
import base64

SCAN_WINDOW_NAME = "Lidar Scan"
SCAN_IMAGE_SIZE = 800
SCAN_MAP_SIZE_M = 20.0
SCAN_PIXELS_PER_METER = SCAN_IMAGE_SIZE / SCAN_MAP_SIZE_M
SCAN_CENTER = (SCAN_IMAGE_SIZE // 2, SCAN_IMAGE_SIZE // 2)


def get_contour_points(scan):
    """
    Возвращает массив (N,2) пиксельных координат, упорядоченных по углу,
    для построения замкнутого контура. Точки с dist < 0.1 отбрасываются.
    """
    if len(scan) == 0:
        return np.empty((0, 2), dtype=np.int32)

    angles = (scan['angle']+180) % 360.0
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
        cv2.imshow('scan', img)
    return scan, img


def parse_line(line):
    line = line.strip()
    data, t = line.split(b' ')
    t = float(t.decode('utf-8'))
    data = pickle.loads(base64.b64decode(data))
    return data, t


def cluster_lidar_points_v2(scan, distance_threshold=0.5, min_cluster_size=3):
    """
    Кластеризует точки лидара. Возвращает список массивов (k,2) с декартовыми
    координатами в МЕТРАХ (x, y) относительно центра сканирования.
    """
    angles = scan['angle']
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
    cos_a = np.cos(np.deg2rad(ang))
    sin_a = np.sin(np.deg2rad(ang))

    for i in range(1, n):
        # Расстояние между лучами по теореме косинусов
        d_sq = (dst[i] ** 2 + dst[i - 1] ** 2
                - 2 * dst[i] * dst[i - 1] * (cos_a[i] * cos_a[i - 1] + sin_a[i] * sin_a[i - 1]))
        if d_sq >= distance_threshold ** 2:
            if i - start >= min_cluster_size:
                # Декартовы координаты в метрах (без SCAN_CENTER)
                pts = np.column_stack((
                    dst[start:i] * cos_a[start:i],
                    dst[start:i] * sin_a[start:i]
                ))
                clusters.append(pts)
            start = i

    # Последний кластер
    if n - start >= min_cluster_size:
        pts = np.column_stack((
            dst[start:n] * cos_a[start:n],
            dst[start:n] * sin_a[start:n]
        ))
        clusters.append(pts)

    return clusters


def cluster_to_pixels(cluster):
    """Переводит кластер (N,2) в метрах в пиксельные координаты изображения."""
    x_px = SCAN_CENTER[0] + cluster[:, 0] * SCAN_PIXELS_PER_METER
    y_px = SCAN_CENTER[1] + cluster[:, 1] * SCAN_PIXELS_PER_METER
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


def main():
    t = 0
    with open('lidar_log.txt', 'rb') as f:
        for i in f.readlines():
            data, new_t = parse_line(i)
            key = cv2.waitKey(round((new_t - t) * 1000))
            scan, img = get_scan(data)
            clusters = cluster_lidar_points_v2(scan, distance_threshold=0.35, min_cluster_size=5)
            print(f"Найдено кластеров: {len(clusters)}")
            for idx, cl in enumerate(clusters):
                left, right = get_cluster_extremes(cl)
                print(f"  Кластер {idx}: размер {len(cl)}, "
                      f"левая граница (x={left[0]:.2f}м, y={left[1]:.2f}м), "
                      f"правая граница (x={right[0]:.2f}м, y={right[1]:.2f}м)")

                # Перевод в пиксели для рисования
                cl_px = cluster_to_pixels(cl).astype(np.int32)
                cv2.polylines(img, [cl_px], False, (255, 0, 255), 2)

                left_px = cluster_to_pixels(left.reshape(1, -1)).astype(np.int32)[0]
                right_px = cluster_to_pixels(right.reshape(1, -1)).astype(np.int32)[0]
                cv2.circle(img, tuple(left_px), 5, (255, 0, 0), -1)
                cv2.circle(img, tuple(right_px), 5, (255, 0, 0), -1)

            cv2.imshow('scan_poly', img)
            if key == 27:
                return
            print(f"FPS: {1/(new_t - t):.1f}")
            t = new_t


if __name__ == '__main__':
    main()