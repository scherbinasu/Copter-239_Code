import cv2
import numpy as np
import pickle
import base64

# ---------- константы ----------
SCAN_WINDOW_NAME = "Lidar Scan"
SCAN_IMAGE_SIZE = 800
SCAN_MAP_SIZE_M = 20.0
SCAN_PIXELS_PER_METER = SCAN_IMAGE_SIZE / SCAN_MAP_SIZE_M
SCAN_CENTER = (SCAN_IMAGE_SIZE // 2, SCAN_IMAGE_SIZE // 2)

MIN_DIST_M = 0.2          # минимальное расстояние между соединяемыми точками (метры)
cv2.draw

# ---------- рабочие функции ----------
def get_cartesian_points(scan):
    """
    Возвращает кортеж:
        points_xy : np.ndarray (N,2) пиксельных координат валидных точек
        angles    : np.ndarray (N,)   углов в градусах (0..360)
        distances : np.ndarray (N,)   дистанций в метрах
    Точки с dist < 0.1 отбрасываются.
    """
    if len(scan) == 0:
        return np.empty((0, 2)), np.array([]), np.array([])

    angles = (scan['angle'] + 180) % 360.0
    dists = scan['distance']
    valid = dists >= 0.1
    if not np.any(valid):
        return np.empty((0, 2)), np.array([]), np.array([])

    angles = angles[valid]
    dists = dists[valid]

    # сортировка по углу
    sort_idx = np.argsort(angles)
    angles = angles[sort_idx]
    dists = dists[sort_idx]

    theta = np.deg2rad(angles)
    x_m = dists * np.cos(theta)
    y_m = dists * np.sin(theta)

    x_px = SCAN_CENTER[0] + (x_m * SCAN_PIXELS_PER_METER)
    y_px = SCAN_CENTER[1] + (y_m * SCAN_PIXELS_PER_METER)
    x_idx = np.round(x_px).astype(int)
    y_idx = np.round(y_px).astype(int)

    # отсекаем за границами изображения
    in_bounds = (x_idx >= 0) & (x_idx < SCAN_IMAGE_SIZE) & (y_idx >= 0) & (y_idx < SCAN_IMAGE_SIZE)
    x_idx = x_idx[in_bounds]
    y_idx = y_idx[in_bounds]
    angles = angles[in_bounds]
    dists = dists[in_bounds]

    return np.column_stack([x_idx, y_idx]), angles, dists


def draw_scan_hsv(img, scan):
    """Рисует цветной круг лидара (HSV) на изображении img."""
    if len(scan) == 0:
        return

    angles = (scan['angle'] + 180) % 360.0
    dists = scan['distance']
    intensities = scan['intensity']
    valid = dists >= 0.1
    angles = angles[valid]
    dists = dists[valid]
    intensities = intensities[valid]

    theta = np.deg2rad(angles)
    x_m = dists * np.cos(theta)
    y_m = dists * np.sin(theta)

    x_px = SCAN_CENTER[0] + (x_m * SCAN_PIXELS_PER_METER)
    y_px = SCAN_CENTER[1] + (y_m * SCAN_PIXELS_PER_METER)
    x_idx = np.round(x_px).astype(int)
    y_idx = np.round(y_px).astype(int)

    in_bounds = (x_idx >= 0) & (x_idx < SCAN_IMAGE_SIZE) & (y_idx >= 0) & (y_idx < SCAN_IMAGE_SIZE)
    x_idx = x_idx[in_bounds]
    y_idx = y_idx[in_bounds]
    angles_f = angles[in_bounds]
    intensities_f = intensities[in_bounds]

    h = (angles_f / 360.0 * 179).astype(np.uint8)
    max_int = np.max(intensities_f) if len(intensities_f) > 0 else 1
    s = np.clip((intensities_f / max_int) * 255, 0, 255).astype(np.uint8)
    v = np.full_like(h, 255, dtype=np.uint8)

    img[y_idx, x_idx] = np.stack([h, s, v], axis=1)

    # метка центра
    x_center, y_center = SCAN_CENTER
    half = int(SCAN_PIXELS_PER_METER * 0.2)
    img[y_center - half: y_center + half, x_center - half:x_center + half] = (255, 255, 255)


def compute_distance_matrix(angles_deg, distances):
    """
    Вычисляет полную матрицу попарных расстояний (N x N) между лучами.
    angles_deg – в градусах, distances – в метрах.
    Возвращает np.ndarray формы (N, N).
    """
    theta = np.deg2rad(angles_deg)
    N = len(distances)
    # разность углов
    dtheta = np.abs(theta[:, np.newaxis] - theta[np.newaxis, :])
    # расстояния
    r_i = distances[:, np.newaxis]
    r_j = distances[np.newaxis, :]
    dist_mat = np.sqrt(r_i**2 + r_j**2 - 2 * r_i * r_j * np.cos(dtheta))
    return dist_mat


def build_matching_edges(dist_mat, min_dist):
    """
    Жадно строит паросочетание: каждая точка соединяется не более одного раза.
    Возвращает список кортежей (i, j) – индексы соединённых точек.
    """
    N = dist_mat.shape[0]
    used = np.zeros(N, dtype=bool)
    edges = []

    # Чтобы результат не зависел от порядка, можно сортировать точки по расстоянию до центра
    # или просто идти подряд. Идём подряд.
    for i in range(N):
        if used[i]:
            continue
        # маскируем: исключаем саму себя, уже использованные точки, и расстояния <= min_dist
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        mask[used] = False
        mask[dist_mat[i] >= min_dist] = False
        if not np.any(mask):
            continue
        j = np.argmin(dist_mat[i][mask])          # индекс в маскированном массиве
        j_global = np.arange(N)[mask][j]          # реальный индекс точки
        edges.append((i, j_global))
        used[i] = True
        used[j_global] = True
    return edges


def draw_edges(img_bgr, points_xy, edges):
    """Рисует отрезки между точками на BGR-изображении."""
    for i, j in edges:
        pt1 = tuple(points_xy[i])
        pt2 = tuple(points_xy[j])
        cv2.line(img_bgr, pt1, pt2, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)


# ---------- парсер лога ----------
def parse_line(line):
    line = line.strip()
    data, t = line.split(b' ')
    t = float(t.decode('utf-8'))
    data = pickle.loads(base64.b64decode(data))
    return data, t


def main():
    t = 0
    with open('lidar_log.txt', 'rb') as f:
        for line in f.readlines():
            data, new_t = parse_line(line)
            key = cv2.waitKey(round((new_t - t) * 1000))

            # ----- рендеринг скана -----
            img_hsv = np.zeros((SCAN_IMAGE_SIZE, SCAN_IMAGE_SIZE, 3), dtype=np.uint8)
            draw_scan_hsv(img_hsv, data)

            # получаем координаты точек
            pts_xy, angles, distances = get_cartesian_points(data)
            if len(pts_xy) > 2:
                # матрица расстояний
                dist_mat = compute_distance_matrix(angles, distances)

                # строим рёбра
                edges = build_matching_edges(dist_mat, MIN_DIST_M)

                # рисуем линии поверх скана
                img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
                draw_edges(img_bgr, pts_xy, edges)
            else:
                img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

            cv2.imshow(SCAN_WINDOW_NAME, img_bgr)

            if key == 27:   # Esc
                return

            # частота кадров (для информации)
            if new_t != t:
                print(f"fps: {1/(new_t - t):.1f}")
            t = new_t

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()