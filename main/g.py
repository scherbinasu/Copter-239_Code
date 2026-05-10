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

MIN_DIST_M = 0.1          # минимальное расстояние между соединяемыми точками (метры)

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
    dtheta = np.abs(theta[:, np.newaxis] - theta[np.newaxis, :])
    r_i = distances[:, np.newaxis]
    r_j = distances[np.newaxis, :]
    dist_mat = np.sqrt(r_i**2 + r_j**2 - 2 * r_i * r_j * np.cos(dtheta))
    return dist_mat


def build_proximity_graph(dist_mat, merge_dist):
    """
    Строит граф по матрице расстояний: ребро между i и j, если dist <= merge_dist.
    Возвращает список рёбер (i, j).
    """
    N = dist_mat.shape[0]
    # Используем верхний треугольник без диагонали
    i, j = np.triu_indices(N, k=1)
    mask = dist_mat[i, j] <= merge_dist
    edges = list(zip(i[mask], j[mask]))
    return edges


def find_connected_components(num_points, edges):
    """ (та же функция, что и раньше) """
    parent = list(range(num_points))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in edges:
        union(i, j)

    comp_dict = {}
    for idx in range(num_points):
        root = find(idx)
        comp_dict.setdefault(root, []).append(idx)

    return list(comp_dict.values())


def build_contours(components, points_xy):
    contours = []
    for comp in components:
        if len(comp) < 2:
            pt = points_xy[comp[0]]
            contours.append(np.array([[pt]], dtype=np.int32))
        else:
            pts = points_xy[comp]
            hull = cv2.convexHull(pts)
            # убираем замыкающую точку, если она есть
            if len(hull) > 1 and np.array_equal(hull[0], hull[-1]):
                hull = hull[:-1]
            contours.append(hull)
    return contours




# ---------- парсер лога ----------
def parse_line(line):
    line = line.strip()
    data, t = line.split(b' ')
    t = float(t.decode('utf-8'))
    data = pickle.loads(base64.b64decode(data))
    return data, t
def draw_contours_as_lines(img_bgr, contours):
    """
    Рисует контуры как незамкнутые ломаные (без замыкания последней и первой точек).
    Одиночные точки не рисуются (или можно нарисовать кружками, если нужно).
    """
    for cnt in contours:
        if cnt.shape[0] < 2:
            # Одиночная точка — рисуем маленький кружок (опционально)
            if cnt.shape[0] == 1:
                cv2.circle(img_bgr, tuple(cnt[0][0]), 2, (0, 255, 255), -1)
            continue
        # Рисуем ломаную без замыкания
        cv2.polylines(img_bgr, [cnt], isClosed=False, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

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
            img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

            if len(pts_xy) >= 2:
                # матрица расстояний
                dist_mat = compute_distance_matrix(angles, distances)

                MERGE_DIST_M = 0.1  # порог для объединения в объект (можно настроить)
                edges = build_proximity_graph(dist_mat, MERGE_DIST_M)
                components = find_connected_components(len(pts_xy), edges)

                # строим выпуклые оболочки (контуры)
                contours = build_contours(components, pts_xy)

                # отрисовываем контуры поверх скана
                draw_contours_as_lines(img_bgr, contours)

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