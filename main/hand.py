import base64
import pickle
import math
import cv2
import numpy as np
from mavsdk.offboard import OffboardError
import cv2
from robot.robots import *
import traceback




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


def cluster_lidar_points_v2(scan, distance_threshold=0.5, min_cluster_size=3,
                            target_point=None, angle_tol=2.0, dist_tol=0.1):
    """
    Возвращает:
      clusters : список np.array (k, 2) с полярными координатами (angle [град], distance [м])
      target_cluster_idx : индекс кластера с ближайшей к target_point точкой,
                           или индекс ближайшего кластера (если target задана, но точка не попала
                           ни в один кластер), или -1 если кластеров нет.
    """
    angles = scan['angle']
    dists = scan['distance']
    valid = dists >= 0.1
    ang = angles[valid]
    dst = dists[valid]
    n = len(ang)

    if n < min_cluster_size:
        return [], -1

    # Поиск индекса ближайшей точки к целевой
    target_idx = -1
    if target_point is not None:
        t_ang, t_dst = target_point
        in_angle = np.abs(ang - t_ang) <= angle_tol
        in_dist = np.abs(dst - t_dst) <= dist_tol
        candidates = np.where(in_angle & in_dist)[0]
        if len(candidates) > 0:
            d_ang = np.abs(ang[candidates] - t_ang) / angle_tol
            d_dst = np.abs(dst[candidates] - t_dst) / dist_tol
            metric = d_ang**2 + d_dst**2
            target_idx = candidates[np.argmin(metric)]
        else:
            # Без допусков — просто ближайшая точка во всём скане
            d_ang_all = np.abs(ang - t_ang)
            d_ang_all = np.minimum(d_ang_all, 360.0 - d_ang_all)  # циклический угол
            d_dst_all = np.abs(dst - t_dst)
            metric_all = d_ang_all**2 + d_dst_all**2
            target_idx = np.argmin(metric_all)

    cos_a = np.cos(np.deg2rad(ang))
    sin_a = np.sin(np.deg2rad(ang))

    clusters = []
    target_cluster_idx = -1
    start = 0
    for i in range(1, n):
        d_sq = (dst[i]**2 + dst[i-1]**2
                - 2 * dst[i] * dst[i-1] * (cos_a[i]*cos_a[i-1] + sin_a[i]*sin_a[i-1]))
        if d_sq >= distance_threshold**2:
            if i - start >= min_cluster_size:
                cluster_pts = np.column_stack((ang[start:i], dst[start:i]))
                clusters.append(cluster_pts)
                if target_idx != -1 and start <= target_idx < i:
                    target_cluster_idx = len(clusters) - 1
            start = i

    if n - start >= min_cluster_size:
        cluster_pts = np.column_stack((ang[start:n], dst[start:n]))
        clusters.append(cluster_pts)
        if target_idx != -1 and start <= target_idx < n:
            target_cluster_idx = len(clusters) - 1

    # ---------- Итеративное объединение соседних кластеров ----------
    changed = True
    while changed and len(clusters) >= 2:
        changed = False
        new_clusters = []
        i = 0
        while i < len(clusters):
            if i == len(clusters) - 1:
                new_clusters.append(clusters[i])
                break
            # Расстояние между концом текущего и началом следующего
            last_ang, last_dst = clusters[i][-1]
            first_ang, first_dst = clusters[i+1][0]
            d_sq = (last_dst**2 + first_dst**2
                    - 2 * last_dst * first_dst * np.cos(np.deg2rad(last_ang - first_ang)))
            if d_sq < distance_threshold**2:
                merged = np.vstack((clusters[i], clusters[i+1]))
                new_clusters.append(merged)
                if target_cluster_idx == i or target_cluster_idx == i+1:
                    target_cluster_idx = len(new_clusters) - 1
                elif target_cluster_idx > i+1:
                    target_cluster_idx -= 1
                i += 2
                changed = True
            else:
                new_clusters.append(clusters[i])
                i += 1
        clusters = new_clusters

    # ---------- Fallback: если target не попал в кластеры, берём ближайший ----------
    if target_cluster_idx == -1 and len(clusters) > 0 and target_point is not None:
        t_ang, t_dst = target_point
        best_dist = float('inf')
        for i, cl in enumerate(clusters):
            d_ang = np.abs(cl[:, 0] - t_ang)
            d_ang = np.minimum(d_ang, 360.0 - d_ang)
            d_dst = np.abs(cl[:, 1] - t_dst)
            min_metric = np.min(d_ang**2 + d_dst**2)
            if min_metric < best_dist:
                best_dist = min_metric
                target_cluster_idx = i

    return clusters, target_cluster_idx




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

def find_triangle_angles(p1, p2):
    # Вычисляем длины катетов
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    angle1 = math.degrees(math.atan2(dy, dx))  # Угол при точке (x1, y1)
    return (angle1+180) % 180
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

import numpy as np

def angle_of_line_polar_cosine(r1, theta1, r2, theta2,
                               angle_unit='deg', output_unit='deg'):
    """
    Угол наклона прямой, соединяющей две точки в полярных координатах,
    относительно оси X. Вычисляется через теорему косинусов без явного
    преобразования точек в декартову систему.

    Параметры
    ----------
    r1, theta1 : float
        Полярные координаты первой точки.
    r2, theta2 : float
        Полярные координаты второй точки.
    angle_unit : str
        'deg' или 'rad' – единицы входных углов.
    output_unit : str
        'deg' или 'rad' – единицы выходного угла.

    Возвращает
    -------
    angle : float или None
        Угол прямой (отрезка от точки 1 к точке 2) в диапазоне (-180°, 180°] / (-π, π].
        Если точки совпадают, возвращает None.
    """
    # 1. Переводим входные углы в радианы
    if angle_unit == 'deg':
        t1 = np.deg2rad(theta1)
        t2 = np.deg2rad(theta2)
    else:
        t1, t2 = theta1, theta2

    # 2. Разность углов, приведённая к (-π, π]
    delta_theta = (t2 - t1 + np.pi) % (2 * np.pi) - np.pi

    # 3. Расстояние d между точками по теореме косинусов
    # d = sqrt(r1^2 + r2^2 - 2 * r1 * r2 * cos(delta_theta))
    d = np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * np.cos(delta_theta))
    if np.isclose(d, 0):
        return None

    # 4. Угол α между радиус-вектором r1 и отрезком (всегда неотрицательный)
    # По теореме косинусов: cos(α) = (d^2 + r1^2 - r2^2) / (2 * d * r1)
    cos_alpha = (d**2 + r1**2 - r2**2) / (2 * d * r1)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)  # всегда [0, π]

    # 5. Направление отрезка: к θ1 добавляем α со знаком delta_theta
    # Если delta_theta > 0, точка 2 "правее" по углу -> прибавляем α
    # Если delta_theta < 0, вычитаем α
    angle_rad = t1 + np.sign(delta_theta) * alpha

    # 6. Нормализация в (-π, π]
    angle_rad = (angle_rad + np.pi) % (2 * np.pi) - np.pi

    # 7. Вывод
    if output_unit == 'deg':
        return np.rad2deg(angle_rad)
    return angle_rad
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


async def main():

    old_vector = (90, 1)
    direction = 0
    async with Drone() as drone:
        def cluster_to_pixels(cluster):
            """Переводит кластер (N,2) в метрах в пиксельные координаты изображения."""
            x_px = drone.SCAN_CENTER[0] - cluster[:, 1] * drone.SCAN_PIXELS_PER_METER * np.sin(cluster[:, 0] * np.pi / 180)
            y_px = drone.SCAN_CENTER[1] - cluster[:, 1] * drone.SCAN_PIXELS_PER_METER * np.cos(cluster[:, 0] * np.pi / 180)
            return np.column_stack((x_px, y_px))
        try:
            while 1:
                key = drone.web.waitKey(100)
                scan = drone.lidar.get_scan()
                img = np.zeros((drone.SCAN_IMAGE_SIZE, drone.SCAN_IMAGE_SIZE, 3), dtype=np.uint8)
                drone.draw_scan_hsv(img, scan)
                clusters, idx = cluster_lidar_points_v2(scan, 0.5, 5, old_vector)
                clx = clusters[idx]
                print(clx[:, 1])
                vector = clx[np.argmin(clx[:, 1])]
                print(vector)
                old_vector = vector
                diff = np.abs(direction - vector[0]+90) % 360
                dist_a = np.minimum(diff, 360 - diff)
                direction = np.where(dist_a <= 90, (vector[0]+270)%360, vector[0]+90)
                cv2.line(img, (int(drone.SCAN_CENTER[0]-vector[1]*drone.SCAN_PIXELS_PER_METER*np.sin(math.radians(direction))),
                               int(drone.SCAN_CENTER[1]-vector[1]*drone.SCAN_PIXELS_PER_METER*np.cos(math.radians(direction)))), drone.SCAN_CENTER, (255, 50, 77), 2)
                cv2.line(img, (int(drone.SCAN_CENTER[0] - vector[1] * drone.SCAN_PIXELS_PER_METER * np.sin(math.radians(vector[0]))),
                               int(drone.SCAN_CENTER[1] - vector[1] * drone.SCAN_PIXELS_PER_METER * np.cos(math.radians(vector[0])))),
                         drone.SCAN_CENTER, (46, 255, 77), 2)
                for i, cl in enumerate(clusters):
                    c = cluster_to_pixels(cl)
                    # print(c)
                    cv2.polylines(img, [c.astype(int)], False, (0, 0, 255), 1)
                    cv2.circle(img, c[0].astype(int), 1, (120, 120, 0), -1)
                    cv2.circle(img, c[-1].astype(int), 1, (120, 120, 0), -1)

                drone.web.imshow('scan_poly', img)
                if key == 27:
                    return
        except Exception as ex:
            traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())

