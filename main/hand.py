import math, cv2
import traceback

from pygments.formatters import img

from robot.control.abstractions import *
from robot.robots import *

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


def cluster_to_pixels(cluster, drone):
    """Переводит кластер (N,2) в метрах в пиксельные координаты изображения."""
    x_px = drone.SCAN_CENTER[0] + cluster[:, 0] * drone.SCAN_PIXELS_PER_METER
    y_px = drone.SCAN_CENTER[1] + cluster[:, 1] * drone.SCAN_PIXELS_PER_METER
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


reg = PID_regulator(1, 0, 0, 1)
def get_abc(x1, y1, x2, y2):
    """Convert line to Ax + By + C = 0"""
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    return a, b, c


def get_intersection(p1, p2, p3, p4):
    """Get intersection of (x1, y1) -> (x2, y2) and
    (x3, y3) -> (x4, y4)"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    # Ax + By + C = 0
    a1, b1, c1 = get_abc(x1, y1, x2, y2)
    a2, b2, c2 = get_abc(x3, y3, x4, y4)
    # [ A1 B1 ] [ X ] = [ -C1 ]
    # [ A2 B2 ] [ Y ]   [ -C2 ]

    # Find X and Y
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-12:  # Parallel
        return None, None
    inv_det = 1.0 / det
    # [  B2 -B1 ] [ -C1 ] : det = [ X ]
    # [ -A2  A1 ] [ -C2 ]         [ Y ]
    x = inv_det * (b1 * c2 - b2 * c1)
    y = inv_det * (a2 * c1 - a1 * c2)

    return x, y


async def main():
    print(222)
    fd_count = len(os.listdir('/proc/self/fd'))
    print(f"Открыто fd перед стартом drone: {fd_count}")
    async with Drone() as drone:
        print(111)
        try:
            while True:
                scan = drone.get_scan()
                wall_angle, wall_angle_raw, nearest = angle_wall(scan, 180, 45)
                clusters = cluster_lidar_points_v2(scan, 0.2, 5)
                img = np.zeros((drone.SCAN_IMAGE_SIZE, drone.SCAN_IMAGE_SIZE, 3), dtype=np.uint8)
                for cluster in clusters:

                    cl_px = cluster_to_pixels(cluster, drone).astype(np.int32)
                    cv2.polylines(img, [cl_px], False, (255, 0, 255), 2)
                    left, right = get_cluster_extremes(cluster)
                    left_px = cluster_to_pixels(left.reshape(1, -1), drone).astype(np.int32)[0]
                    right_px = cluster_to_pixels(right.reshape(1, -1), drone).astype(np.int32)[0]
                    cv2.circle(img, tuple(left_px), 5, (255, 0, 0), -1)
                    cv2.circle(img, tuple(right_px), 5, (255, 0, 0), -1)
                drone.web.imshow("clusters", img)
                print(f'wall_angle: {wall_angle:.2f}, wall_angle_raw: {wall_angle_raw:.2f}, nearest: {nearest:.2f}')
        except Exception as e:
            traceback.print_exc()


if __name__ == '__main__':
    selector = selectors.PollSelector()
    loop = asyncio.SelectorEventLoop(selector)
    asyncio.set_event_loop(loop)
    asyncio.run(main())
