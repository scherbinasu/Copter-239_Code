import math
import traceback

from robot.control.abstractions import *
from robot.robots import *


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


async def main():
    async with Drone() as drone:
        try:
            while True:
                scan = drone.get_scan()
                a, beta_rad, gamma_rad = angle_wall(scan, 180, 45)

                print(f"Сторона a: {a:.3f}\tУгол β-γ:   {math.degrees(beta_rad) - math.degrees(gamma_rad):.2f}°")
        except Exception as e:
            traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())
