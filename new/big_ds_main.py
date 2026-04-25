import colors
import traceback
import cv2
from robot.findMask import FindMask
from robot.robot import AquaRobot
from robot.control.abstractions import *

# ----------------------------------------------------------------------
# Вспомогательные функции (без классов, только импортированные из abstractions)
# ----------------------------------------------------------------------

def _turn_to_gate(robot: AquaRobot, color, min_area: int = 500, rotation_speed: int = 30):
    """
    Вращает робота на месте, пока не будут обнаружены ворота заданного цвета.
    Ворота считаются найденными, если на изображении есть как минимум 2 контура
    и площадь второго контура больше min_area.
    """
    while True:
        img = robot.camera.get_frame()
        robot.gui.imshow('img', img)

        mask = FindMask(img)
        mask.normalize()
        mask.inRangeF(color)
        mask.findContours()
        mask.sortedContours()
        contours = mask.contours

        # Визуализация
        robot.video.write(cv2.drawContours(img, contours[:2], -1, (255, 0, 0), -1))

        if len(contours) >= 2 and cv2.contourArea(contours[1]) > min_area:
            robot.motor_left.set_motor(0)
            robot.motor_right.set_motor(0)
            break

        robot.motor_left.set_motor(rotation_speed)
        robot.motor_right.set_motor(-rotation_speed)


def _pass_through_gate(robot: AquaRobot, color, pid_yaw: PID_regulator,
                       pid_speed: PID_regulator, min_area: int = 500):
    """
    Проезжает через ворота, используя два PID-регулятора:
    - pid_yaw  → коррекция курса по горизонтальному смещению от центра ворот.
    - pid_speed → управление скоростью на основе расстояния между столбами ворот.
    """
    while True:
        img = robot.camera.get_frame()
        mask = FindMask(img)
        mask.normalize()
        mask.inRangeF(color)
        mask.findContours()
        mask.sortedContours()
        contours = mask.contours

        if len(contours) >= 2 and cv2.contourArea(contours[1]) > min_area:
            # Центры левого и правого столбов ворот
            cnt_left = FindMask(contours=[contours[0]]).getCenter()[0]   # Point
            cnt_right = FindMask(contours=[contours[1]]).getCenter()[0]  # Point
            gate_center = (cnt_left + cnt_right) / 2.0

            delta_yaw = (mask.cntr_frame - gate_center).x   # float
            delta_speed = abs((cnt_left - cnt_right).x)     # float

            u_yaw = pid_yaw(delta_yaw * 2)
            u_speed = pid_speed(delta_speed)

            robot.motor_left.set_motor(u_speed - u_yaw)
            robot.motor_right.set_motor(u_speed + u_yaw)

            # Визуализация
            vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            vis = cv2.drawContours(vis, contours[:2], -1, (255, 0, 0), -1)
            cv2.circle(vis, cnt_left.to_int(), 5, (0, 255, 0), 2)
            cv2.circle(vis, cnt_right.to_int(), 5, (0, 0, 255), 2)
            cv2.circle(vis, gate_center.to_int(), 5, (0, 255, 255), 2)
            robot.video.write(vis)
        else:
            robot.sleepV(2)
            robot.motor_left.set_motor(0)
            robot.motor_right.set_motor(0)
            break


def _get_best_contour_for_color(img, color):
    """
    Возвращает кортеж (контур, площадь, центр) для наибольшего контура заданного цвета.
    Если контуров нет — (None, 0, None).
    """
    mask = FindMask(img)
    mask.normalize()
    mask.inRangeF(color)
    mask.findContours()
    mask.sortedContours()
    contours = mask.contours
    if not contours:
        return None, 0, None

    cnt = contours[0]
    area = cv2.contourArea(cnt)
    center = FindMask(contours=[cnt]).getCenter()[0]   # Point
    return cnt, area, center


# ----------------------------------------------------------------------
# Основной сценарий
# ----------------------------------------------------------------------
try:
    robot = AquaRobot()
    robot.start(cv2=cv2)

    # Цвета портов (очередь выполнения)
    queue = [colors.Yellow, colors.Green, colors.Red]
    queue_names = ['Yellow', 'Green', 'Red']
    Orange = colors.Orange

    # PID-регуляторы (параметры взяты из исходного кода)
    pid_yaw_port   = PID_regulator(-0.04, 0, 0, 0)
    pid_speed_port = PID_regulator(0.0004, 0, 0, 90000)

    pid_yaw_gate   = PID_regulator(-0.08, 0, 0, 0)
    pid_speed_gate = PID_regulator(0.0004, 0, 0, 90000)

    # ---------- 1. Первый проезд через ворота ----------
    _turn_to_gate(robot, Orange)
    robot.sleepV(2)
    _pass_through_gate(robot, Orange, pid_yaw_gate, pid_speed_gate)
    robot.sleepV(2)

    # ---------- 2. Посещение всех портов ----------
    completed_indices = set()
    while len(completed_indices) < len(queue):
        frame = robot.camera.get_frame()

        best_idx = -1
        best_area = 0
        best_contour = None
        best_center = None

        # Находим самый большой контур среди ещё не посещённых цветов
        for idx, color in enumerate(queue):
            if idx in completed_indices:
                continue
            cnt, area, center = _get_best_contour_for_color(frame, color)
            if cnt is not None and area > best_area:
                best_area = area
                best_idx = idx
                best_contour = cnt
                best_center = center

        if best_idx != -1 and best_area > 500:
            current_color = queue[best_idx]
            print(f"Target port: {queue_names[best_idx]}")

            # Нижняя точка контура (по оси Y)
            min_y = max(best_contour, key=lambda p: p[0][1])[0][1]

            # Центр кадра (уже определён в mask)
            mask = FindMask(frame)
            frame_center = mask.cntr_frame   # Point

            # Если порт ещё далеко — едем вперёд с PID-коррекцией
            if min_y <= frame_center.y * 2 - 5:
                delta_x = (frame_center - best_center).x
                u_yaw = pid_yaw_port(delta_x)
                u_speed = pid_speed_port(best_area)

                robot.motor_left.set_motor(u_speed - u_yaw)
                robot.motor_right.set_motor(u_speed + u_yaw)

                # Визуализация
                vis = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                vis = cv2.drawContours(vis, [best_contour], -1, (255, 0, 0), -1)
                cv2.line(vis, (0, min_y), (820, min_y), (0, 0, 0), 3)
                robot.video.write(vis)
            else:
                # Достигли порта — отъезжаем назад и помечаем цвет как выполненный
                robot.sleepV(2)
                completed_indices.add(best_idx)

                # Небольшой отъезд назад
                robot.motor_left.set_motor(-35 + u_yaw * 3)
                robot.motor_right.set_motor(-35 - u_yaw * 3)

                # Пауза (короче для последнего порта)
                pause = 1 if len(completed_indices) == len(queue) else 3
                robot.sleepV(pause)

                robot.motor_left.set_motor(0)
                robot.motor_right.set_motor(0)
        else:
            # Ничего не видим — вращаемся в сторону, куда накопилась интегральная ошибка
            speed = 25
            reverse = -1 if pid_yaw_port.integral_err <= 0 else 1
            robot.motor_left.set_motor(speed * reverse)
            robot.motor_right.set_motor(-speed * reverse)

    # ---------- 3. Финальный проезд через ворота ----------
    _turn_to_gate(robot, Orange)
    robot.sleepV(2)
    _pass_through_gate(robot, Orange, pid_yaw_gate, pid_speed_gate)

except Exception:
    print("Произошла ошибка:")
    traceback.print_exc()