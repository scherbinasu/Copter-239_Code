import colors
import traceback
import cv2
from robot.findMask import FindMask
from robot.robot import AquaRobot
from robot.control.abstractions import *

# ---------- Вспомогательные функции ----------
def turn_to_gate(robot, color, min_contour_area=500, rotation_speed=30):
    """Вращение робота до обнаружения ворот заданного цвета."""
    while True:
        img = robot.camera.get_frame()
        robot.gui.imshow('img', img)
        mask = FindMask(img)
        mask.normalize()
        mask.inRangeF(color)
        mask.findContours()
        mask.sortedContours()
        contours = mask.contours

        robot.video.write(cv2.drawContours(img, contours[:2], -1, (255, 0, 0), -1))

        if len(contours) >= 2 and cv2.contourArea(contours[1]) > min_contour_area:
            robot.motor_left.set_motor(0)
            robot.motor_right.set_motor(0)
            break

        robot.motor_left.set_motor(rotation_speed)
        robot.motor_right.set_motor(-rotation_speed)

def pass_through_gate(robot, color, pid_yaw, pid_speed, min_contour_area=500):
    """Проезд через ворота с использованием PID-регуляторов."""
    while True:
        img = robot.camera.get_frame()
        mask = FindMask(img)
        mask.normalize()
        mask.inRangeF(color)
        mask.findContours()
        mask.sortedContours()
        contours = mask.contours

        if len(contours) >= 2 and cv2.contourArea(contours[1]) > min_contour_area:
            cnt1 = FindMask(contours=[contours[0]]).getCenter()[0]
            cnt2 = FindMask(contours=[contours[1]]).getCenter()[0]
            center = (cnt1 + cnt2) / 2

            delta_yaw = (mask.cntr_frame - center).x
            delta_speed = abs((cnt1 - cnt2).x)

            u_yaw = pid_yaw(delta_yaw * 2)
            u_speed = pid_speed(delta_speed)

            robot.motor_left.set_motor(u_speed - u_yaw)
            robot.motor_right.set_motor(u_speed + u_yaw)

            # Визуализация
            masked = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            masked = cv2.drawContours(masked, contours[:2], -1, (255, 0, 0), -1)
            cv2.circle(masked, cnt1.to_int(), 5, (0, 255, 0), 2)
            cv2.circle(masked, cnt2.to_int(), 5, (0, 0, 255), 2)
            cv2.circle(masked, center.to_int(), 5, (0, 255, 255), 2)
            robot.video.write(masked)
        else:
            robot.sleepV(2)
            robot.motor_left.set_motor(0)
            robot.motor_right.set_motor(0)
            break

def get_largest_contour_for_color(img, color):
    """Возвращает наибольший контур заданного цвета на изображении и его центр."""
    mask = FindMask(img)
    mask.normalize()
    mask.inRangeF(color)
    mask.findContours()
    mask.sortedContours()
    contours = mask.contours
    if not contours:
        return None, None, None
    cnt = contours[0]
    area = cv2.contourArea(cnt)
    center = FindMask(contours=[cnt]).getCenter()[0]
    return cnt, area, center

# ---------- Основной код ----------
try:
    robot = AquaRobot()
    robot.start(cv2=cv2)

    # Цвета портов (очередь выполнения)
    queue = [colors.Yellow, colors.Green, colors.Red]
    queue_names = ['Yellow', 'Green', 'Red']
    Orange = colors.Orange

    PID_yaw_port = PID_regulator(-0.04, 0, 0, 0)
    PID_speed_port = PID_regulator(0.0004, 0, 0, 90000)

    PID_yaw_gate = PID_regulator(-0.08, 0, 0, 0)
    PID_speed_gate = PID_regulator(0.0004, 0, 0, 90000)

    # Первый проезд через ворота
    turn_to_gate(robot, Orange)
    robot.sleepV(2)
    pass_through_gate(robot, Orange, PID_yaw_gate, PID_speed_gate)
    robot.sleepV(2)

    # Обработка портов
    completed_indices = set()
    while len(completed_indices) < len(queue):
        img = robot.camera.get_frame()

        best_color_idx = -1
        best_area = 0
        best_contour = None
        best_center = None

        # Поиск самого большого контура среди оставшихся цветов
        for idx, color in enumerate(queue):
            if idx in completed_indices:
                continue
            cnt, area, center = get_largest_contour_for_color(img, color)
            if cnt is not None and area > best_area:
                best_area = area
                best_color_idx = idx
                best_contour = cnt
                best_center = center

        if best_color_idx != -1 and best_area > 500:
            color = queue[best_color_idx]
            print(queue_names[best_color_idx])

            # Находим нижнюю точку контура
            min_y = max(best_contour, key=lambda p: p[0][1])[0][1]
            mask = FindMask(img)  # для получения cntr_frame
            if min_y <= mask.cntr_frame.y * 2 - 5:
                delta_x = (mask.cntr_frame - best_center).x
                u_yaw = PID_yaw_port(delta_x)
                u_speed = PID_speed_port(best_area)

                robot.motor_left.set_motor(u_speed - u_yaw)
                robot.motor_right.set_motor(u_speed + u_yaw)

                # Визуализация
                display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                display = cv2.drawContours(display, [best_contour], -1, (255, 0, 0), -1)
                cv2.line(display, (0, min_y), (820, min_y), (0, 0, 0), 3)
                robot.video.write(display)
            else:
                # Достигли порта – отъезжаем назад и удаляем цвет из очереди
                robot.sleepV(2)
                completed_indices.add(best_color_idx)
                robot.motor_left.set_motor(-35 + u_yaw * 3)
                robot.motor_right.set_motor(-35 - u_yaw * 3)
                robot.sleepV(3 if len(completed_indices) < 3 else 1)
                robot.motor_left.set_motor(0)
                robot.motor_right.set_motor(0)
        else:
            # Если ничего не видим – вращаемся
            speed = 25
            reverse = -1 if PID_yaw_port.integral_err <= 0 else 1
            robot.motor_left.set_motor(speed * reverse)
            robot.motor_right.set_motor(-speed * reverse)

    # Финальный проезд через ворота
    turn_to_gate(robot, Orange)
    robot.sleepV(2)
    pass_through_gate(robot, Orange, PID_yaw_gate, PID_speed_gate)

except Exception:
    print("Произошла ошибка:")
    traceback.print_exc()