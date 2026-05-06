#!/usr/bin/env python3
"""
Пример автономного полёта в помещении с H-Flow и MAVSDK (Offboard, PositionNED).
Выполняет взлёт и смещение на 5 метров вперёд (на север).
"""



if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import math
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw
from mavsdk.telemetry import PositionVelocityNed, EulerAngle

# Глобальные переменные для хранения последних значений телеметрии
current_position_ned = PositionVelocityNed(None, None)  # north, east, down
current_attitude_euler = EulerAngle(0.0, 0.0, 0.0, timestamp_us=5.0)            # roll, pitch, yaw

# Флаги получения первых данных
position_received = False
attitude_received = False

async def subscribe_telemetry_attitude(telemetry):
    """Подписываемся на обновления позиции и углов Эйлера."""
    global current_attitude_euler, attitude_received
    async for euler in telemetry.attitude_euler():
        current_attitude_euler = euler
        #print(euler)
        attitude_received = True

async def subscribe_telemetry_position(telemetry):
    """Подписываемся на обновления позиции и углов Эйлера."""
    global current_position_ned, position_received

    async for pos_vel in telemetry.position_velocity_ned():
        current_position_ned = pos_vel.position
        #print(pos_vel)
        position_received = True

async def run():
    drone = System(mavsdk_server_address="localhost", port=50051)
    await drone.connect(system_address="serial:///dev/ttyACM0:115200")

    print("Ожидание подключения дрона...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"Дрон подключён")
            break

    telemetry = drone.telemetry
    offboard = drone.offboard

    # Запускаем задачу постоянного обновления телеметрии
    asyncio.ensure_future(subscribe_telemetry_position(telemetry))
    asyncio.ensure_future(subscribe_telemetry_attitude(telemetry))
    # Ждём получения первых данных (не более 10 секунд)
    for i in range(100):
        if position_received and attitude_received:
            break
        await asyncio.sleep(0.1)
    else:
        print("Не удалось получить телеметрию")
        return

    print("-- Взлёт и переход в Offboard")
    try:
        # 1. Включаем Offboard‑режим (отправляем setpoint с текущими координатами)
        #    Для безопасного перехода необходимо начать отправку setpoint ДО команды arm/takeoff
        #await offboard.start()
        #print("Offboard запущен")

        # 2. Задаём начальную точку висения на месте (высота 2 метра, down = -2.0)
        #    Используем текущие north, east (они будут ~0 на старте) и фиксированную высоту
        start_north = current_position_ned.north_m
        start_east = current_position_ned.east_m
        target_down = -1   # 2 метра над землёй (ось Down направлена вниз)
        start_yaw = current_attitude_euler.yaw_deg
        #exit(0)
        await offboard.set_position_ned(
            PositionNedYaw(start_north, start_east, target_down, start_yaw)
        )

        await offboard.start()
        print("Offboard запущен")

        # 3. Армим двигатели
        print("-- Арминг")
        await drone.action.arm()

        # 4. Ждём, пока дрон наберёт высоту (разница down меньше 0.2 м)
        print("-- Взлёт до 2 метров")
        while True:
            await offboard.set_position_ned(
                PositionNedYaw(start_north, start_east, target_down, start_yaw)
            )
            altitude_error = abs(current_position_ned.down_m - target_down)
            if altitude_error < 0.2:
                print("Высота достигнута")
                break
            await asyncio.sleep(0.1)

        # 5. Выполняем точное смещение
        distance_north = 1.5   # на 5 метров вперёд
        distance_east  = 0.0
        print(f"-- Смещение на {distance_north} м на север, {distance_east} м на восток")
        success = await move_relative(offboard, telemetry,
                                      distance_north, distance_east,
                                      timeout=10.0)
        if success:
            print("-- Достигнута целевая точка")
        else:
            print("-- Ошибка: не удалось достичь цели за отведённое время")
        await asyncio.sleep(2)
        success = await move_relative(offboard, telemetry, -1.5, 0.0, timeout=10.0)
        # 6. Посадка (или возврат в точку взлёта)
        print("-- Посадка")
        await drone.action.land()

    except OffboardError as e:
        print(f"Offboard ошибка: {e}")
    finally:
        await offboard.stop()
        print("Offboard остановлен")


async def move_relative(offboard, telemetry, delta_north_m: float, delta_east_m: float,
                        timeout: float = 10.0) -> bool:
    """
    Перемещение дрона на указанное расстояние относительно текущего положения.
    Высота сохраняется неизменной.
    Возвращает True, если целевая точка достигнута, иначе False.
    """
    # Вычисляем целевую точку
    target_north = current_position_ned.north_m + delta_north_m
    target_east  = current_position_ned.east_m  + delta_east_m
    target_down  = current_position_ned.down_m - 0.1               # высота остаётся прежней
    target_yaw   = current_attitude_euler.yaw_deg             # курс также сохраняем

    start_time = asyncio.get_event_loop().time()

    while True:
        # Отправляем setpoint в целевую точку (непрерывная отправка обязательна)
        await offboard.set_position_ned(
            PositionNedYaw(target_north, target_east, target_down, target_yaw)
        )

        # Оцениваем ошибку по горизонтали
        err_north = target_north - current_position_ned.north_m
        err_east  = target_east  - current_position_ned.east_m
        horizontal_error = math.hypot(err_north, err_east)

        if horizontal_error < 0.1:   # порог прибытия 20 см
            print(f"Цель достигнута с ошибкой {horizontal_error:.3f} м")
            return True

        if (asyncio.get_event_loop().time() - start_time) > timeout:
            print(f"Таймаут при движении, текущая ошибка {horizontal_error:.3f} м")
            return False

        await asyncio.sleep(0.05)  # частота отправки ~20 Гц


if __name__ == "__main__":
    asyncio.run(run())
