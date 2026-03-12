import asyncio
from mavsdk import System
from mavsdk.offboard import (OffboardError, VelocityBodyYawspeed)

# -------------------------------------------------------------------
# 1. Взлёт на n метров (ожидание достижения высоты)
# -------------------------------------------------------------------
async def takeoff_n_meters(drone: System, n: float):
    """
    Взлетает на высоту n метров и ждёт, пока высота не будет достигнута.
    """
    print(f"Взлетаем на {n} метров...")
    # Устанавливаем целевую высоту взлёта
    await drone.action.set_takeoff_altitude(n)
    # Запускаем взлёт
    await drone.action.takeoff()

    # Ожидаем достижения нужной высоты (с допуском 0.1 м)
    async for position in drone.telemetry.position():
        current_alt = position.relative_altitude_m
        print(f"  Текущая высота: {current_alt:.2f} м")
        if current_alt >= n - 0.1:
            print("✅ Заданная высота достигнута")
            break

# -------------------------------------------------------------------
# 2. Посадка и ожидание касания земли
# -------------------------------------------------------------------
async def land(drone: System):
    """
    Выполняет посадку и ждёт, пока дрон не окажется на земле.
    """
    print("Выполняем посадку...")
    await drone.action.land()

    # Ждём, пока относительная высота не станет близкой к 0
    async for position in drone.telemetry.position():
        current_alt = position.relative_altitude_m
        print(f"  Высота при посадке: {current_alt:.2f} м")
        if current_alt <= 0.2:          # считаем, что дрон на земле
            print("✅ Посадка завершена")
            break

# -------------------------------------------------------------------
# 3. Управление скоростями в системе координат тела дрона
#    vx : вперёд (+) / назад (–)   [м/с]
#    vy : вправо (+) / влево (–)    [м/с]
#    vz : вниз (+) / вверх (–)      [м/с]
#    yaw_rate : скорость вращения вокруг вертикальной оси [рад/с]
# -------------------------------------------------------------------
async def set_velocity_body(drone: System,
                            vx: float, vy: float, vz: float,
                            yaw_rate: float):
    """
    Управление скоростью в системе координат тела (body).
    vx: вперед (+), назад (-) м/с
    vy: вправо (+), влево (-) м/с
    vz: вниз (+), вверх (-) м/с
    yaw_rate: скорость вращения вокруг вертикальной оси (рад/с) (положительно по часовой)
    """
    try:
        # Пытаемся запустить offboard (если ещё не запущен)
        await drone.offboard.start()
    except OffboardError as e:
        print(f"Не удалось запустить offboard: {e}")
        return

    # Формируем команду скоростей в теле дрона
    vel_msg = VelocityBodyYawspeed(vx, vy, -vz, yaw_rate)
    await drone.offboard.set_velocity_body(vel_msg)

    print(f"🚀 Установлены скорости: vx={vx}, vy={vy}, vz={vz}, yaw_rate={yaw_rate}")

if __name__ == "__main__":
    async def main():
        # Подключаемся к уже запущенному mavsdk_server
        drone = System(mavsdk_server_address="localhost", port=50051)
        await drone.connect(system_address="serial:///dev/ttyACM0:57600")

        # Ждём подключения к дрону
        async for state in drone.core.connection_state():
            if state.is_connected:
                print("✅ Дрон подключён")
                break

        # Если дрон уже вооружён – снимаем с охраны (чтобы начать с чистого листа)
        async for armed in drone.telemetry.armed():
            if armed:
                print("⚠️ Дрон уже вооружён, выполняем disarm...")
                await drone.action.disarm()
                await asyncio.sleep(2)
            break

        # Вооружаем дрон
        print("🔒 Вооружение...")
        await drone.action.arm()
        print("✅ Дрон вооружён")

        # Взлетаем на 2 метра
        await takeoff_n_meters(drone, 2.0)

        # Двигаемся вперёд со скоростью 1 м/с в течение 3 секунд
        await set_velocity_body(drone, 1.0, 0.0, 0.0, 0.0)
        await asyncio.sleep(3)

        # Останавливаемся
        await set_velocity_body(drone, 0.0, 0.0, 0.0, 0.0)
        await asyncio.sleep(1)

        # Садимся
        await land(drone)

        # Отключаем моторы
        await drone.action.disarm()
        print("🔓 Дрон отключён")
    asyncio.run(main())