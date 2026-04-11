import asyncio
import drone.control.mavsdk.mavsdk as mavsdk

async def run():
    drone = mavsdk.System(mavsdk_server_address="localhost", port=50051)
    await drone.connect(system_address="serial:///dev/ttyACM0:57600")

    print("Ожидание подключения дрона...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Подключен к дрону!")
            break

    # --- Отключаем проверку GPS в ARMING_CHECK ---
    print("-- Настройка параметров для arm без GPS")
    # Для ArduPilot бит GPS = 4. Убираем его из маски (по умолчанию 1 = проверять всё)
    ARMING_CHECK_DEFAULT = 1      # 0b00000001
    GPS_BIT = 4                   # 0b00000100
    new_mask = ARMING_CHECK_DEFAULT ^ GPS_BIT  # 1 ^ 4 = 5

    try:
        await drone.param.set_param_int("ARMING_CHECK", new_mask)
        print(f"-- Параметр ARMING_CHECK установлен в {new_mask}")
        # Небольшая пауза, чтобы параметр точно применился
        await asyncio.sleep(1)
    except Exception as e:
        print(f"Ошибка установки параметра: {e}")
        # Если параметр не меняется, возможно, полётный контроллер не отвечает
        return

    print("-- Выполняется arm")
    await drone.action.arm_force()
    print("-- Дрон заarm-лен")
    await asyncio.sleep(2)
    await drone.action.disarm()
    print("-- Дрон disarm")

if __name__ == "__main__":
    asyncio.run(run())