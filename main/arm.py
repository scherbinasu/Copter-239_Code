from time import sleep
import drone.control.mavsdk.mavsdk as mavsdk
import asyncio


async def main():
    # Запускаем сервер
    mavsdk.ensure_server_running('./mavsdk_server')

    # Создаем объект дрона
    drone = mavsdk.System(
        mavsdk_server_address="localhost",
        port=50051
    )

    # Подключаемся к дрону
    await drone.connect(system_address="serial:///dev/ttyACM0:57600")

    # Ждем установления соединения
    async for connection_state in drone.core.connection_state():
        if connection_state.is_connected:
            break

    try:
        # Армаж выполняется через объект drone
        print("Попытка заармить дрон...")
        arm_result = await drone.action.arm()

        if arm_result.result == drone.action.Result.SUCCESS:
            print("Дрон успешно заармен!")
        else:
            print(f"Ошибка арминга: {arm_result.result}")

        await asyncio.sleep(2)  # Используйте asyncio.sleep вместо sleep

        # Дизарм
        await drone.action.disarm()
        print("Дрон дизармен")

    except Exception as e:
        print(f"Произошла ошибка: {e}")
    finally:
        # Правильно завершаем соединение
        await drone.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
