
import os
import socket
import subprocess
import time
import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed, PositionNedYaw
from mavsdk.telemetry import PositionVelocityNed, EulerAngle

def is_port_open(host: str = "localhost", port: int = 50051) -> bool:
    """Проверяет, отвечает ли указанный порт (может использоваться сервером)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def ensure_server_running(server_path: str,
                          connection_url: str = None,
                          port: int = 50051,
                          timeout: float = 10.0) -> subprocess.Popen:
    """
    Проверяет, запущен ли mavsdk_server на указанном порту.
    Если нет – запускает его как подпроцесс и ждёт, пока порт откроется.

    :param server_path: путь к исполняемому файлу mavsdk_server
    :param connection_url: (опционально) строка подключения, например "serial:///dev/ttyACM0:57600"
    :param port: порт gRPC сервера (по умолчанию 50051)
    :param timeout: максимальное время ожидания запуска (сек)
    :return: объект процесса (Popen) или None, если сервер уже работал
    """
    if is_port_open(port=port):
        print(f"✅ Сервер уже запущен на порту {port}")
        return None

    if not os.path.isfile(server_path):
        raise FileNotFoundError(f"Исполняемый файл не найден: {server_path}")

    print(f"🚀 Запускаем mavsdk_server из {server_path}...")
    cmd = [server_path] + ['-p', str(port), '--verbose']
    if connection_url:
        cmd.append(connection_url)

    proc = subprocess.Popen(cmd)

    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_open(port=port):
            print(f"✅ Сервер успешно запущен и слушает порт {port}")
            return proc
        time.sleep(0.2)

    proc.kill()
    raise RuntimeError(f"Не удалось запустить mavsdk_server (порт {port} не открылся за {timeout} сек)")


async def takeoff_n_meters(drone: System, n: float):
    """Взлетает на высоту n метров и ждёт, пока высота не будет достигнута."""
    # Убедимся, что дрон вооружён
    async for armed in drone.telemetry.armed():
        if not armed:
            raise RuntimeError("Дрон не вооружён, взлёт невозможен")
        break
    # Небольшая пауза для стабилизации
    await asyncio.sleep(1)

    print(f"Взлетаем на {n} метров...")
    await drone.action.set_takeoff_altitude(n)
    await drone.action.takeoff()

    async for position in drone.telemetry.position():
        current_alt = position.relative_altitude_m
        print(f"  Текущая высота: {current_alt:.2f} м")
        if current_alt >= n - 0.1:
            print("✅ Заданная высота достигнута")
            break


async def land(drone: System):
    """
    Выполняет посадку и ждёт, пока дрон не окажется на земле.
    """
    print("Выполняем посадку...")
    await drone.action.land()

    async for position in drone.telemetry.position():
        current_alt = position.relative_altitude_m
        print(f"  Высота при посадке: {current_alt:.2f} м")
        if current_alt <= 0.1:
            print("✅ Посадка завершена")
            break


async def set_velocity_body(drone: System,
                            vx: float, vy: float, vz: float,
                            yaw_rate: float):
    """
    Переключает дрон в режим OFFBOARD и задаёт желаемые скорости в теле дрона.
    vx : вперёд (+) / назад (–)   [м/с]
    vy : вправо (+) / влево (–)    [м/с]
    vz : вниз (+) / вверх (–)      [м/с]   (в NED вниз – положительно)
    yaw_rate : скорость вращения вокруг вертикальной оси [рад/с]
    """
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"Не удалось запустить offboard: {e}")
        return

    vel_msg = VelocityBodyYawspeed(vx, vy, vz, yaw_rate)
    await drone.offboard.set_velocity_body(vel_msg)

    print(f"🚀 Установлены скорости: vx={vx}, vy={vy}, vz={vz}, yaw_rate={yaw_rate}")


class System(System):
    pass