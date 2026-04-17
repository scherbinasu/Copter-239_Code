
import sys
from pathlib import Path
import asyncio

# Добавляем текущую директорию в sys.path
sys.path.append(str(Path(__file__).parent))  # Теперь указываем только родительскую директорию

from control.abstractions import *
import control.mavsdk.mavsdk as mavsdk
import control.camera.camera as camera
import control.lidar.ms200k.oradar_lidar as lidar
class Drone:
    def __init__(self):
        self.lidar = lidar.LidarReader()
        self.camera = camera.HardCamera()

    async def start(self):
        mavsdk.ensure_server_running('/home/ubuntu/main/mavsdk_server', "serial:///dev/ttyACM0:115200")
        self.drone = mavsdk.System(mavsdk_server_address="localhost", port=50051)
        await self.drone.connect(system_address="serial:///dev/ttyACM0:115200")
        self.lidar.start()
        self.camera.start()
    def get_frame(self):
        return self.camera.get_frame()
    def get_scan(self):
        return self.lidar.get_scan()
    def land(self):
        return mavsdk.land(self.drone)
    def takeoff(self, n: float = 1.0):
        return mavsdk.takeoff_n_meters(self.drone, n)
    async def arm(self):
        return await self.drone.action.arm()
    async def disarm(self):
        return await self.drone.action.disarm()
    async def sleep(self, delay):
        await asyncio.sleep(delay)
    def set_velocity(self,
                          vx: float = 0, vy: float = 0, vz: float = 0,
                          yaw_rate: float = 0):
        return mavsdk.set_velocity_body(self.drone, vx, vy, -vz, yaw_rate)
    async def release(self):
        await self.drone.close()
        self.lidar.stop()
        self.camera.release()

    async def set_param(self, name: str, value: int | float, retries: int = 2):
        """Устанавливает параметр автопилота с повторными попытками."""
        for attempt in range(retries + 1):
            try:
                if isinstance(value, int):
                    await asyncio.wait_for(
                        self.drone.param.set_param_int(name, value),
                        timeout=5.0
                    )
                else:
                    await asyncio.wait_for(
                        self.drone.param.set_param_float(name, value),
                        timeout=5.0
                    )
                print(f"✅ Параметр {name} установлен в {value}")
                return
            except Exception as e:
                print(f"⚠️ Попытка {attempt + 1}/{retries + 1} не удалась: {e}")
                if attempt < retries:
                    await asyncio.sleep(1)
                else:
                    print(f"❌ Не удалось установить параметр {name} после {retries + 1} попыток")
                    # Не прерываем выполнение, просто логируем

    async def wait_ready(self, timeout: float = 10.0):
        """Ожидает, пока дрон станет готов к арму (is_armable == True)."""
        start = asyncio.get_event_loop().time()
        async for health in self.drone.telemetry.health():
            if health.is_armable:
                print("Дрон готов к включению моторов.")
                return True
            if asyncio.get_event_loop().time() - start > timeout:
                print("Таймаут ожидания готовности дрона.")
                return False
            await asyncio.sleep(0.5)




