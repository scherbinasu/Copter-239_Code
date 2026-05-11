import asyncio
import sys
import traceback
from pathlib import Path
import os
import selectors
os.environ['GRPC_POLL_STRATEGY'] = 'poll'
import numpy as np

# Добавляем текущую директорию в sys.path
sys.path.append(str(Path(__file__).parent))  # Теперь указываем только родительскую директорию
import control.mavsdk.mavsdk as mavsdk
import control.camera.camera as camera
import control.lidar.ms200k.oradar_lidar as lidar
import control.web.webGUI as webGUI


class Drone:
    SCAN_WINDOW_NAME = "Lidar Scan"
    SCAN_IMAGE_SIZE = 800
    SCAN_MAP_SIZE_M = 20.0
    SCAN_PIXELS_PER_METER = SCAN_IMAGE_SIZE / SCAN_MAP_SIZE_M
    SCAN_CENTER = (SCAN_IMAGE_SIZE // 2, SCAN_IMAGE_SIZE // 2)

    def __init__(self):

        self.lidar = lidar.LidarReader()
        self.camera = camera.HardCamera()
        self.web = webGUI.WebGUI()

    async def start(self):
        print("Starting Drone")
        try:
            mavsdk.ensure_server_running('/home/ubuntu/main/robot/control/mavsdk/mavsdk_server',
                                         "serial:///dev/ttyACM0:115200")
            self.drone = mavsdk.System(mavsdk_server_address="localhost", port=50051)
            await self.drone.connect(system_address="serial:///dev/ttyACM0:115200")
            self.lidar.start()
            self.web.start()
            self.camera.start()
            self.telemetry = self.drone.telemetry
            self.offboard = self.drone.offboard
        except Exception as e:
            traceback.print_exc()

    def get_contour_points(self, scan):
        """
        Возвращает массив (N,2) пиксельных координат, упорядоченных по углу,
        для построения замкнутого контура. Точки с dist < 0.1 отбрасываются.
        """
        if len(scan) == 0:
            return np.empty((0, 2), dtype=np.int32)

        angles = scan['angle']
        dists = scan['distance']
        valid = dists >= 0.1
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.int32)

        angles = angles[valid]
        dists = dists[valid]

        # Сортировка по углу (как при сканировании)
        sort_idx = np.argsort(angles)
        angles = angles[sort_idx]
        dists = dists[sort_idx]

        # Преобразование в декартовы координаты (с зеркалированием, как в основном коде)
        theta = np.deg2rad(angles)
        x_m = dists * -np.sin(theta)
        y_m = dists * -np.cos(theta)

        x_px = self.SCAN_CENTER[0] + (x_m * self.SCAN_PIXELS_PER_METER)
        y_px = self.SCAN_CENTER[1] + (y_m * self.SCAN_PIXELS_PER_METER)
        x_idx = np.round(x_px).astype(int)
        y_idx = np.round(y_px).astype(int)

        # Оставляем только точки в пределах изображения
        in_bounds = (x_idx >= 0) & (x_idx < self.SCAN_IMAGE_SIZE) & (y_idx >= 0) & (y_idx < self.SCAN_IMAGE_SIZE)
        x_idx = x_idx[in_bounds]
        y_idx = y_idx[in_bounds]

        return np.column_stack([x_idx, y_idx]).astype(np.int32)

    def get_frame(self):
        img = self.camera.get_frame()
        self.web.imshow('raw', img)
        return img

    def draw_scan_hsv(self, img, scan):
        """
        Векторизованное рисование скана в HSV (цветной круг).
        img – пустое HSV-изображение (np.uint8).
        """
        if len(scan) == 0:
            return

        # --- фильтрация и преобразование координат ---
        angles = scan['angle']  # поворот на 0° (оставили как было)
        dists = scan['distance']
        intensities = scan['intensity']
        valid = dists >= 0.1
        angles = angles[valid]
        dists = dists[valid]
        intensities = intensities[valid]

        # полярные -> декартовы
        theta = np.deg2rad(angles)
        x_m = dists * -np.sin(theta)
        y_m = dists * np.cos(theta)

        x_px = self.SCAN_CENTER[0] + (x_m * self.SCAN_PIXELS_PER_METER)
        y_px = self.SCAN_CENTER[1] - (y_m * self.SCAN_PIXELS_PER_METER)
        x_idx = np.round(x_px).astype(int)
        y_idx = np.round(y_px).astype(int)

        # точки в пределах изображения
        in_bounds = (x_idx >= 0) & (x_idx < self.SCAN_IMAGE_SIZE) & (y_idx >= 0) & (y_idx < self.SCAN_IMAGE_SIZE)
        x_idx = x_idx[in_bounds]
        y_idx = y_idx[in_bounds]
        angles_f = angles[in_bounds]
        intensities_f = intensities[in_bounds]

        # --- формирование цветов HSV ---
        # Hue: 0..179
        h = (angles_f / 360.0 * 179).astype(np.uint8)

        # Нормализация насыщенности по максимальной интенсивности текущего скана
        max_int = np.max(intensities_f) if len(intensities_f) > 0 else 1
        s = np.clip((intensities_f / max_int) * 255, 0, 255).astype(np.uint8)

        v = np.full_like(h, 255, dtype=np.uint8)  # постоянная яркость

        # назначаем пиксели в HSV-изображении
        img[y_idx, x_idx] = np.stack([h, s, v], axis=1)
        x_center, y_center = self.SCAN_CENTER
        half = int(self.SCAN_PIXELS_PER_METER * 0.2)
        img[y_center - half: y_center + half, x_center - half:x_center + half] = (255, 255, 255)

    def get_scan(self):
        scan = self.lidar.get_scan()
        img = np.zeros((self.SCAN_IMAGE_SIZE, self.SCAN_IMAGE_SIZE, 3), dtype=np.uint8)
        self.draw_scan_hsv(img, scan)
        if not img is None:
            self.web.imshow('scan', img)
        return scan

    async def land(self):
        return await mavsdk.land(self.drone)

    async def takeoff(self, n: float = 1.0):
        return await mavsdk.takeoff_n_meters(self.drone, n)

    async def ascent(self, n: float = 1.0):
        # current_position_ned = mavsdk.PositionVelocityNed(None, None)  # north, east, down
        # current_attitude_euler = mavsdk.EulerAngle(0.0, 0.0, 0.0, timestamp_us=5.0)
        async for pos_vel in self.telemetry.position_velocity_ned():
            print(pos_vel)
            current_position_ned = pos_vel.position
            start_north = current_position_ned.north_m
            start_east = current_position_ned.east_m
            target_down = -n
            await self.offboard.set_position_ned(
                mavsdk.PositionNedYaw(start_north, start_east, target_down, 0)
            )

            await self.offboard.start()
            print("Offboard запущен")

            # 3. Армим двигатели
            print("-- Арминг")
            await self.drone.action.arm()
            break

        # 4. Ждём, пока дрон наберёт высоту (разница down меньше 0.1 м)
        print(f"-- Взлёт до {n} метров")
        async for pos_vel in self.telemetry.position_velocity_ned():
            current_position_ned = pos_vel.position
            await self.offboard.set_position_ned(
                mavsdk.PositionNedYaw(start_north, start_east, target_down, 0)
            )
            altitude_error = abs(current_position_ned.down_m - target_down)
            if altitude_error < 0.1:
                print("Высота достигнута")
                break
            await asyncio.sleep(0.1)

    async def arm(self):
        return await self.drone.action.arm()

    async def disarm(self):
        return await self.drone.action.disarm()

    async def sleep(self, delay):
        await asyncio.sleep(delay)

    async def set_velocity(self,
                           vx: float = 0, vy: float = 0, vz: float = 0,
                           yaw_rate: float = 0):
        """
           Переключает дрон в режим OFFBOARD и задаёт желаемые скорости в теле дрона.
           vx : вперёд (+) / назад (–)   [м/с]
           vy : вправо (+) / влево (–)    [м/с]
           vz : вниз (+) / вверх (–)      [м/с]   (в NED вниз – положительно)
           yaw_rate : скорость вращения вокруг вертикальной оси [рад/с]
        """
        return await mavsdk.set_velocity_body(self.drone, vx, vy, vz, yaw_rate)

    async def release(self):
        # await self.drone.close()
        self.lidar.stop()
        self.camera.release()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()

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
