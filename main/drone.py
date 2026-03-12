from abstractions.abstractions import *
import control.camera.camera as camera
import control.lidar.ms200k.oradar_lidar as lidar
import control.mavsdk.mavsdk as mavsdk
import asyncio
class Drone:
    def __init__(self):
        self.lidar = lidar.LidarReader()
        self.camera = camera.HardCamera()

    async def start(self):
        mavsdk.ensure_server_running('./mavsdk_server')
        self.drone = mavsdk.System(mavsdk_server_address="localhost", port=50051)
        await self.drone.connect(system_address="serial:///dev/ttyACM0:57600")
        self.lidar.start()
        self.camera.start()
    def get_frame(self):
        return self.camera.get_frame()
    def get_scan(self):
        return self.lidar.get_scan()
    def land(self):
        return mavsdk.land(self.drone)
    def takeoff(self, n:float):
        return mavsdk.takeoff_n_meters(self.drone, n)
    def set_velocity(self,
                          vx: float, vy: float, vz: float,
                          yaw_rate: float):
        return mavsdk.set_velocity_body(self.drone, vx, vy, vz, yaw_rate)
    def release(self):
        self.drone.disconnect()
        self.lidar.stop()
        self.camera.release()

    def __del__(self):
        self.release()




