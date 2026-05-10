from mavsdk.offboard import OffboardError

from robot.robots import *
import traceback
async def main():
    async with Drone() as drone:
        try:
            await drone.ascent(1)
            await drone.sleep(2)
            await drone.set_velocity(vy=0.5)
            await drone.sleep(2)
            await drone.set_velocity()
            await drone.sleep(2)
            await drone.land()
        except:
            traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())
