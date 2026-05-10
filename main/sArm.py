import traceback
from robot.robots import *
drone = Drone()


async def main():
    await drone.start()
    await drone.arm()
    await asyncio.sleep(0.25)
    await drone.disarm()



if __name__ == "__main__":
    asyncio.run(main())
