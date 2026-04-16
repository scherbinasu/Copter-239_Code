from time import sleep

import asyncio


async def main():
    import drone.drone as drone
    drone = drone.Drone()
    await drone.start()
    await drone.arm()
    await asyncio.sleep(1)
    await drone.set_velocity(vz=1)
    await asyncio.sleep(1)
    await drone.set_velocity(vx=2)
    await asyncio.sleep(2)
    await drone.land()



if __name__ == "__main__":
    asyncio.run(main())
