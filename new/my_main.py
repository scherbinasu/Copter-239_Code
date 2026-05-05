from robot.robots import *
drone = Drone()
drone.start()
async def main():
    await drone.arm()
    await drone.takeoff(1)
    await drone.set_velocity(vx=1)
    await drone.sleep(2)
    await drone.set_velocity()
    await drone.land()
    await drone.disarm()


if __name__ == '__main__':
    asyncio.run(main())