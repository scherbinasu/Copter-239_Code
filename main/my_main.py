from robot.robots import *
drone = Drone()

async def main():
    await drone.start()
    await drone.arm()
    await drone.wait_ready()
    await drone.takeoff(1)
    await drone.set_velocity(vx=0.5)
    await drone.sleep(2)
    await drone.set_velocity()
    await drone.land()
    await drone.disarm()


if __name__ == '__main__':
    asyncio.run(main())