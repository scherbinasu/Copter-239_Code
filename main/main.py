from drone.drone import *

async def main():
    import drone.drone as drone
    drone = drone.Drone()
    await drone.start()
    await drone.arm()
    await drone.sleep(0.5)
    await drone.takeoff(1.5)
    while True:
        frame = drone.get_frame()
        scan = drone.get_scan()



if __name__ == "__main__":
    asyncio.run(main())

