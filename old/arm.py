from time import time
start_time = time()
import asyncio


async def main():
    import drone.drone as drone
    drone = drone.Drone()
    await drone.start()
    print(time() - start_time)
    print(drone.get_scan().tolist())
    await drone.arm()
    await asyncio.sleep(0.25)
    await drone.disarm()



if __name__ == "__main__":
    asyncio.run(main())
