from robot.robots import *
drone = Drone()
async def main():
    await drone.start()
    while True:
        scan = await drone.get_scan()
        drone.web.imshow(drone.get_contour_points(scan))
        drone.web.waitKey(10)


if __name__ == '__main__':
    asyncio.run(main())
