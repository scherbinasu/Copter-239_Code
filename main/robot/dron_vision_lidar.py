import traceback

from robot.robots import *
import cv2

async def main():
    async with Drone() as drone:
        try:
            while True:
                scan = drone.get_scan()
                img = cv2.polylines(np.zeros((drone.SCAN_IMAGE_SIZE, drone.SCAN_IMAGE_SIZE, 3), dtype=np.uint8), [drone.get_contour_points(scan)], isClosed=True,
                              color=(255, 255, 255), thickness=2)
                drone.web.imshow('cnt scan', img)
                drone.web.waitKey(10)
        except:
            traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())
