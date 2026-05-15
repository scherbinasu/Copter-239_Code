import json
import time
import traceback

import cv2
import numpy as np
import occupancy_grid
from robot.robots import Drone

print('-a')
from robot import robots
import asyncio


print('a')
drone = robots.Drone()
print('b')


def create_trackbars(frames_amount):
    config = {}
    try:
        with open('config.json', 'r') as h:
            config = json.load(h)
    except FileNotFoundError:
        pass
    config['frame'] = min(config.get('frame', 0), frames_amount-1)
    # drone.web.namedWindow('Trackbars')
    drone.web.createTrackbar('Frame', 'Trackbars', config.get('frame', 0), frames_amount-1, lambda x: None)
    drone.web.createTrackbar('Scale', 'Trackbars', config.get('scale', 20), 40, lambda x: None)
    drone.web.createTrackbar('Tolerance', 'Trackbars', config.get('tolerance', 30), 60, lambda x: None)
    drone.web.createTrackbar('Radius', 'Trackbars', config.get('radius', 25), 50, lambda x: None)
    drone.web.createTrackbar('Median window', 'Trackbars', config.get('window', 1), 10, lambda x: None)


def update_config():
    config = {'frame': drone.web.getTrackbarPos('Frame'),
              'scale': drone.web.getTrackbarPos('Scale'),
              'tolerance': drone.web.getTrackbarPos('Tolerance'),
              'radius': drone.web.getTrackbarPos('Radius'),
              'window': drone.web.getTrackbarPos('Median window')}
    with open('config.json', 'w') as h:
        json.dump(config, h)
    config['scale'] = max(config['scale'], 1)
    return config


def show_frame(lidar_frame, config):
    t = time.time()
    grid = occupancy_grid.make_grid(lidar_frame, config)
    dt = 1000 * (time.time() - t)
    pixels_per_meter = occupancy_grid.WIDTH / config['scale']
    cm_per_pix = 100 / pixels_per_meter
    radius = round(10 * pixels_per_meter / 100)
    cv2.circle(grid, (round(occupancy_grid.X_CENTER), round(occupancy_grid.Y_CENTER)), radius, (255, 255, 255), thickness=-1)
    cv2.putText(grid, f'{dt:.2f} ms; res: {cm_per_pix:.2f} cm', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    drone.web.imshow('Image', grid)


async def main():
    async with drone:
        print('c')
        frames = occupancy_grid.read_log()
        create_trackbars(len(frames))
        while True:
            config = update_config()
            frame = drone.get_scan()
            show_frame(frame, config)
            key = drone.web.waitKey(30)
            if key == 27 or key == ord('q'):
                break


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        print(traceback.format_exception(e))
