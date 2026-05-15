import asyncio
import time
import traceback

import cv2
import occupancy_grid_wall as wall
from robot import robots


drone = robots.Drone()


def show_frame(lidar_frame, config):
    t = time.time()
    grid = wall.occupancy_grid.make_grid(lidar_frame, config)
    mean, tangent = wall.find_wall(grid, config, True)
    dist = wall.pixels_to_cm(wall.get_wall_distance(mean, tangent, True), config)
    vx, vy = wall.get_speed(tangent, dist)
    dt = 1000 * (time.time() - t)
    wall.draw_drone_speed(grid, vx, vy)
    cv2.circle(grid, (round(wall.occupancy_grid.X_CENTER), round(wall.occupancy_grid.Y_CENTER)), 2, (255, 255, 255),
               thickness=-1)
    cv2.putText(grid, f'{dt:.2f} ms; res: {wall.pixels_to_cm(1, config):.2f} cm', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    drone.web.imshow('Image', grid)
    return vx, vy


async def main():
    async with drone:
        print('c')
        try:
            frames = wall.occupancy_grid.read_log('lidar_log_wall.txt')
            config = wall.occupancy_grid.read_config()
            while True:
                frame = drone.get_scan()
                vx, vy = show_frame(frame, config)
                key = drone.web.waitKey(30)
                if key == 27 or key == ord('q'):
                    break
        except Exception as e:
            print(''.join(traceback.format_exception(e)))
            raise


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        print(traceback.format_exception(e))
