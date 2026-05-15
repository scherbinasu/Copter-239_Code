import asyncio
import time
import traceback

import cv2
import occupancy_grid_wall as wall


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
    cv2.imshow('Image', cv2.resize(grid, None, None, 4, 4, cv2.INTER_NEAREST))
    return vx, vy


def main():
    delta = -1
    config = wall.occupancy_grid.read_config()
    with open('lidar_log_line.txt', 'rb') as log:
        for i in log.readlines():
            frame, t = wall.occupancy_grid.parse_line(i)
            if delta == -1:
                delta = time.time() - t
            dt = round((delta + t - time.time()) * 1000)
            key = -1
            if dt > 0:
                key = cv2.waitKey(dt)
            show_frame(frame, config)
            if key == 27 or key == ord('q'):
                break


if __name__ == '__main__':
    main()
