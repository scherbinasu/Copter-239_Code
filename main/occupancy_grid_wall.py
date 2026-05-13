import math
import cv2
import numpy as np

import occupancy_grid


def sector_distances(mask, center, pixels_per_meter, sector_deg):
    cx, cy = center
    result = {}
    for angle_deg in range(0, 360, sector_deg):
        angle = math.radians(angle_deg)

        dx = -math.sin(angle)
        dy = -math.cos(angle)

        dist = 200

        for r in range(200):
            x = int(cx + dx * r)
            y = int(cy + dy * r)

            if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]:
                dist = r
                break

            if mask[y, x] > 0:
                dist = r
                break

        result[angle_deg] = dist / pixels_per_meter

    return result


def draw_angle(grid, angle, color):
    length = 10
    cx, cy = occupancy_grid.X_CENTER, occupancy_grid.Y_CENTER
    angle = math.radians(angle)
    dx = -math.sin(angle)
    dy = -math.cos(angle)
    x2 = int(cx + dx * length)
    y2 = int(cy + dy * length)
    cx, cy = int(cx), int(cy)
    cv2.line(grid, (cx, cy), (x2, y2), color, 2)


def main():
    data = occupancy_grid.read_log('lidar_log_wall.txt')[184:839]
    config = occupancy_grid.read_config()
    pixels_per_meter = occupancy_grid.WIDTH / config['scale']
    threshold = 30 / 100
    current_right = 270
    for line in data:
        grid = occupancy_grid.make_grid(line, config)
        mask = np.zeros((occupancy_grid.HEIGHT, occupancy_grid.WIDTH), dtype=np.uint8)
        mask[np.all(grid == occupancy_grid.OCCUPIED, axis=2)] = 255
        sector_deg = 5
        sectors = sector_distances(mask,
                                   (occupancy_grid.X_CENTER, occupancy_grid.Y_CENTER),
                                   pixels_per_meter, sector_deg)

        search_range = 60
        best_angle = None
        best_dist = 999
        for da in range(-search_range, search_range + 1, sector_deg):
            a = (current_right + da) % 360
            d = sectors[a]
            if d < best_dist:
                best_dist = d
                best_angle = a
        current_right = best_angle

        forward = (current_right - 90) % 360

        clearance = 0.1
        ok = True
        search_range = 10
        for da in range(-search_range, search_range + 1, sector_deg):
            a = (forward + da) % 360
            if sectors[a] < clearance:
                ok = False
                break
        if not ok:
            current_right -= 90

        draw_angle(grid, current_right, (0, 255, 0))

        cv2.imshow('Grid', cv2.resize(grid, None, None, 3, 3, interpolation=cv2.INTER_NEAREST))
        key = cv2.waitKey(10)
        if key == ord('q') or key == 27:
            break


if __name__ == '__main__':
    main()
