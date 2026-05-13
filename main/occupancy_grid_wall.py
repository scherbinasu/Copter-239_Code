import math
import cv2
import numpy as np
import occupancy_grid


SECTOR_DEG = 5
MAX_DIST = 200


def build_ray_table(sector_deg):
    table = {}
    for angle_deg in range(0, 360, sector_deg):
        angle = math.radians(angle_deg)
        dx = -math.sin(angle)
        dy = -math.cos(angle)
        coords = np.array([
            (
                int(dx * r),
                int(dy * r)
            )
            for r in range(MAX_DIST)
        ], dtype=np.int16)
        table[angle_deg] = coords
    return table


RAY_TABLE = build_ray_table(SECTOR_DEG)


def sector_distances(mask, center, pixels_per_meter):
    cx, cy = center
    cx, cy = round(cx), round(cy)
    result = {}
    h, w = mask.shape
    for angle_deg, offsets in RAY_TABLE.items():
        xs = cx + offsets[:, 0]
        ys = cy + offsets[:, 1]
        valid = (
            (xs >= 0) &
            (ys >= 0) &
            (xs < w) &
            (ys < h)
        )
        xs = xs[valid]
        ys = ys[valid]
        ray = mask[ys, xs]
        hits = np.flatnonzero(ray)
        if len(hits):
            dist = hits[0]
        else:
            dist = len(ray)
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
    angle = 270
    for line in data:
        grid = occupancy_grid.make_grid(line, config)
        mask = (grid[..., 0] == occupancy_grid.OCCUPIED[0]).astype(np.uint8)
        sectors = sector_distances(mask,
                                   (occupancy_grid.X_CENTER, occupancy_grid.Y_CENTER),
                                   pixels_per_meter)

        init_angle = angle
        while sectors[angle] > threshold:
            angle = (angle - 5) % 360
            if angle == init_angle:
                break
        prev_angle = angle
        while sectors[angle] < threshold:
            angle = (angle + 5) % 360
            if angle == prev_angle:
                angle = init_angle
                break

        draw_angle(grid, angle, (0, 0, 255))
        cv2.circle(grid, (round(occupancy_grid.X_CENTER), round(occupancy_grid.Y_CENTER)), 2, (255, 255, 255), thickness=-1)

        cv2.imshow('Grid', cv2.resize(grid, None, None, 3, 3, interpolation=cv2.INTER_NEAREST))
        key = cv2.waitKey(30)
        if key == ord('q') or key == 27:
            break


if __name__ == '__main__':
    main()
