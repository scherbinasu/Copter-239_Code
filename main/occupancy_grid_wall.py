import math
import time

import cv2
import numpy as np
import occupancy_grid


def get_edges(mask, edges=None):
    if edges is None:
        edges = np.zeros((occupancy_grid.HEIGHT, occupancy_grid.WIDTH), np.uint8)

    # t = time.time()
    edges[:] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(edges, contours, -1, 255)
    # print('contours:', (time.time() - t) * 1000)

    # t = time.time()
    # kernel = np.ones((2, 2), np.uint8)
    # edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel, edges)
    # edges *= 255
    # print('morph:', (time.time() - t) * 1000)
    #
    # t = time.time()
    # edges = mask ^ cv2.erode(mask, np.ones((3, 3), np.uint8), edges)
    # edges *= 255
    # print('erode:', (time.time() - t) * 1000)

    return edges


def closest_edge(edges):
    ys, xs = np.where(edges > 0)
    cx = occupancy_grid.X_CENTER
    cy = occupancy_grid.Y_CENTER
    dx = xs - cx
    dy = ys - cy
    dist2 = dx * dx + dy * dy
    i = np.argmin(dist2)
    nearest_x = xs[i]
    nearest_y = ys[i]
    return nearest_x, nearest_y


def get_local_points(edges, x, y, r):
    ys, xs = np.where(edges > 0)
    dx = xs - x
    dy = ys - y
    mask_local = dx * dx + dy * dy < r * r
    local_xs = xs[mask_local]
    local_ys = ys[mask_local]
    return local_xs, local_ys


def get_tangent(xs, ys):
    pts = np.column_stack((xs, ys)).astype(np.float32)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(
        pts,
        mean=None
    )
    tangent = eigenvectors[0]
    return mean, tangent


def get_angle(tangent):
    x, y = tangent
    sin, cos = -x, -y
    return math.degrees(math.atan2(sin, cos))


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


# def draw_tangent()


def main():
    data = occupancy_grid.read_log('lidar_log_wall.txt')[184:839]
    config = occupancy_grid.read_config()
    pixels_per_meter = occupancy_grid.WIDTH / config['scale']
    threshold = 30 / 100
    angle = 270
    edges = np.zeros((occupancy_grid.HEIGHT, occupancy_grid.WIDTH), np.uint8)
    for line in data:
        grid = occupancy_grid.make_grid(line, config)
        mask = (grid[..., 0] == occupancy_grid.OCCUPIED[0]).astype(np.uint8)

        edges = get_edges(mask, edges)
        nearest_x, nearest_y = closest_edge(edges)
        xs, ys = get_local_points(edges, nearest_x, nearest_y, 7)
        mean, tangent = get_tangent(xs, ys)
        angle = get_angle(tangent)

        grid[ys, xs] = (0, 255, 255)
        draw_angle(grid, angle, (0, 0, 255))
        cv2.circle(grid, (round(occupancy_grid.X_CENTER), round(occupancy_grid.Y_CENTER)), 2, (255, 255, 255),
                   thickness=-1)
        # cv2.circle(grid, (nearest_x, nearest_y), 3, (0, 255, 255), -1)

        cv2.imshow('Grid', cv2.resize(grid, None, None, 3, 3, interpolation=cv2.INTER_NEAREST))
        cv2.imshow('Edges', cv2.resize(edges, None, None, 3, 3, interpolation=cv2.INTER_NEAREST))
        key = cv2.waitKey(30)
        if key == ord('q') or key == 27:
            break


if __name__ == '__main__':
    main()
