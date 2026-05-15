import math
import cv2
import numpy as np
import occupancy_grid


RADIUS = 30


def cm_to_pixels(x, config):
    pixels_per_meter = occupancy_grid.WIDTH / config['scale']
    return pixels_per_meter * x / 100


def m_to_pixels(x, config):
    pixels_per_meter = occupancy_grid.WIDTH / config['scale']
    return pixels_per_meter * x


def pixels_to_cm(x, config):
    pixels_per_meter = occupancy_grid.WIDTH / config['scale']
    return x / pixels_per_meter * 100


def pixels_to_m(x, config):
    pixels_per_meter = occupancy_grid.WIDTH / config['scale']
    return x / pixels_per_meter


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
    return mean[0], tangent


def get_angle(tangent):
    x, y = tangent
    sin, cos = -x, -y
    return math.degrees(math.atan2(sin, cos))


def fix_tangent(tangent, mean, mask, right_hand=True):
    tx, ty = tangent
    normal_cw = np.array([-ty, tx])
    normal_ccw = np.array([ty, -tx])
    sample_dist = 5
    p_cw = mean + normal_cw * sample_dist
    p_ccw = mean + normal_ccw * sample_dist
    if mask[int(p_cw[1]), int(p_cw[0])] > 0:
        wall_normal = normal_cw
    else:
        wall_normal = normal_ccw
    if right_hand:
        return np.array([wall_normal[1], -wall_normal[0]])
    else:
        return np.array([-wall_normal[1], wall_normal[0]])


def find_wall(grid, config, right_hand = True):
    mask = (grid[..., 0] == occupancy_grid.OCCUPIED[0]).astype(np.uint8)
    edges = get_edges(mask)
    nearest_x, nearest_y = closest_edge(edges)
    xs, ys = get_local_points(edges, nearest_x, nearest_y, cm_to_pixels(RADIUS, config))
    mean, tangent = get_tangent(xs, ys)
    tangent = fix_tangent(tangent, mean, mask, right_hand)
    return mean, tangent


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


def clamp_drone_speed(vx, vy, max_spd = 0.25):
    hypot = math.hypot(vx, vy)
    if hypot > max_spd:
        vx *= max_spd / hypot
        vy *= max_spd / hypot
    return vx, vy


def get_wall_distance(mean, tangent, right_hand = True):
    wx, wy = mean
    tx, ty = tangent
    drone_x, drone_y = occupancy_grid.X_CENTER - wx, occupancy_grid.Y_CENTER - wy  # relative to wall
    if right_hand:
        normal = np.array([ty, -tx])
    else:
        normal = np.array([-ty, tx])
    return drone_x * normal[0] + drone_y * normal[1]


def draw_drone_speed(grid, vx, vy):
    color_over = (0, 0, 255)
    color_ok = (255, 0, 255)

    max_len = 20
    max_spd = 0.25

    hypot = math.hypot(vx, vy)
    if hypot > max_spd:
        vx *= max_spd / hypot
        vy *= max_spd / hypot
        color = color_over
    else:
        color = color_ok

    scale = max_len / max_spd
    cx, cy = occupancy_grid.X_CENTER, occupancy_grid.Y_CENTER
    dx = vy * scale
    dy = -vx * scale
    x2 = int(cx + dx)
    y2 = int(cy + dy)
    cv2.line(grid, (cx, cy), (x2, y2), color, 2)


def get_speed(tangent, distance_to_wall, right_hand = True):
    tangent_x, tangent_y = -tangent[1], tangent[0]
    v = 0.15
    vx, vy = v*tangent_x, v*tangent_y

    if right_hand:
        normal_x, normal_y = tangent_y, -tangent_x
    else:
        normal_x, normal_y = -tangent_y, tangent_x
    target_wall_distance = 20
    error = target_wall_distance - distance_to_wall
    print(distance_to_wall)
    kP = 0.007
    vx, vy = error * kP * normal_x + vx, error * kP * normal_y + vy
    return vx, vy


def main():
    data = occupancy_grid.read_log('lidar_log_wall.txt')[184:839]
    config = occupancy_grid.read_config()
    angle = 270
    edges = np.zeros((occupancy_grid.HEIGHT, occupancy_grid.WIDTH), np.uint8)
    for line in data:
        grid = occupancy_grid.make_grid(line, config)

        mean, tangent = find_wall(grid, config, True)
        dist = pixels_to_cm(get_wall_distance(mean, tangent, True), config)
        vx, vy = get_speed(tangent, dist)

        draw_drone_speed(grid, vx, vy)
        cv2.circle(grid, (round(occupancy_grid.X_CENTER), round(occupancy_grid.Y_CENTER)), 2, (255, 255, 255),
                   thickness=-1)

        cv2.imshow('Grid', cv2.resize(grid, None, None, 3, 3, interpolation=cv2.INTER_NEAREST))
        key = cv2.waitKey(30)
        if key == ord('q') or key == 27:
            break


if __name__ == '__main__':
    main()
