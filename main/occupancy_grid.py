import pickle
import base64
import json
import cv2
import numpy as np

# IMAGE
WIDTH =  800
HEIGHT = 800
X_CENTER = WIDTH / 2
Y_CENTER = HEIGHT / 2

# COLORS
OCCUPIED = (17, 17, 17)
UNKNOWN = (109, 110, 94)
FREE = (193, 193, 193)


# -------- DATA --------
def parse_line(line):
    line = line.strip()
    data, t = line.split(b' ')
    t = float(t.decode('utf-8'))
    data = pickle.loads(base64.b64decode(data))
    return data, t


def read_log(filename='lidar_log.txt'):
    with open(filename, 'rb') as h:
        return [parse_line(i)[0] for i in h.readlines()]


# -------- TRACKBARS --------
def create_trackbars(frames_amount):
    config = {}
    try:
        with open('config.json', 'r') as h:
            config = json.load(h)
    except FileNotFoundError:
        pass
    config['frame'] = min(config['frame'], frames_amount-1)
    cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Trackbars', 640, 480)
    cv2.createTrackbar('Frame', 'Trackbars', config.get('frame', 0), frames_amount-1, lambda x: None)
    cv2.createTrackbar('Scale', 'Trackbars', config.get('scale', 20), 40, lambda x: None)
    cv2.createTrackbar('Tolerance', 'Trackbars', config.get('tolerance', 30), 60, lambda x: None)
    cv2.createTrackbar('Radius', 'Trackbars', config.get('radius', 25), 50, lambda x: None)
    cv2.createTrackbar('Median window', 'Trackbars', config.get('window', 1), 10, lambda x: None)


def update_config():
    config = {'frame': cv2.getTrackbarPos('Frame', 'Trackbars'),
              'scale': cv2.getTrackbarPos('Scale', 'Trackbars'),
              'tolerance': cv2.getTrackbarPos('Tolerance', 'Trackbars'),
              'radius': cv2.getTrackbarPos('Radius', 'Trackbars'),
              'window': cv2.getTrackbarPos('Median window', 'Trackbars')}
    with open('config.json', 'w') as h:
        json.dump(config, h)
    config['scale'] = max(config['scale'], 1)
    return config


# -------- MATH --------
def lidar_to_pixel(lidar_frame, config):
    pixels_per_meter = WIDTH / config['scale']
    # print('Resolution:', 100/pixels_per_meter, 'cm')
    angles = np.deg2rad(lidar_frame['angle'])
    dists_scaled = lidar_frame['distance'] * pixels_per_meter
    col = X_CENTER - np.sin(angles) * dists_scaled
    row = Y_CENTER - np.cos(angles) * dists_scaled
    points = np.column_stack((col, row))
    points = np.rint(points).astype('int')
    return points


def filter_pixels(pixels):
    mask = (pixels[:, 0] >= 0) & (pixels[:, 0] < HEIGHT) & \
           (pixels[:, 1] >= 0) & (pixels[:, 1] < WIDTH)
    valid = pixels[mask]
    return valid


def remove_close_points(lidar_frame, config):
    tolerance = config['tolerance'] / 100
    mask = (lidar_frame['distance'] >= tolerance)
    return lidar_frame[mask]


def median_filter(lidar_frame, config):
    order = np.argsort(lidar_frame['angle'])
    result = lidar_frame[order]
    distances = result['distance']
    amount = (config['window'] * 2) + 1
    half = amount // 2
    stack = [np.roll(distances, i) for i in range(-half, half + 1)]
    stack = np.vstack(stack)
    filtered = np.median(stack, axis=0)
    result['distance'] = filtered
    return result


def make_grid(lidar_frame, config):
    # Preprocessing
    lidar_frame = median_filter(lidar_frame, config)
    contour = lidar_to_pixel(lidar_frame, config).reshape((-1, 1, 2))

    # Unknown/free
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[:, :] = UNKNOWN
    cv2.drawContours(img, [contour], 0, FREE, thickness=cv2.FILLED)

    # Walls
    lidar_frame = remove_close_points(lidar_frame, config)
    pixels = lidar_to_pixel(lidar_frame, config)
    pixels_valid = filter_pixels(pixels)
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[pixels_valid[:, 1], pixels_valid[:, 0]] = True

    # Dilate
    pixels_per_meter = WIDTH / config['scale']
    radius = config['radius'] / 100 * pixels_per_meter
    diameter = 2 * round(radius - 0.5) + 1  # Целое нечетное число
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    mask = cv2.dilate(mask, kernel)

    img[mask != 0] = OCCUPIED
    return img


# -------- MAIN --------
def show_frame(lidar_frame, config):
    grid = make_grid(lidar_frame, config)
    pixels_per_meter = WIDTH / config['scale']
    radius = round(10 * pixels_per_meter / 100)
    cv2.circle(grid, (round(X_CENTER), round(Y_CENTER)), radius, (255, 255, 255), thickness=-1)
    cv2.imshow('Image', grid)


def main():
    frames = read_log()
    create_trackbars(len(frames))
    while True:
        config = update_config()
        show_frame(frames[config['frame']], config)
        key = cv2.waitKey(10)
        if key == 27 or key == ord('q'):
            break


if __name__ == '__main__':
    main()
