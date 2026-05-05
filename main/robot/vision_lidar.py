import time, cv2
import traceback
import numpy as np
from robot.control.lidar.ms200k.oradar_lidar import LidarReader
from control.web.webGUI import WebGUI

WINDOW_NAME = "Lidar Scan"
IMAGE_SIZE = 800
MAP_SIZE_M = 20.0
PIXELS_PER_METER = IMAGE_SIZE / MAP_SIZE_M
CENTER = (IMAGE_SIZE // 2, IMAGE_SIZE // 2)
BACKGROUND_COLOR = (0, 0, 0)   # чёрный (позже в HSV)

def get_contour_points(scan):
    """
    Возвращает массив (N,2) пиксельных координат, упорядоченных по углу,
    для построения замкнутого контура. Точки с dist < 0.1 отбрасываются.
    """
    if len(scan) == 0:
        return np.empty((0, 2), dtype=np.int32)

    angles = scan['angle'] % 360.0
    dists = scan['distance']
    valid = dists >= 0.1
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.int32)

    angles = angles[valid]
    dists = dists[valid]

    # Сортировка по углу (как при сканировании)
    sort_idx = np.argsort(angles)
    angles = angles[sort_idx]
    dists = dists[sort_idx]

    # Преобразование в декартовы координаты (с зеркалированием, как в основном коде)
    theta = np.deg2rad(angles)
    x_m = dists * np.cos(theta)
    y_m = dists * np.sin(theta)

    x_px = CENTER[0] + (x_m * PIXELS_PER_METER)
    y_px = CENTER[1] + (y_m * PIXELS_PER_METER)
    x_idx = np.round(x_px).astype(int)
    y_idx = np.round(y_px).astype(int)

    # Оставляем только точки в пределах изображения
    in_bounds = (x_idx >= 0) & (x_idx < IMAGE_SIZE) & (y_idx >= 0) & (y_idx < IMAGE_SIZE)
    x_idx = x_idx[in_bounds]
    y_idx = y_idx[in_bounds]

    return np.column_stack([x_idx, y_idx]).astype(np.int32)

def draw_scan_hsv(img, scan):
    """
    Векторизованное рисование скана в HSV (цветной круг).
    img – пустое HSV-изображение (np.uint8).
    """
    if len(scan) == 0:
        return

    # --- фильтрация и преобразование координат ---
    angles = (scan['angle'] + 0) % 360.0          # поворот на 0° (оставили как было)
    dists = scan['distance']
    intensities = scan['intensity']
    valid = dists >= 0.1
    angles = angles[valid]
    dists = dists[valid]
    intensities = intensities[valid]

    # полярные -> декартовы (с уже имеющимся у вас зеркалированием)
    theta = np.deg2rad(angles)
    x_m = dists * np.cos(theta)
    y_m = dists * np.sin(theta)

    x_px = CENTER[0] + (x_m * PIXELS_PER_METER)
    y_px = CENTER[1] + (y_m * PIXELS_PER_METER)
    x_idx = np.round(x_px).astype(int)
    y_idx = np.round(y_px).astype(int)

    # точки в пределах изображения
    in_bounds = (x_idx >= 0) & (x_idx < IMAGE_SIZE) & (y_idx >= 0) & (y_idx < IMAGE_SIZE)
    x_idx = x_idx[in_bounds]
    y_idx = y_idx[in_bounds]
    angles_f = angles[in_bounds]
    intensities_f = intensities[in_bounds]

    # --- формирование цветов HSV ---
    # Hue: 0..179
    h = (angles_f / 360.0 * 179).astype(np.uint8)

    # Нормализация насыщенности по максимальной интенсивности текущего скана
    max_int = np.max(intensities_f) if len(intensities_f) > 0 else 1
    s = np.clip((intensities_f / max_int) * 255, 0, 255).astype(np.uint8)

    v = np.full_like(h, 255, dtype=np.uint8)   # постоянная яркость

    # назначаем пиксели в HSV-изображении
    img[y_idx, x_idx] = np.stack([h, s, v], axis=1)

def main():
    PORT = '/dev/ttyUSB0'
    BAUDRATE = 230400

    lidar = LidarReader(port=PORT, baudrate=BAUDRATE, timeout_ms=500)
    try:
        lidar.start()
        print("Лидар запущен. Для выхода нажмите 'q' или ESC.")

        while True:
            # Создаём чёрный фон в HSV (H=0, S=0, V=0)
            frame_hsv = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

            scan = lidar.get_scan()
            if scan is not None and len(scan) > 0:
                # Рисуем цветные точки
                draw_scan_hsv(frame_hsv, scan)

                # Получаем контур (упорядоченные пиксельные координаты)
                contour = get_contour_points(scan)

            # Конвертируем HSV -> BGR для отображения в OpenCV/JPEG
            frame_bgr = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)

            # Рисуем контур, если он есть
            if scan is not None and len(contour) >= 3:
                frame_bgr_cnt = cv2.polylines(frame_bgr.copy(), [contour], isClosed=True,
                              color=(255, 255, 255), thickness=2)

            gui.imshow(WINDOW_NAME, frame_bgr)
            gui.imshow(WINDOW_NAME + ' test', frame_bgr_cnt)
            key = gui.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        print("\nОстановка по Ctrl+C.")
    except Exception as e:
        traceback.print_exc()
    finally:
        lidar.stop()
        gui.destroyAllWindows()

if __name__ == "__main__":
    gui = WebGUI(host='0.0.0.0', port=5000)
    gui.start()
    main()