import cv2
from control.camera.camera import HardCamera
from control.web.webGUI import WebGUI

def main():
    # 1. Камера
    cap = HardCamera(size=(820, 616))
    cap.start()

    # 2. Веб-интерфейс
    gui = WebGUI(host='0.0.0.0', port=5000)

    # 3. Параметры HSV
    hsv_min = [0, 0, 0]
    hsv_max = [180, 255, 255]
    obrez = 0

    # 4. Колбэки для трекбаров
    def on_h_min(val): hsv_min[0] = val
    def on_h_max(val): hsv_max[0] = val
    def on_s_min(val): hsv_min[1] = val
    def on_s_max(val): hsv_max[1] = val
    def on_v_min(val): hsv_min[2] = val
    def on_v_max(val): hsv_max[2] = val
    def on_obrez(val):
        nonlocal obrez
        obrez = val

    # Создаём трекбары
    gui.createTrackbar("H Min", "control", 0, 180, on_h_min)
    gui.createTrackbar("H Max", "control", 180, 180, on_h_max)
    gui.createTrackbar("S Min", "control", 0, 255, on_s_min)
    gui.createTrackbar("S Max", "control", 255, 255, on_s_max)
    gui.createTrackbar("V Min", "control", 0, 255, on_v_min)
    gui.createTrackbar("V Max", "control", 255, 255, on_v_max)
    gui.createTrackbar("Obrez", "control", 0, 480, on_obrez)

    # 5. Детектор ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    gui.start()
    print("Сервер запущен. Откройте в браузере: http://<IP-адрес>:5000")

    while True:
        frame = cap.get_frame()      # BGR
        if frame is None:
            continue

        # Создаём маску
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = tuple(hsv_min)
        upper = tuple(hsv_max)
        mask = cv2.inRange(hsv, lower, upper)
        mask[:obrez, :] = 0

        # Детекция ArUco на исходном кадре
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Отображаем два окна
        gui.imshow("raw", frame)          # окно с именем "raw"
        gui.imshow("masked", mask)        # окно с именем "masked"

        # Задержка для управления частотой кадров (имитация waitKey)
        if gui.waitKey(30) == ord('q'):   # всегда -1, но оставляем для совместимости
            break

    cap.release()
    gui.destroyAllWindows()

if __name__ == '__main__':
    main()