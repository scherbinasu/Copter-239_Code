import cv2
from picamera2 import Picamera2

class HardCamera:
    def __init__(self, size=(820, 616)):
        self.picam2 = Picamera2()
        # Настройка конфигурации: основной поток (то, что вы получите) и raw (полный сенсор)
        config = self.picam2.create_preview_configuration(
            main={"format": 'XRGB8888', "size": size},
            raw={"size": (3280, 2464)},   # для IMX219 (камера v2)
            buffer_count=2
        )
        # Применяем конфигурацию (без allocator – он не нужен в новых версиях)
        self.picam2.configure(config)
        self._started = False   # флаг, запущена ли камера

    def start(self):
        if not self._started:
            self.picam2.start()
            self._started = True
        return self

    def get_frame(self):
        """Возвращает кадр в формате BGR (для OpenCV)."""
        if not self._started:
            self.start()   # автоматически запускаем, если забыли
        # Захват кадра (блокируется до получения кадра)
        frame = self.picam2.capture_array()
        # Отражение по горизонтали и вертикали (если нужно – уберите или измените)
        frame = cv2.flip(cv2.flip(frame, 1), 0)
        # Конвертация из RGBA в BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        return frame_bgr

    def release(self):
        """Освобождение ресурсов."""
        if hasattr(self, 'picam2') and self._started:
            self.picam2.stop()
            self.picam2.close()
            self._started = False

    def __del__(self):
        self.release()


if __name__ == '__main__':
    # Пример использования
    cap = HardCamera(size=(820, 616))
    cap.start()                     # обязательно запустить!
    frame = cap.get_frame()
    cv2.imwrite('img_output_cam.jpg', frame)
    cap.release()
    print("Снимок сохранён как img_output_cam.jpg")