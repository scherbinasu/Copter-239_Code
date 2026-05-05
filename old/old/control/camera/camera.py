import cv2
from picamera2 import Picamera2

class HardCamera:
    def __init__(self, size=(820, 616)):
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"format": 'XRGB8888', "size": size},
            raw={"size": (3280, 2464)},
            buffer_count=2
        )
        try:
            self.picam2.configure(config, allocator="malloc")
        except Exception as e:
            print(f"Ошибка malloc, пробуем DMA: {e}")
            self.picam2.configure(config)  # по умолчанию DMA
    def start(self):
        return self.picam2.start()

    def get_frame(self):
        # Захват кадра
        frame = self.picam2.capture_array()
        frame = cv2.flip(cv2.flip(frame, 1), 0)
        # Конвертация из RGBA в BGR для OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        return frame_bgr

    def release(self):
        """Явное освобождение ресурсов камеры."""
        if hasattr(self, 'picam2'):
            self.picam2.stop()
            self.picam2.close()

    # Опционально: добавим деструктор для автоматического вызова release,
    # но лучше всегда вызывать release() вручную.
    def __del__(self):
        self.release()

if __name__ == '__main__':
    cap = HardCamera()
    frame = cap.get_frame()
    cv2.imwrite('img_output_cam.jpg', frame)
    
    
