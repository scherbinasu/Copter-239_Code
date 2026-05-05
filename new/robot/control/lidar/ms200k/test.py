import time
from oradar_lidar import LidarReader
import numpy as np
import cv2
def main():
    reader = LidarReader(port='/dev/ttyUSB0')
    try:
        reader.start()
        i = 0
        while 1:
            i += 1
            scan = reader.get_scan()
            if scan is not None:
                bw = np.zeros(scan.shape, dtype=np.uint8)
                bw = bw[:, :, (scan['angle'], scan['distance'], scan['intensity'])]
                print(f"Скан {i}: {len(scan)} точек, среднее расстояние {scan['distance'].mean():.3f} м")
                print(f'{np.array(scan["distance"].tolist()[:10]+scan["distance"].tolist()[-10:]).mean():.3f}')
            else:
                print("Таймаут")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Прервано")
    except Exception as e:
        print(type(e).__name__, e)
    finally:
        reader.stop()

if __name__ == "__main__":
    main()
