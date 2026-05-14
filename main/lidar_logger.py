from mavsdk.offboard import OffboardError
from robot.robots import *
import traceback, base64, pickle
import numpy as np
import time


async def main():
    t = 0
    amount = 0
    async with Drone() as drone:
        with open('lidar_log.txt', 'wb') as log:
            while True:
                try:
                    lidar_data: np.ndarray = drone.lidar.get_scan()
                    data_bytes = pickle.dumps(lidar_data)
                    data = base64.b64encode(data_bytes)
                    log.write(data)
                    log.write(b' ')
                    if t == 0:
                        t = time.time()
                    log.write(str(time.time() - t).encode('ascii'))
                    log.write(b'\n')
                    amount = amount + len(data)
                    print('\ramount MB: ' + str(amount/1024/1024), end='\t\t\t')
                except Exception as e:
                    traceback.print_exc()
                    break
        print()


if __name__ == '__main__':
    asyncio.run(main())
