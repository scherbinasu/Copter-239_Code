from robot.robots import Drone
import numpy as np
import asyncio

async def main():
    async with Drone() as drone:
        while True:
            scan = drone.get_scan()                        # получение данных лидара
            angles_deg = scan['angle']                     # углы в градусах
            distances = scan['distance']                   # дистанции в метрах

            valid = distances >= 0.1                       # отбрасываем невалидные измерения
            ang = angles_deg[valid]
            dist = distances[valid]

            if len(dist) < 2:
                await asyncio.sleep(0.1)
                continue

            # углы в радианы
            theta = np.deg2rad(ang)

            # индексы всех уникальных пар (верхний треугольник без диагонали)
            i, j = np.triu_indices(len(dist), k=1)

            # разность углов для всех пар
            dtheta = np.abs(theta[i] - theta[j])

            # попарные расстояния по теореме косинусов (векторизованно)
            pairwise_dist = np.sqrt(
                dist[i]**2 + dist[j]**2 -
                2 * dist[i] * dist[j] * np.cos(dtheta)
            )

            # пример вывода: первые 5 пар и соответствующие им расстояния
            for idx in range(min(5, len(pairwise_dist))):
                print(f"Пара {i[idx]:3d}-{j[idx]:3d}: "
                      f"dist={pairwise_dist[idx]:.3f} м, "
                      f"углы {ang[i[idx]]:.1f}° и {ang[j[idx]]:.1f}°")
            print(f"Всего уникальных пар: {len(pairwise_dist)}\n")

            await asyncio.sleep(0.5)   # чтобы не заваливать консоль

if __name__ == '__main__':
    asyncio.run(main())