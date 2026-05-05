import sys, time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from drone import Point, PID_regulator, Direction, Drone
drone = Drone()
drone.start()
drone.takeoff()
time.sleep(5)
scan = drone.get_scan()
index_scan_sorted_distance = sorted(range(len(scan[2:-2])), key=lambda k: sum([scan[k+i].distance for i in [-2, -1, 0, 1, 2]]))





