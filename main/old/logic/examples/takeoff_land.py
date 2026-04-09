import sys, time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from drone import Point, PID_regulator, Direction, Drone
drone = Drone()
drone.start()
drone.takeoff(1)
time.sleep(1)
drone.land()