import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from drone import Point, PID_regulator, Direction, Drone


