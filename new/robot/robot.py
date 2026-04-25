import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import control.camera.camera as camera
import control.motor.motors as motors
import control.web.webGUI as webGUI
import time


class AquaRobot:
    def __init__(self, sizeFrame=(820, 616)):
        self.video = None
        self.sizeFrame = sizeFrame
        self.camera = camera.HardCamera(sizeFrame)
        self.motor_left = motors.HardMotor(pwm_channel=motors.PWM_CHANNEL_1, hz=motors.PWM_FREQ, chip=motors.PWM_CHIP)
        self.motor_right = motors.HardMotor(pwm_channel=motors.PWM_CHANNEL_2, hz=motors.PWM_FREQ, chip=motors.PWM_CHIP)
        self.gui = webGUI.WebGUI()

    def start(self, cv2=None, FPS=30, OUTPUT_FILE="output.mp4"):
        self.camera.start()
        self.motor_left.start()
        self.motor_right.start()
        self.gui.start()
        if not cv2 is None and self.video is None:
            self.FPS = FPS
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video = cv2.VideoWriter(OUTPUT_FILE, fourcc, self.FPS, self.sizeFrame)
        time.sleep(3)

    def stop(self):
        self.camera.release()
        self.motor_left.stop()
        self.motor_right.stop()
        self.gui.destroyAllWindows()

    def sleepV(self, sleep_time):
        time_start = time.time()
        while time_start + sleep_time > time.time():
            self.video.write(self.camera.get_frame())
            time.sleep(1 / self.FPS)

