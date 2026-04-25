import robot.control.motor.motors as motors

import time
try:
    motor_left = motors.HardMotor(pwm_channel=motors.PWM_CHANNEL_2, hz=motors.PWM_FREQ, chip=motors.PWM_CHIP)
    motor_right = motors.HardMotor(pwm_channel=motors.PWM_CHANNEL_1, hz=motors.PWM_FREQ, chip=motors.PWM_CHIP)
    time.sleep(2)
    motor_left.set_motor(30)
    motor_right.set_motor(30)
    print('sleep')
    while 1:time.sleep(3600)
except KeyboardInterrupt:pass
motor_left.set_motor(0)
motor_right.set_motor(0)
motor_left.stop()
motor_right.stop()