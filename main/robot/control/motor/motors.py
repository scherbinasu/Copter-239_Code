from rpi_hardware_pwm import HardwarePWM
import time
import sys

# --- Настройки ---
PWM_CHIP = 0               # для Raspberry Pi 4 и более старых версий
PWM_CHANNEL_1 = 0          # соответствует GPIO 12
PWM_CHANNEL_2 = 1          # соответствует GPIO 13
PWM_FREQ = 50              # частота 50 Гц

# Соответствие duty cycle ширине импульса при 50 Гц (период = 20000 мкс)
DUTY_MIN = 5.0             # 1000 мкс → 5%
DUTY_STOP = 7.5            # 1500 мкс → 7.5%
DUTY_MAX = 10.0            # 2000 мкс → 10%
class HardMotor(HardwarePWM):
    reversed = False
    cor_kf = 1 # коэффициент до множения скорости мотора
    just_kf = 1.35 # пропорция между скорость вперёд и назад (скорость вперёд/скорость назад)
    def __init__(self, *args, **kw):
        self.pwm = HardwarePWM(*args, **kw)
    def start(self):
        self.pwm.start(DUTY_STOP)
    def set_motor(self, power_percent, output=False):
        """
        power_percent: от -100 до 100
        """
        # Ограничение
        power_percent = max(-100, min(100, (power_percent*self.cor_kf)*((2*int(self.reversed))-1)))

        # Преобразование процента в duty cycle
        if power_percent >= 0:
            duty = DUTY_STOP + (power_percent / (100.0*self.just_kf)) * (DUTY_MAX - DUTY_STOP)
        else:
            duty = DUTY_STOP + (power_percent / 100.0) * (DUTY_STOP - DUTY_MIN)

        self.pwm.change_duty_cycle(duty)
        if output:print(f"Канал {self.pwm.pwm_channel}: {power_percent:4}% -> duty {duty:.2f}%")

    def stop(self):
        self.pwm.stop()

if __name__ == '__main__':
    # Инициализация PWM
    pwm2 = HardwarePWM(pwm_channel=PWM_CHANNEL_1, hz=PWM_FREQ, chip=PWM_CHIP)
    pwm1 = HardwarePWM(pwm_channel=PWM_CHANNEL_2, hz=PWM_FREQ, chip=PWM_CHIP)

    def set_motor(pwm, power_percent):
        """
        power_percent: от -100 до 100
        """
        # Ограничение
        power_percent = max(-100, min(100, power_percent))

        # Преобразование процента в duty cycle
        if power_percent >= 0:
            duty = DUTY_STOP + (power_percent / 100.0) * (DUTY_MAX - DUTY_STOP)
        else:
            duty = DUTY_STOP + (power_percent / 100.0) * (DUTY_STOP - DUTY_MIN)

        pwm.change_duty_cycle(duty)
        print(f"Канал {pwm.pwm_channel}: {power_percent:4}% -> duty {duty:.2f}%")

    # Запуск ШИМ с начальным нулём (для инициализации)
    print("Запуск PWM и инициализация (5 секунд стоп)...")
    pwm1.start(DUTY_STOP)
    pwm2.start(DUTY_STOP)
    time.sleep(5)

    print("Инициализация завершена. Начинаем тестовый цикл.")
    try:
        while True:
            print("\n--- Полный вперёд (+100%) ---")
            set_motor(pwm1, 100)
            set_motor(pwm2, 100)
            time.sleep(3)

            print("\n--- Стоп (0%) ---")
            set_motor(pwm1, 0)
            set_motor(pwm2, 0)
            time.sleep(3)

            print("\n--- Полный назад (-100%) ---")
            set_motor(pwm1, -100)
            set_motor(pwm2, -100)
            time.sleep(3)

            print("\n--- Стоп (0%) ---")
            set_motor(pwm1, 0)
            set_motor(pwm2, 0)
            time.sleep(3)

            print("\n--- Полный поворот (-100%, 100%) ---")
            set_motor(pwm1, -100)
            set_motor(pwm2, 100)
            time.sleep(3)

            print("\n--- Стоп (0%) ---")
            set_motor(pwm1, 0)
            set_motor(pwm2, 0)
            time.sleep(3)

            print("\n--- Полный поворот (100%, -100%) ---")
            set_motor(pwm1, 100)
            set_motor(pwm2, -100)
            time.sleep(3)

            print("\n--- Стоп (0%) ---")
            set_motor(pwm1, 0)
            set_motor(pwm2, 0)
            time.sleep(3)
    except KeyboardInterrupt:
        print("\nПрограмма остановлена пользователем.")
    finally:
        # Останавливаем сигналы и выключаем PWM
        print("Остановка моторов и выход...")
        set_motor(pwm1, 0)
        set_motor(pwm2, 0)
        time.sleep(1)
        pwm1.stop()
        pwm2.stop()
        print("Готово.")
