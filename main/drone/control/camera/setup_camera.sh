#!/bin/bash

# Скрипт для автоматической настройки камеры на Raspberry Pi (Raspberry Pi OS / Ubuntu)
# Запускать с правами sudo: sudo bash setup_camera.sh

set -e # Останавливает скрипт при любой ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_good() {
    echo -e "${GREEN}[✓] $1${NC}"
}

print_bad() {
    echo -e "${RED}[✗] $1${NC}"
}

print_info() {
    echo -e "${YELLOW}[➜] $1${NC}"
}

# Проверка прав root
if [[ $EUID -ne 0 ]]; then
   print_bad "Этот скрипт должен запускаться с правами root (sudo)."
   exit 1
fi

print_info "Начинаю автоматическую настройку камеры Raspberry Pi..."

# 1. Обновление системы
print_info "Обновление списка пакетов..."
apt update

print_info "Обновление установленных пакетов..."
apt upgrade -y

# 2. Установка необходимых пакетов
print_info "Установка необходимых пакетов..."
apt install -y python3-picamera2 python3-opencv v4l-utils i2c-tools

# 3. Настройка config.txt
CONFIG_FILE="/boot/firmware/config.txt"
if [ ! -f "$CONFIG_FILE" ]; then
    CONFIG_FILE="/boot/config.txt" # Альтернативный путь для старых версий ОС
fi

print_info "Настройка файла $CONFIG_FILE..."

# Функция для добавления или изменения параметра в config.txt
set_config_var() {
    local VAR=$1
    local VALUE=$2
    local CONFIG=$3
    if grep -q "^$VAR=" "$CONFIG"; then
        sed -i "s/^$VAR=.*/$VAR=$VALUE/" "$CONFIG"
    elif grep -q "^#$VAR=" "$CONFIG"; then
        sed -i "s/^#$VAR=.*/$VAR=$VALUE/" "$CONFIG"
    else
        echo "$VAR=$VALUE" >> "$CONFIG"
    fi
}

# Добавляем/изменяем параметры для камеры
set_config_var "camera_auto_detect" "0" "$CONFIG_FILE"
set_config_var "start_x" "1" "$CONFIG_FILE"
set_config_var "gpu_mem" "128" "$CONFIG_FILE"

# Добавляем overlay для камеры (по умолчанию для v2, замените на свой при необходимости)
if ! grep -q "^dtoverlay=imx219" "$CONFIG_FILE"; then
    echo "dtoverlay=imx219" >> "$CONFIG_FILE"
fi

# Удаляем проблемную строку, если она есть
sed -i '/dtoverlay=v4l2-ctl/d' "$CONFIG_FILE"

# 4. Настройка прав доступа
print_info "Настройка прав доступа..."
# Добавляем текущего пользователя в группу video
if id -nG "$SUDO_USER" | grep -qw "video"; then
    print_info "Пользователь $SUDO_USER уже в группе video."
else
    usermod -a -G video "$SUDO_USER"
    print_good "Пользователь $SUDO_USER добавлен в группу video."
fi

# Создаем правило udev для автоматической выдачи прав на устройства камеры
UDEV_RULE="/etc/udev/rules.d/99-camera-perms.rules"
cat > "$UDEV_RULE" << EOF
# Правила для доступа к камере всем пользователям из группы video
SUBSYSTEM=="video4linux", GROUP="video", MODE="0660"
SUBSYSTEM=="media", GROUP="video", MODE="0660"
SUBSYSTEM=="vchiq", GROUP="video", MODE="0660"
SUBSYSTEM=="vc-sm", GROUP="video", MODE="0660"
SUBSYSTEM=="bcm2708_vcio", GROUP="video", MODE="0660"
EOF

# 5. Настройка автозагрузки модуля ядра
print_info "Настройка автозагрузки модуля ядра..."
MODULE_FILE="/etc/modules-load.d/camera.conf"
cat > "$MODULE_FILE" << EOF
# Модуль для поддержки камеры
bcm2835-v4l2
EOF

# 6. Завершение
print_good "Настройка завершена!"
print_info "Для применения всех изменений требуется перезагрузка."
print_info "После перезагрузки проверьте камеру командой: libcamera-hello"
print_info "Или выполните тестовый Python-скрипт: python3 -c \"from picamera2 import Picamera2; picam2 = Picamera2(); picam2.start(); picam2.capture_file('test.jpg'); print('OK')\""

# Предложение перезагрузиться
read -p "Перезагрузить систему сейчас? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    reboot
fi