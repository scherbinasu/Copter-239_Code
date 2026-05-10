#!/bin/bash

# ============================================
# Скрипт настройки точки доступа Wi-Fi
# для Raspberry Pi 4 с Ubuntu 25 (и выше)
# Запуск: sudo bash setup_ap.sh
# ============================================

set -e  # Прерывать выполнение при ошибке

# ---------- Параметры (можно изменить) ----------
AP_SSID="Copter239"
AP_PASSWORD="copter239"
AP_IP="10.239.239.239"
AP_NETMASK="255.255.255.0"
AP_DHCP_RANGE_START="10.239.239.100"
AP_DHCP_RANGE_END="10.239.239.200"
WLAN_IFACE="wlan0"
# ------------------------------------------------

# Проверка прав root
if [[ $EUID -ne 0 ]]; then
   echo "Этот скрипт нужно запускать с правами root (sudo)."
   exit 1
fi

echo ">>> Начинаем настройку точки доступа Wi-Fi <<<"
echo "SSID: $AP_SSID"
echo "Пароль: $AP_PASSWORD"
echo "IP-адрес точки доступа: $AP_IP"

# ---------- 1. Обновление системы и установка пакетов ----------
echo ">>> Установка необходимых пакетов..."
apt update
apt install -y hostapd dnsmasq iw rfkill netfilter-persistent iptables-persistent

# ---------- 2. Остановка и отключение конфликтующих сервисов ----------
echo ">>> Отключение NetworkManager, wpa_supplicant и systemd-resolved..."
systemctl stop NetworkManager 2>/dev/null || true
systemctl disable NetworkManager 2>/dev/null || true
systemctl stop wpa_supplicant 2>/dev/null || true
systemctl mask wpa_supplicant 2>/dev/null || true
systemctl stop systemd-resolved 2>/dev/null || true
systemctl disable systemd-resolved 2>/dev/null || true

# Удаляем старый resolv.conf и прописываем Google DNS
rm -f /etc/resolv.conf
echo "nameserver 8.8.8.8" > /etc/resolv.conf

# ---------- 3. Очистка предыдущих конфигураций (если были) ----------
echo ">>> Очистка старых настроек..."
# Удаляем возможные конфиги systemd-networkd для wlan0
rm -f /etc/systemd/network/10-wlan0.network
# Останавливаем hostapd и dnsmasq, если запущены
systemctl stop hostapd 2>/dev/null || true
systemctl stop dnsmasq 2>/dev/null || true

# ---------- 4. Перевод wlan0 в режим точки доступа ----------
echo ">>> Настройка беспроводного интерфейса $WLAN_IFACE..."
# Разблокируем Wi-Fi (на всякий случай)
rfkill unblock wifi
# Отключаем интерфейс
ip link set $WLAN_IFACE down
# Переводим в режим AP
iw dev $WLAN_IFACE set type ap
# Назначаем статический IP
ip addr flush dev $WLAN_IFACE
ip addr add $AP_IP/24 dev $WLAN_IFACE
# Включаем интерфейс
ip link set $WLAN_IFACE up

# ---------- 5. Настройка hostapd ----------
echo ">>> Конфигурирование hostapd..."
cat > /etc/hostapd/hostapd.conf << EOF
interface=$WLAN_IFACE
driver=nl80211
ssid=$AP_SSID
hw_mode=g
channel=7
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase=$AP_PASSWORD
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP
EOF

# Указываем путь к конфигу в настройках демона
sed -i 's|^#DAEMON_CONF=""|DAEMON_CONF="/etc/hostapd/hostapd.conf"|' /etc/default/hostapd

# ---------- 6. Настройка dnsmasq ----------
echo ">>> Конфигурирование dnsmasq (DHCP-сервер)..."
# Сохраняем оригинальный конфиг
mv /etc/dnsmasq.conf /etc/dnsmasq.conf.backup 2>/dev/null || true
cat > /etc/dnsmasq.conf << EOF
interface=$WLAN_IFACE
dhcp-range=$AP_DHCP_RANGE_START,$AP_DHCP_RANGE_END,$AP_NETMASK,24h
dhcp-option=3,$AP_IP
dhcp-option=6,8.8.8.8,8.8.4.4
no-resolv
server=8.8.8.8
EOF

# ---------- 7. Включение IP-форвардинга ----------
echo ">>> Включение IP-маршрутизации..."
# Создаём отдельный конфиг для гарантированного применения
echo "net.ipv4.ip_forward=1" > /etc/sysctl.d/99-ipforward.conf
sysctl -p /etc/sysctl.d/99-ipforward.conf

# ---------- 8. Настройка NAT (для доступа в интернет) ----------
echo ">>> Настройка правил iptables (NAT)..."

# Определяем внешний интерфейс (Ethernet или USB-модем)
# Приоритет: сначала смотрим USB-подобные (enx...), затем eth0
EXT_IFACE=""
for iface in $(ls /sys/class/net/); do
    if [[ "$iface" =~ ^enx[0-9a-f]+ ]] || [[ "$iface" == "eth0" ]]; then
        # Проверяем, что интерфейс поднят и имеет IP
        if ip link show "$iface" | grep -q "state UP" && ip addr show "$iface" | grep -q "inet "; then
            EXT_IFACE="$iface"
            break
        fi
    fi
done

# Если не нашли активный интерфейс, берём первый попавшийся (потом можно будет вручную исправить)
if [[ -z "$EXT_IFACE" ]]; then
    EXT_IFACE="eth0"
    echo "Предупреждение: не найден активный внешний интерфейс. Использую $EXT_IFACE (возможно, потребуется изменить вручную)."
else
    echo "Внешний интерфейс для выхода в интернет: $EXT_IFACE"
fi

# Очищаем старые правила (осторожно, чтобы не сломать SSH!)
iptables -t nat -F POSTROUTING
iptables -F FORWARD

# Добавляем правило MASQUERADE
iptables -t nat -A POSTROUTING -o $EXT_IFACE -j MASQUERADE
# Разрешаем форвардинг между wlan0 и внешним интерфейсом
iptables -A FORWARD -i $WLAN_IFACE -o $EXT_IFACE -j ACCEPT
iptables -A FORWARD -i $EXT_IFACE -o $WLAN_IFACE -m state --state RELATED,ESTABLISHED -j ACCEPT

# Сохраняем правила
netfilter-persistent save

# ---------- 9. Создание systemd-сервиса для перевода wlan0 в режим AP при загрузке ----------
echo ">>> Создание сервиса для автоматического перевода wlan0 в режим AP..."
cat > /etc/systemd/system/wlan0-ap-mode.service << EOF
[Unit]
Description=Set wlan0 to AP mode
Before=hostapd.service

[Service]
Type=oneshot
ExecStart=/usr/sbin/iw dev $WLAN_IFACE set type ap
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

# Включаем сервис
systemctl enable wlan0-ap-mode.service

# ---------- 10. Запуск и включение сервисов ----------
echo ">>> Запуск hostapd и dnsmasq..."
systemctl unmask hostapd
systemctl enable hostapd
systemctl start hostapd

systemctl enable dnsmasq
systemctl start dnsmasq

# ---------- 11. Финальная проверка ----------
echo ""
echo "==============================================="
echo "Настройка завершена!"
echo "Точка доступа Wi-Fi '$AP_SSID' должна быть активна."
echo "IP-адрес Raspberry Pi в этой сети: $AP_IP"
echo "Подключайтесь с паролем: $AP_PASSWORD"
echo ""
echo "Для проверки статуса используйте:"
echo "  systemctl status hostapd"
echo "  systemctl status dnsmasq"
echo ""
echo "Если точка доступа не появилась, перезагрузите устройство: sudo reboot"
echo "==============================================="