#!/usr/bin/env bash
set -euo pipefail

# Create a stable /dev/motor_serial alias for the WHEELTEC MS42DC gripper.
# The vendor example script contains a placeholder serial number. This script
# detects the serial number of the currently connected CH9102 adapter instead.
#
# Usage:
#   sudo ./scripts/setup_ms42dc_udev.sh

if [[ ${EUID} -ne 0 ]]; then
    echo "请使用 sudo 运行：sudo $0" >&2
    exit 1
fi

mapfile -t candidates < <(
    find /dev/serial/by-id -maxdepth 1 -type l \
        -name 'usb-1a86_USB_Single_Serial_*-if00' -print 2>/dev/null
)

if (( ${#candidates[@]} == 0 )); then
    echo "未找到 CH9102 串口。请确认夹爪 USB 已连接。" >&2
    exit 1
fi
if (( ${#candidates[@]} > 1 )); then
    echo "检测到多个 CH9102 串口，请只连接夹爪后重试：" >&2
    printf '  %s\n' "${candidates[@]}" >&2
    exit 1
fi

device=${candidates[0]}
serial=$(udevadm info --query=property --name="$device" | sed -n 's/^ID_SERIAL_SHORT=//p')
if [[ -z "$serial" ]]; then
    echo "无法读取 USB 序列号：$device" >&2
    exit 1
fi

rule=/etc/udev/rules.d/99-ms42dc-gripper.rules
cat > "$rule" <<RULE
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d4", ATTRS{serial}=="$serial", MODE="0660", GROUP="dialout", SYMLINK+="motor_serial"
RULE

udevadm control --reload-rules
udevadm trigger --action=change --subsystem-match=tty

if [[ -n ${SUDO_USER:-} ]] && ! id -nG "$SUDO_USER" | tr ' ' '\n' | grep -qx dialout; then
    usermod -aG dialout "$SUDO_USER"
    echo "已将用户 $SUDO_USER 加入 dialout 组；重新登录后永久生效。"
fi

echo "USB 序列号: $serial"
echo "已安装规则: $rule"
echo "当前会话可用以下命令验证："
echo "  sg dialout -c 'ls -l /dev/motor_serial'"
