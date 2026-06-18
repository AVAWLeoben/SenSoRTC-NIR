import serial
import time


def create_serial(port: str = "/dev/ttyCH341USB0", baudrate: int = 115200, timeout: float = 1.0):
    """Create and open a serial connection to the Arduino."""
    try:
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        time.sleep(2.0)  # Arduino often resets when serial opens
        return ser
    except Exception as e:
        print(f"Error while connecting to Arduino: {e}")


def send_arduino_telegram(message, ser: serial.Serial):
    if ser is None or not ser.is_open:
        return

    try:
        normalized = _normalize_message(message)
        byte_val = int(normalized, 2)
        ser.write(bytes([byte_val]))
    except Exception as e:
        print(f"Error while sending to Arduino: {e}")

def close_serial(ser: serial.Serial):
    """Close the serial connection cleanly."""
    try:
        if ser is not None and ser.is_open:
            close_message = '00000000'
            normalized = _normalize_message(close_message)
            send_arduino_telegram(normalized, ser)
            ser.close()
    except Exception as e:
        print(f"Error while closing serial connection: {e}")


def _normalize_message(message) -> str:
    """Convert supported message formats into an 8-character 0/1 string."""
    if isinstance(message, bytes):
        message = message.decode("ascii", errors="ignore")

    if isinstance(message, str):
        message = message.strip()
        if len(message) != 8 or any(c not in ("0", "1") for c in message):
            raise ValueError(f"Invalid message '{message}'. Expected exactly 8 characters of 0/1.")
        return message

    if isinstance(message, (list, tuple)):
        if len(message) != 8:
            raise ValueError(f"Invalid message length {len(message)}. Expected 8 values.")
        if any(v not in (0, 1, "0", "1", False, True) for v in message):
            raise ValueError(f"Invalid list/tuple values: {message}")
        return "".join("1" if int(v) else "0" for v in message)

    raise TypeError(f"Unsupported message type: {type(message)}")