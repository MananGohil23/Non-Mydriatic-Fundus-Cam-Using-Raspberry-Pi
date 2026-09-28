import threading
import time

try:
    from gpiozero import LED, Button

    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False


class LedController:
    MODE_IR = "ir"
    MODE_WHITE = "white"
    MODE_OFF = "off"

    def __init__(self, ir_pin, white_pin, mock=False):
        self.ir_pin = ir_pin
        self.white_pin = white_pin
        self.mock = bool(mock) or not GPIO_AVAILABLE
        self._lock = threading.RLock()
        self._mode = self.MODE_OFF
        self._ir = None
        self._white = None
        if not self.mock:
            self._ir = LED(ir_pin)
            self._white = LED(white_pin)
        self._apply(self.MODE_OFF)

    def _apply(self, mode):
        if not self.mock:
            if mode == self.MODE_IR:
                self._white.off()
                self._ir.on()
            elif mode == self.MODE_WHITE:
                self._ir.off()
                self._white.on()
            else:
                self._ir.off()
                self._white.off()

    def set_mode(self, mode):
        if mode not in (self.MODE_IR, self.MODE_WHITE, self.MODE_OFF):
            raise ValueError("mode must be ir, white or off")
        with self._lock:
            self._apply(mode)
            self._mode = mode
            return self._mode

    def get_mode(self):
        with self._lock:
            return self._mode

    def flash_white(self, duration_ms):
        with self._lock:
            self._apply(self.MODE_WHITE)
            self._mode = self.MODE_WHITE
        time.sleep(max(0, duration_ms) / 1000.0)
        with self._lock:
            self._apply(self.MODE_IR)
            self._mode = self.MODE_IR

    def status(self):
        return {
            "mode": self.get_mode(),
            "mock": self.mock,
            "gpio_available": GPIO_AVAILABLE,
            "ir_pin": self.ir_pin,
            "white_pin": self.white_pin,
        }

    def close(self):
        with self._lock:
            self._apply(self.MODE_OFF)
            for led in (self._ir, self._white):
                if led is not None:
                    try:
                        led.close()
                    except Exception:
                        pass


class ShutterButton:
    def __init__(self, pin, callback, mock=False):
        self.pin = pin
        self.callback = callback
        self.mock = bool(mock) or not GPIO_AVAILABLE
        self._button = None
        if not self.mock:
            self._button = Button(pin, pull_up=True, bounce_time=0.05)
            self._button.when_pressed = callback

    def status(self):
        return {
            "enabled": not self.mock,
            "pin": self.pin,
            "mock": self.mock,
            "pressed": bool(self._button and self._button.is_pressed),
        }

    def close(self):
        if self._button is not None:
            try:
                self._button.close()
            except Exception:
                pass
