from gpiozero import LED, Button
from signal import pause

# Define the GPIO pins based on the wiring above
button = Button(17, pull_up=True) # Uses internal pull-up resistor
ir_led = LED(27)
white_led = LED(22)

# Define what happens when the button is pressed
def turn_on_leds():
    print("Button pressed: LEDs ON")
    ir_led.on()
    white_led.on()

# Define what happens when the button is released
def turn_off_leds():
    print("Button released: LEDs OFF")
    ir_led.off()
    white_led.off()

# Link the actions to the button state
button.when_pressed = turn_on_leds
button.when_released = turn_off_leds

print("Program running. Press the button...")

# Keep the script running indefinitely to listen for the button
pause()
