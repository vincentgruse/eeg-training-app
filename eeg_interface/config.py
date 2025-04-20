# OpenBCI Board Settings
BOARD_SERIAL_PORT = '/dev/ttyUSB1' # Serial port for OpenBCI board

# ESP32 Settings
ESP32_SERIAL_PORT = '/dev/ttyUSB0' # Serial port for ESP32
ESP32_BAUD_RATE = 115200 # Baud rate for ESP32 serial communication

# ML Model Settings
MODEL_PATH = '/path/to/the/model' # Path to the ML model
CONFIDENCE_THRESHOLD = 0.75 # Threshold for command execution

# Processing Settings
COMMAND_COOLDOWN = 1.5 # Time between commands (seconds)

# ESP32 Command Mapping
COMMAND_MAP = {
    'FORWARD': 'forward\n',
    'BACKWARD': 'backward\n',
    'LEFT': 'turn -90\n',
    'RIGHT': 'turn 90\n',
    'STOP': 'stop\n'
}

# Filter Settings
FILTER_CENTER_FREQ = 15.0 # Center frequency for bandpass filter
FILTER_BANDWIDTH = 8.0 # Bandwidth for bandpass filter
FILTER_ORDER = 3 # Filter order