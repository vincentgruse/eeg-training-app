import numpy as np
import serial
import time
import threading
import logging
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bci_controller.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BCI_Controller")

class BCIController:
    def __init__(self, model_path=config.MODEL_PATH, 
                 esp32_port=config.ESP32_SERIAL_PORT, 
                 esp32_baud=config.ESP32_BAUD_RATE,
                 board_port=config.BOARD_SERIAL_PORT,
                 confidence_threshold=config.CONFIDENCE_THRESHOLD):
        self.model = self.load_model(model_path)
        self.esp32_port = esp32_port
        self.esp32_baud = esp32_baud
        self.board_port = board_port
        self.confidence_threshold = confidence_threshold
        self.command_map = config.COMMAND_MAP
        
        # Initialize variables
        self.is_running = False
        self.last_command_time = 0
        self.command_cooldown = config.COMMAND_COOLDOWN
        
        # Set up board params
        self.board_id = BoardIds.CYTON_BOARD
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.buffer_size = self.sampling_rate  # 1 second of data
        
        # ESP32 connection
        self.esp32 = None
    
    # Connect to the ESP32 via serial
    def connect_esp32(self):
        try:
            self.esp32 = serial.Serial(self.esp32_port, self.esp32_baud, timeout=1)
            logger.info(f"Connected to ESP32 on {self.esp32_port}")
            time.sleep(2)  # Allow time for connection to establish
            response = self.send_command("status")  # Check connection
            return True
        except Exception as e:
            logger.error(f"Failed to connect to ESP32: {e}")
            self.esp32 = None
            return False
    
    # Load ML model
    def load_model(self, model_path):
        try:
            # This is a placeholder - replace with model loading code
            # For a dummy model that returns random predictions:
            class DummyModel:
                def __call__(self, data):
                    directions = ['FORWARD', 'BACKWARD', 'LEFT', 'RIGHT', 'STOP']
                    # Generate random probabilities for each direction
                    probs = np.random.random(5)
                    return dict(zip(directions, probs))
            
            logger.info(f"Loaded model from {model_path}")
            return DummyModel()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    # Process cyton board data
    def process_board_data(self, data):
        try:
            # Extract EEG data (only the channels we need)
            eeg_data = data[self.eeg_channels, :]
            
            # Apply preprocessing if we end up needing (bandpass filter)
            for channel in range(eeg_data.shape[0]):
                DataFilter.perform_bandpass(
                    eeg_data[channel, :],
                    self.sampling_rate,
                    config.FILTER_CENTER_FREQ,
                    config.FILTER_BANDWIDTH,
                    config.FILTER_ORDER,
                    0,  # filter type
                    0.0 # ripple
                )
            
            # Run model inference
            self.run_inference(eeg_data)
        except Exception as e:
            logger.error(f"Error processing board data: {e}")
    
    # Run the ML model on the EEG data
    def run_inference(self, eeg_data):
        try:
            # Run inference
            predictions = self.model(eeg_data)
            
            # Log predictions for debugging
            logger.debug(f"Predictions: {predictions}")
            
            # Check if any prediction exceeds the threshold
            current_time = time.time()
            if current_time - self.last_command_time >= self.command_cooldown:
                for direction, probability in predictions.items():
                    if probability >= self.confidence_threshold:
                        self.send_command(direction)
                        self.last_command_time = current_time
                        break  # Only send one command at a time
        except Exception as e:
            logger.error(f"Error during inference: {e}")
    
    # Send command to ESP32
    def send_command(self, direction):
        if self.esp32 is None:
            logger.error("Cannot send command - no connection to ESP32")
            return None
        
        try:
            # Get the command string from the map
            if direction in self.command_map:
                command = self.command_map[direction]
                logger.info(f"Sending command: {direction}")
                self.esp32.write(command.encode())
            else:
                # For other commands, send directly
                logger.info(f"Sending direct command: {direction}")
                self.esp32.write(f"{direction}\n".encode())
            
            # Read response
            response = self.esp32.readline().decode().strip()
            logger.info(f"ESP32 response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return None
    
    # Start EEG data acquisition using BrainFlow
    def start_acquisition(self):
        if self.is_running:
            logger.warning("Data acquisition already running")
            return False
        
        # Make sure ESP32 is connected
        if self.esp32 is None:
            if not self.connect_esp32():
                logger.error("Cannot start acquisition - ESP32 connection failed")
                return False
        
        self.is_running = True
        logger.info("Starting EEG data acquisition")
        
        # Start Cyton board in a separate thread
        def start_board():
            board = None
            try:
                # Initialize board parameters
                params = BrainFlowInputParams()
                params.serial_port = self.board_port
                
                # Initialize board
                board = BoardShim(self.board_id, params)
                logger.info("Preparing board session...")
                board.prepare_session()
                
                # Start streaming
                logger.info("Starting board stream...")
                board.start_stream()
                
                # Process data while running
                while self.is_running:
                    # Check if data is available
                    data_count = board.get_board_data_count()
                    if data_count >= self.buffer_size:
                        # Get data and process it
                        data = board.get_current_board_data(self.buffer_size)
                        self.process_board_data(data)
                    else:
                        # Small sleep to prevent CPU hogging
                        time.sleep(0.1)
                
                # Clean up
                logger.info("Stopping board stream...")
                board.stop_stream()
                board.release_session()
                logger.info("Board session released")
                
            except Exception as e:
                logger.error(f"Error in board acquisition: {e}")
                self.is_running = False
                # Clean up if board was initialized
                if board is not None:
                    try:
                        board.stop_stream()
                        board.release_session()
                    except:
                        pass
        
        self.acquisition_thread = threading.Thread(target=start_board)
        self.acquisition_thread.daemon = True
        self.acquisition_thread.start()
        return True
    
    # Stop data aquisition
    def stop_acquisition(self):
        if not self.is_running:
            logger.warning("Data acquisition not running")
            return
        
        logger.info("Stopping EEG data acquisition")
        self.is_running = False
        
        # Wait for acquisition thread to finish
        if hasattr(self, 'acquisition_thread') and self.acquisition_thread.is_alive():
            self.acquisition_thread.join(timeout=5)
        
        # Close ESP32 connection
        if self.esp32 is not None:
            self.send_command("stop")  # Stop any movement
            self.esp32.close()
            logger.info("Closed ESP32 connection")

def main():
    try:
        # Initialize controller
        controller = BCIController()
        
        # Start dashboard bridge
        from dashboard_bridge import DashboardBridge
        dashboard = DashboardBridge(controller)
        dashboard.start_in_thread()
        
        # Start data acquisition
        if controller.start_acquisition():
            # Keep the script running
            logger.info("BCI Controller running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        else:
            logger.error("Failed to start acquisition")
            
    except KeyboardInterrupt:
        logger.info("Stopping BCI Controller")
        if 'controller' in locals():
            controller.stop_acquisition()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if 'controller' in locals():
            controller.stop_acquisition()

if __name__ == "__main__":
    main()