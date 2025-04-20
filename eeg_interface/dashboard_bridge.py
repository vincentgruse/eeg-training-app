import websockets
import asyncio
import json
import threading
import logging
import numpy as np
import time

logger = logging.getLogger("Dashboard_Bridge")

class DashboardBridge:
    def __init__(self, bci_controller, host="localhost", port=8765):
        self.bci_controller = bci_controller
        self.host = host
        self.port = port
        self.connected_clients = set()
        self.running = False
        logger.info(f"Dashboard bridge initialized at ws://{host}:{port}")
        
        # EEG channel names (8 channels)
        self.channel_names = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']

    async def send_data(self):
        while self.running:
            try:
                # Get data from BCI controller
                eeg_data = self.get_latest_eeg_data()
                classification_data = self.get_latest_classification()
                system_status = self.get_system_status()
                
                data = {
                    "eegData": eeg_data,
                    "classificationData": classification_data,
                    "systemStatus": system_status
                }
                
                # Send to all connected clients
                if self.connected_clients:
                    await asyncio.gather(
                        *[client.send(json.dumps(data)) for client in self.connected_clients]
                    )
            except Exception as e:
                logger.error(f"Error in send_data: {e}")
            
            await asyncio.sleep(0.1)  # Update 10 times per second

    async def handler(self, websocket, path):
        logger.info(f"New client connected")
        self.connected_clients.add(websocket)
        try:
            async for message in websocket:
                # Handle any commands from the dashboard
                try:
                    data = json.loads(message)
                    if 'command' in data:
                        await self.handle_command(data['command'], data.get('params', {}))
                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON: {message}")
        except Exception as e:
            logger.error(f"Error in websocket handler: {e}")
        finally:
            self.connected_clients.remove(websocket)
            logger.info(f"Client disconnected")

    async def handle_command(self, command, params):
        logger.info(f"Received command: {command} with params: {params}")
        try:
            if command == "send_command":
                direction = params.get("direction")
                if direction and hasattr(self.bci_controller, 'send_command'):
                    self.bci_controller.send_command(direction)
            elif command == "reset":
                if hasattr(self.bci_controller, 'stop_acquisition'):
                    self.bci_controller.stop_acquisition()
                    time.sleep(1)
                    self.bci_controller.start_acquisition()
        except Exception as e:
            logger.error(f"Error handling command {command}: {e}")

    def get_latest_eeg_data(self):
        """Extract EEG data from the BCI controller in a format suitable for the dashboard"""
        result = []
        
        # Check if the latest data is available in the BCI controller
        if hasattr(self.bci_controller, 'board') and self.bci_controller.is_running:
            try:
                # Get the latest data from the board
                data_count = self.bci_controller.board.get_board_data_count()
                if data_count > 50:  # Ensure we have enough data points
                    # Get the latest 50 data points
                    data = self.bci_controller.board.get_current_board_data(50)
                    
                    # Process each channel
                    for i, channel_name in enumerate(self.channel_names):
                        if i < len(self.bci_controller.eeg_channels):
                            channel_idx = self.bci_controller.eeg_channels[i]
                            if channel_idx < data.shape[0]:
                                # Get channel data and convert to list of time-value pairs
                                channel_data = data[channel_idx, :]
                                values = []
                                
                                now = time.time() * 1000  # Current time in ms
                                sample_interval = 1000 / self.bci_controller.sampling_rate
                                
                                for j in range(len(channel_data)):
                                    values.append({
                                        "time": now - (len(channel_data) - j) * sample_interval,
                                        "value": float(channel_data[j])
                                    })
                                
                                result.append({
                                    "channel": channel_name,
                                    "values": values
                                })
            except Exception as e:
                logger.error(f"Error getting EEG data: {e}")
        
        # If no data available, return empty channels with dummy data
        if not result:
            current_time = time.time() * 1000
            for channel_name in self.channel_names:
                values = []
                for i in range(50):
                    values.append({
                        "time": current_time - (50 - i) * 20,
                        "value": 0  # Zero value for no data
                    })
                result.append({
                    "channel": channel_name,
                    "values": values
                })
        
        return result

    def get_latest_classification(self):
        """Get the latest classification data from the BCI controller"""
        default_data = {
            'FORWARD': 0.2,
            'BACKWARD': 0.2,
            'LEFT': 0.2,
            'RIGHT': 0.2,
            'STOP': 0.2
        }
        
        try:
            # Try to get the latest prediction from the model
            if hasattr(self.bci_controller, 'last_prediction') and self.bci_controller.last_prediction:
                return self.bci_controller.last_prediction
        except Exception as e:
            logger.error(f"Error getting classification data: {e}")
        
        return default_data

    def get_system_status(self):
        """Get the current system status"""
        try:
            board_connected = self.bci_controller.is_running
            esp32_connected = hasattr(self.bci_controller, 'esp32') and self.bci_controller.esp32 is not None
            
            # Try to get the last command
            last_command = "NONE"
            if hasattr(self.bci_controller, 'last_command') and self.bci_controller.last_command:
                last_command = self.bci_controller.last_command
            
            # Get distance if available
            distance = 100  # Default value
            obstacle_detected = False
            
            # Get command confidence
            command_confidence = 0.0
            if hasattr(self.bci_controller, 'last_prediction') and self.bci_controller.last_prediction:
                confidences = list(self.bci_controller.last_prediction.values())
                if confidences:
                    command_confidence = max(confidences)
            
            # Get speed setting
            speed = 150
            if hasattr(self.bci_controller, 'command_map') and 'FORWARD' in self.bci_controller.command_map:
                speed = self.bci_controller.command_map['FORWARD']
            
            return {
                "boardConnected": board_connected,
                "esp32Connected": esp32_connected,
                "signalQuality": "Good",
                "batteryLevel": 78,  # Placeholder, replace with actual value if available
                "lastCommand": last_command,
                "obstacleDetected": obstacle_detected,
                "distance": distance,
                "speed": speed,
                "commandConfidence": command_confidence
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            
            # Return default values on error
            return {
                "boardConnected": False,
                "esp32Connected": False,
                "signalQuality": "Poor",
                "batteryLevel": 0,
                "lastCommand": "ERROR",
                "obstacleDetected": False,
                "distance": 0,
                "speed": 0,
                "commandConfidence": 0
            }

    async def start_server(self):
        self.running = True
        server = await websockets.serve(self.handler, self.host, self.port)
        logger.info(f"Dashboard WebSocket server started at ws://{self.host}:{self.port}")
        
        try:
            await self.send_data()
        finally:
            self.running = False
            server.close()
            await server.wait_closed()

    def start(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.start_server())
        loop.close()

    def start_in_thread(self):
        thread = threading.Thread(target=self.start, daemon=True)
        thread.start()
        logger.info("Dashboard bridge started in background thread")
        return thread