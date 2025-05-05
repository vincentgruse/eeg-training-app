import { BCIData } from '../types';

class BCIDataService {
  private url: string;
  private socket: WebSocket | null;
  private reconnectTimer: number | null;
  private callbacks: {
    onData: (data: BCIData) => void;
    onConnect: () => void;
    onDisconnect: () => void;
  };

  constructor(url = 'ws://localhost:8765') {
    this.url = url;
    this.socket = null;
    this.reconnectTimer = null;
    this.callbacks = {
      onData: () => {},
      onConnect: () => {},
      onDisconnect: () => {},
    };
  }

  public connect(): void {
    // Clear any existing reconnect timer
    if (this.reconnectTimer) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    // Create new WebSocket connection
    this.socket = new WebSocket(this.url);

    this.socket.onopen = () => {
      console.log('Connected to BCI server');
      this.callbacks.onConnect();
    };

    this.socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as BCIData;
        this.callbacks.onData(data);
      } catch (e) {
        console.error('Error parsing data:', e);
      }
    };

    this.socket.onclose = () => {
      console.log('Disconnected from BCI server');
      this.callbacks.onDisconnect();
      
      // Try to reconnect after a delay
      this.reconnectTimer = window.setTimeout(() => this.connect(), 3000);
    };

    this.socket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  public onData(callback: (data: BCIData) => void): void {
    this.callbacks.onData = callback;
  }

  public onConnect(callback: () => void): void {
    this.callbacks.onConnect = callback;
  }

  public onDisconnect(callback: () => void): void {
    this.callbacks.onDisconnect = callback;
  }

  public disconnect(): void {
    if (this.socket) {
      this.socket.close();
    }
    
    if (this.reconnectTimer) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  public sendCommand(command: string, params: Record<string, any> = {}): void {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({
        command,
        params
      }));
    } else {
      console.error('Cannot send command: WebSocket is not connected');
    }
  }
}

const bciDataService = new BCIDataService();
export default bciDataService;