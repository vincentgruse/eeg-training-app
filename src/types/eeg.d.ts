export interface EEGDevice {
    id: string;
    name: string;
    serialNumber?: string;
    status: 'connected' | 'disconnected' | 'streaming';
  }
  
  export interface EEGChannel {
    id: number;
    name: string;
    enabled: boolean;
  }
  
  export interface EEGData {
    timestamp: number;
    channels: number[];
    markers?: string[];
  }
  
  export interface RecordingSession {
    id: string;
    startTime: Date;
    endTime?: Date;
    participantId: string;
    experimentType: string;
    data: EEGData[];
  }
  
  export interface Stimulus {
    id: string;
    type: 'visual' | 'audio';
    content: string; // Path
    duration: number; // in milliseconds
    marker: string; // Marker to include in EEG data
  }