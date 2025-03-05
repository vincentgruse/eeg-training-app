declare global {
  interface Window {
    eegApi: {
      connect: (params: { serialPort: string }) => Promise<{ 
        success: boolean; 
        message: string;
        boardInfo?: {
          sampleRate: number;
          eegChannels: number[];
          totalChannels: number;
        }
      }>;
      startStream: () => Promise<{ success: boolean; message: string }>;
      stopStream: () => Promise<{ success: boolean; message: string }>;
      getData: () => Promise<{ 
        success: boolean; 
        data?: {
          timestamps: number[];
          channels: number[][];
          samplingRate: number;
        }; 
        message?: string 
      }>;
      disconnect: () => Promise<{ success: boolean; message: string }>;
      addMarker: (marker: string) => Promise<{ success: boolean; message: string }>;
    };
  }
}

// Helper functions to work with the IPC API
export const connectToDevice = async (serialPort: string) => {
  try {
    return await window.eegApi.connect({ serialPort });
  } catch (error: unknown) {
    console.error('Failed to connect to device:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    return { success: false, message: errorMessage };
  }
};

export const startDataStream = async () => {
  try {
    return await window.eegApi.startStream();
  } catch (error: unknown) {
    console.error('Failed to start data stream:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    return { success: false, message: errorMessage };
  }
};

export const stopDataStream = async () => {
  try {
    return await window.eegApi.stopStream();
  } catch (error: unknown) {
    console.error('Failed to stop data stream:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    return { success: false, message: errorMessage };
  }
};

export const disconnectDevice = async () => {
  try {
    return await window.eegApi.disconnect();
  } catch (error: unknown) {
    console.error('Failed to disconnect device:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    return { success: false, message: errorMessage };
  }
};

export const addMarker = async (marker: string) => {
  try {
    return await window.eegApi.addMarker(marker);
  } catch (error: unknown) {
    console.error('Failed to add marker:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    return { success: false, message: errorMessage };
  }
};

export const getLatestData = async () => {
  try {
    const response = await window.eegApi.getData();
    if (response.success && response.data) {
      return response.data;
    }
    return null;
  } catch (error: unknown) {
    console.error('Failed to get latest data:', error);
    return null;
  }
};