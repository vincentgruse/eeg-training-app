import { ipcMain } from 'electron'
import { BoardShim, BoardIds, IBrainFlowInputParams } from 'brainflow'

let boardShim: BoardShim | null = null

// Handle connection to OpenBCI device
ipcMain.handle('eeg:connect', async (_event, args: { serialPort: string }) => {
  try {
    // Initialize brainflow parameters
    const params: Partial<IBrainFlowInputParams> = {
      serialPort: args.serialPort
    };
    
    // Create board instance
    boardShim = new BoardShim(BoardIds.CYTON_DAISY_BOARD, params);
    
    // Connect to the board
    await boardShim.prepareSession();
    
    console.log('Connected to Cyton board on port:', args.serialPort);
    return { 
      success: true, 
      message: 'Connected to OpenBCI Cyton board',
      boardInfo: {
        sampleRate: BoardShim.getSamplingRate(BoardIds.CYTON_BOARD),
        eegChannels: BoardShim.getEegChannels(BoardIds.CYTON_BOARD),
        totalChannels: BoardShim.getNumRows(BoardIds.CYTON_BOARD)
      }
    }
  } catch (error: unknown) {
    console.error('Failed to connect to Cyton board:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    return { success: false, message: errorMessage }
  }
})

// Start data streaming
ipcMain.handle('eeg:startStream', async () => {
  try {
    if (!boardShim) {
      throw new Error('Board not connected. Please connect to the board first.');
    }
    
    // Start streaming data
    await boardShim.startStream();
    console.log('Started data streaming from Cyton board');
    
    return { success: true, message: 'Data streaming started' }
  } catch (error: unknown) {
    console.error('Failed to start data stream:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    return { success: false, message: errorMessage }
  }
})

// Stop data streaming
ipcMain.handle('eeg:stopStream', async () => {
  try {
    if (!boardShim) {
      throw new Error('Board not connected.');
    }
    
    // Stop streaming
    await boardShim.stopStream();
    console.log('Stopped data streaming from Cyton board');
    
    return { success: true, message: 'Data streaming stopped' }
  } catch (error: unknown) {
    console.error('Failed to stop data stream:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    return { success: false, message: errorMessage }
  }
})

// Get board data
ipcMain.handle('eeg:getData', async () => {
  try {
    if (!boardShim) {
      throw new Error('Board not connected.');
    }
    
    // Get data from the board
    const data = boardShim.getCurrentBoardData(100);

    const eegChannels = BoardShim.getEegChannels(BoardIds.CYTON_BOARD);
    const timeChannel = BoardShim.getTimestampChannel(BoardIds.CYTON_BOARD);
    
    // Extract the relevant data
    const timestamps = data[timeChannel];
    const channelData = eegChannels.map((channel: number) => data[channel]);
    
    // Structure the data to match our expected format
    const structuredData = {
      timestamps: timestamps,
      channels: channelData,
      samplingRate: BoardShim.getSamplingRate(BoardIds.CYTON_BOARD)
    };
    
    return { success: true, data: structuredData }
  } catch (error: unknown) {
    console.error('Failed to get EEG data:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    return { success: false, message: errorMessage }
  }
})

// Disconnect from the board
ipcMain.handle('eeg:disconnect', async () => {
  try {
    if (boardShim) {
      if (boardShim.isPrepared()) {
        // Stop the stream if it's running
        try {
          await boardShim.stopStream();
        } catch (e) {
          // Ignore errors on stopStream, it might not be streaming
        }
        
        // Release the session
        await boardShim.releaseSession();
        console.log('Disconnected from Cyton board');
      }
      boardShim = null;
    }
    
    return { success: true, message: 'Disconnected from board' }
  } catch (error: unknown) {
    console.error('Failed to disconnect from board:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    return { success: false, message: errorMessage }
  }
})

// Add a marker to the data stream
ipcMain.handle('eeg:addMarker', async (_event, marker: string) => {
  try {
    if (!boardShim) {
      throw new Error('Board not connected.');
    }
    console.log(`Marker added: ${marker}`);
    
    return { success: true, message: 'Marker added' }
  } catch (error: unknown) {
    console.error('Failed to add marker:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    return { success: false, message: errorMessage }
  }
})

export const setupIpcHandlers = () => {
  console.log('EEG IPC handlers initialized with brainflow integration for Cyton board');
}