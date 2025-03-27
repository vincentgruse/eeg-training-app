import { contextBridge, ipcRenderer } from 'electron'

// Expose protected methods for stimulus presentation
contextBridge.exposeInMainWorld('stimulusApi', {
  // Save data to file
  saveSessionData: (data: string, filename: string) => {
    return ipcRenderer.invoke('stimulus:saveSessionData', data, filename);
  }
})

// For main process messages
contextBridge.exposeInMainWorld('electronAPI', {
  onMainProcessMessage: (callback: (message: string) => void) => {
    ipcRenderer.on('main-process-message', (_event, message) => callback(message));
  }
})

// Expose EEG processing methods
contextBridge.exposeInMainWorld('eegProcessorApi', {
  // Browse for EEG data file
  browseForEegFile: () => {
    return ipcRenderer.invoke('eeg:browseForFile');
  },
  
  // Process EEG data with a specific file
  processEEGData: (filePath: string) => {
    return ipcRenderer.invoke('eeg:processData', filePath);
  }
});