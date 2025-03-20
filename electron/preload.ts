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