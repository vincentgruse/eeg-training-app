import { contextBridge, ipcRenderer } from 'electron'

// Expose protected methods for stimulus presentation
contextBridge.exposeInMainWorld('stimulusApi', {
  // Get all image paths from the assets directory
  getImagePaths: () => ipcRenderer.invoke('stimulus:getImagePaths')
})

// For main process messages
contextBridge.exposeInMainWorld('electronAPI', {
  onMainProcessMessage: (callback: (message: string) => void) => {
    ipcRenderer.on('main-process-message', (_event, message) => callback(message));
  }
})