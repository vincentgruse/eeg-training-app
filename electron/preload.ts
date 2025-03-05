import { contextBridge, ipcRenderer } from 'electron'

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('eegApi', {
  connect: (params: { serialPort: string }) => ipcRenderer.invoke('eeg:connect', params),
  startStream: () => ipcRenderer.invoke('eeg:startStream'),
  stopStream: () => ipcRenderer.invoke('eeg:stopStream'),
  getData: () => ipcRenderer.invoke('eeg:getData'),
  disconnect: () => ipcRenderer.invoke('eeg:disconnect'),
  addMarker: (marker: string) => ipcRenderer.invoke('eeg:addMarker', marker)
})

// For stimulus presentation
contextBridge.exposeInMainWorld('stimulusApi', {
  present: (stimulus: any) => ipcRenderer.invoke('stimulus:present', stimulus)
})

// For recording
contextBridge.exposeInMainWorld('recordingApi', {
  startRecording: (sessionInfo: any) => ipcRenderer.invoke('recording:start', sessionInfo),
  stopRecording: () => ipcRenderer.invoke('recording:stop'),
  addMarker: (marker: string) => ipcRenderer.invoke('recording:addMarker', marker)
})

// For main process messages
contextBridge.exposeInMainWorld('electronAPI', {
  onMainProcessMessage: (callback: (message: string) => void) => {
    ipcRenderer.on('main-process-message', (_event, message) => callback(message));
  }
})