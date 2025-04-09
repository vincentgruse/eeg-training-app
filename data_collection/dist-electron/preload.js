"use strict";
const electron = require("electron");
electron.contextBridge.exposeInMainWorld("stimulusApi", {
  // Save data to file
  saveSessionData: (data, filename) => {
    return electron.ipcRenderer.invoke("stimulus:saveSessionData", data, filename);
  }
});
electron.contextBridge.exposeInMainWorld("electronAPI", {
  onMainProcessMessage: (callback) => {
    electron.ipcRenderer.on("main-process-message", (_event, message) => callback(message));
  }
});
electron.contextBridge.exposeInMainWorld("eegProcessorApi", {
  // Browse for EEG data file
  browseForEegFile: () => {
    return electron.ipcRenderer.invoke("eeg:browseForFile");
  },
  // Process EEG data with a specific file
  processEEGData: (filePath) => {
    return electron.ipcRenderer.invoke("eeg:processData", filePath);
  }
});
