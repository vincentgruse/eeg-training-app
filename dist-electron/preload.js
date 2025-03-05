"use strict";
const electron = require("electron");
electron.contextBridge.exposeInMainWorld("eegApi", {
  connect: (params) => electron.ipcRenderer.invoke("eeg:connect", params),
  startStream: () => electron.ipcRenderer.invoke("eeg:startStream"),
  stopStream: () => electron.ipcRenderer.invoke("eeg:stopStream"),
  getData: () => electron.ipcRenderer.invoke("eeg:getData"),
  disconnect: () => electron.ipcRenderer.invoke("eeg:disconnect"),
  addMarker: (marker) => electron.ipcRenderer.invoke("eeg:addMarker", marker)
});
electron.contextBridge.exposeInMainWorld("stimulusApi", {
  present: (stimulus) => electron.ipcRenderer.invoke("stimulus:present", stimulus)
});
electron.contextBridge.exposeInMainWorld("recordingApi", {
  startRecording: (sessionInfo) => electron.ipcRenderer.invoke("recording:start", sessionInfo),
  stopRecording: () => electron.ipcRenderer.invoke("recording:stop"),
  addMarker: (marker) => electron.ipcRenderer.invoke("recording:addMarker", marker)
});
electron.contextBridge.exposeInMainWorld("electronAPI", {
  onMainProcessMessage: (callback) => {
    electron.ipcRenderer.on("main-process-message", (_event, message) => callback(message));
  }
});
