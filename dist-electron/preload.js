"use strict";
const electron = require("electron");
electron.contextBridge.exposeInMainWorld("stimulusApi", {
  // Get all image paths from the assets directory
  getImagePaths: () => electron.ipcRenderer.invoke("stimulus:getImagePaths")
});
electron.contextBridge.exposeInMainWorld("electronAPI", {
  onMainProcessMessage: (callback) => {
    electron.ipcRenderer.on("main-process-message", (_event, message) => callback(message));
  }
});
