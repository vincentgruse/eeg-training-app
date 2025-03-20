"use strict";
const electron = require("electron");
console.log("Preload script executing...");
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
