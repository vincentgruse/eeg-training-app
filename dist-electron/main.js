"use strict";
const electron = require("electron");
const path = require("path");
const brainflow = require("brainflow");
let boardShim = null;
electron.ipcMain.handle("eeg:connect", async (_event, args) => {
  try {
    const params = {
      serialPort: args.serialPort
    };
    boardShim = new brainflow.BoardShim(brainflow.BoardIds.CYTON_DAISY_BOARD, params);
    await boardShim.prepareSession();
    console.log("Connected to Cyton board on port:", args.serialPort);
    return {
      success: true,
      message: "Connected to OpenBCI Cyton board",
      boardInfo: {
        sampleRate: brainflow.BoardShim.getSamplingRate(brainflow.BoardIds.CYTON_BOARD),
        eegChannels: brainflow.BoardShim.getEegChannels(brainflow.BoardIds.CYTON_BOARD),
        totalChannels: brainflow.BoardShim.getNumRows(brainflow.BoardIds.CYTON_BOARD)
      }
    };
  } catch (error) {
    console.error("Failed to connect to Cyton board:", error);
    const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
    return { success: false, message: errorMessage };
  }
});
electron.ipcMain.handle("eeg:startStream", async () => {
  try {
    if (!boardShim) {
      throw new Error("Board not connected. Please connect to the board first.");
    }
    await boardShim.startStream();
    console.log("Started data streaming from Cyton board");
    return { success: true, message: "Data streaming started" };
  } catch (error) {
    console.error("Failed to start data stream:", error);
    const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
    return { success: false, message: errorMessage };
  }
});
electron.ipcMain.handle("eeg:stopStream", async () => {
  try {
    if (!boardShim) {
      throw new Error("Board not connected.");
    }
    await boardShim.stopStream();
    console.log("Stopped data streaming from Cyton board");
    return { success: true, message: "Data streaming stopped" };
  } catch (error) {
    console.error("Failed to stop data stream:", error);
    const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
    return { success: false, message: errorMessage };
  }
});
electron.ipcMain.handle("eeg:getData", async () => {
  try {
    if (!boardShim) {
      throw new Error("Board not connected.");
    }
    const data = boardShim.getCurrentBoardData(100);
    const eegChannels = brainflow.BoardShim.getEegChannels(brainflow.BoardIds.CYTON_BOARD);
    const timeChannel = brainflow.BoardShim.getTimestampChannel(brainflow.BoardIds.CYTON_BOARD);
    const timestamps = data[timeChannel];
    const channelData = eegChannels.map((channel) => data[channel]);
    const structuredData = {
      timestamps,
      channels: channelData,
      samplingRate: brainflow.BoardShim.getSamplingRate(brainflow.BoardIds.CYTON_BOARD)
    };
    return { success: true, data: structuredData };
  } catch (error) {
    console.error("Failed to get EEG data:", error);
    const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
    return { success: false, message: errorMessage };
  }
});
electron.ipcMain.handle("eeg:disconnect", async () => {
  try {
    if (boardShim) {
      if (boardShim.isPrepared()) {
        try {
          await boardShim.stopStream();
        } catch (e) {
        }
        await boardShim.releaseSession();
        console.log("Disconnected from Cyton board");
      }
      boardShim = null;
    }
    return { success: true, message: "Disconnected from board" };
  } catch (error) {
    console.error("Failed to disconnect from board:", error);
    const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
    return { success: false, message: errorMessage };
  }
});
electron.ipcMain.handle("eeg:addMarker", async (_event, marker) => {
  try {
    if (!boardShim) {
      throw new Error("Board not connected.");
    }
    console.log(`Marker added: ${marker}`);
    return { success: true, message: "Marker added" };
  } catch (error) {
    console.error("Failed to add marker:", error);
    const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
    return { success: false, message: errorMessage };
  }
});
const setupIpcHandlers = () => {
  console.log("EEG IPC handlers initialized with brainflow integration for Cyton board");
};
process.env.DIST = path.join(__dirname, "../dist");
process.env.VITE_PUBLIC = electron.app.isPackaged ? process.env.DIST : path.join(process.env.DIST, "../public");
let win;
const VITE_DEV_SERVER_URL = process.env["VITE_DEV_SERVER_URL"];
function createWindow() {
  win = new electron.BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      nodeIntegration: false,
      contextIsolation: true
    }
  });
  setupIpcHandlers();
  win.webContents.on("did-finish-load", () => {
    win == null ? void 0 : win.webContents.send("main-process-message", (/* @__PURE__ */ new Date()).toLocaleString());
  });
  if (VITE_DEV_SERVER_URL) {
    win.loadURL(VITE_DEV_SERVER_URL);
    win.webContents.openDevTools();
  } else {
    win.loadFile(path.join(process.env.DIST, "index.html"));
  }
}
electron.app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    electron.app.quit();
    win = null;
  }
});
electron.app.whenReady().then(createWindow);
electron.app.on("activate", () => {
  const allWindows = electron.BrowserWindow.getAllWindows();
  if (allWindows.length === 0) {
    createWindow();
  } else {
    allWindows[0].focus();
  }
});
