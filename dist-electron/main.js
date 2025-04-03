"use strict";
const electron = require("electron");
const path = require("path");
const fs = require("fs");
const child_process = require("child_process");
const setupIpcHandlers = () => {
  electron.ipcMain.handle("stimulus:saveSessionData", async (_, data, filename) => {
    try {
      const projectRoot = process.env.APP_ROOT || electron.app.getAppPath();
      const dataDir = path.join(projectRoot, "data");
      if (!fs.existsSync(dataDir)) {
        fs.mkdirSync(dataDir, { recursive: true });
      }
      const filePath = path.join(dataDir, filename);
      fs.writeFileSync(filePath, data);
      console.log(`Session data automatically saved to: ${filePath}`);
      return { success: true, path: filePath };
    } catch (error) {
      console.error("Failed to save session data:", error);
      const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
      return { success: false, message: errorMessage };
    }
  });
};
const setupEEGProcessorHandlers = () => {
  electron.ipcMain.handle("eeg:browseForFile", async () => {
    try {
      const result = await electron.dialog.showOpenDialog({
        properties: ["openFile"],
        title: "Select OpenBCI EEG Data File",
        filters: [
          { name: "OpenBCI Data", extensions: ["txt", "csv"] },
          { name: "All Files", extensions: ["*"] }
        ]
      });
      if (result.canceled || result.filePaths.length === 0) {
        return { success: false };
      }
      return {
        success: true,
        filePath: result.filePaths[0]
      };
    } catch (error) {
      console.error("Error browsing for file:", error);
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      return { success: false, error: errorMessage };
    }
  });
  electron.ipcMain.handle("eeg:processData", async (_, filePath) => {
    try {
      if (!fs.existsSync(filePath)) {
        console.error(`EEG data file not found: ${filePath}`);
        return {
          success: false,
          error: `EEG data file not found: ${filePath}`
        };
      }
      const scriptPath = path.resolve(__dirname, "../resources/eeg_processor.py");
      if (!fs.existsSync(scriptPath)) {
        console.error(`EEG processor script not found at: ${scriptPath}`);
        return {
          success: false,
          error: `EEG processor script not found at: ${scriptPath}`
        };
      }
      const dataDir = path.join(process.env.APP_ROOT || electron.app.getAppPath(), "data");
      let participantFile = findMostRecentParticipantFile(dataDir);
      if (!participantFile) {
        console.error(`No participant info files found in ${dataDir}`);
        return {
          success: false,
          error: `No participant info files found in ${dataDir}`
        };
      }
      console.log("Launching EEG processor script...");
      console.log(`Script path: ${scriptPath}`);
      console.log(`EEG data file: ${filePath}`);
      console.log(`Participant info file: ${participantFile}`);
      const pythonProcess = child_process.spawn("python", [
        scriptPath,
        "--eeg",
        filePath,
        "--participant",
        participantFile
      ]);
      let output = "";
      let errorOutput = "";
      pythonProcess.stdout.on("data", (data) => {
        const text = data.toString();
        output += text;
        console.log(`EEG Processor: ${text.trim()}`);
      });
      pythonProcess.stderr.on("data", (data) => {
        const text = data.toString();
        errorOutput += text;
        console.error(`EEG Processor Error: ${text.trim()}`);
      });
      return new Promise((resolve) => {
        pythonProcess.on("close", (code) => {
          if (code === 0) {
            console.log("EEG processing completed successfully");
            resolve({ success: true, output });
          } else {
            console.error(`EEG processing failed with code ${code}`);
            resolve({ success: false, error: errorOutput || `Process exited with code ${code}` });
          }
        });
      });
    } catch (error) {
      console.error("Failed to launch EEG processor:", error);
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      return { success: false, error: errorMessage };
    }
  });
};
function findMostRecentParticipantFile(dataDir) {
  if (!fs.existsSync(dataDir)) {
    console.error(`Data directory ${dataDir} does not exist`);
    return null;
  }
  const participantFiles = [];
  try {
    for (const file of fs.readdirSync(dataDir)) {
      if (file.startsWith("participant_") && file.endsWith(".txt")) {
        const fullPath = path.join(dataDir, file);
        participantFiles.push([fullPath, fs.statSync(fullPath).mtimeMs]);
      }
    }
  } catch (error) {
    console.error("Error reading data directory:", error);
    return null;
  }
  if (participantFiles.length === 0) {
    return null;
  }
  participantFiles.sort((a, b) => b[1] - a[1]);
  return participantFiles[0][0];
}
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
  setupEEGProcessorHandlers();
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
