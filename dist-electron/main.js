"use strict";
const electron = require("electron");
const path = require("path");
const fs = require("fs");
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
