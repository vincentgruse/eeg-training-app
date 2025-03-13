"use strict";
const electron = require("electron");
const path = require("path");
const fs = require("fs");
const setupIpcHandlers = () => {
  console.log("Setting up IPC handlers for stimulus presentation");
  electron.ipcMain.handle("stimulus:getImagePaths", async () => {
    try {
      const result = {};
      const categories = ["left", "right", "forward", "backward", "stop"];
      const assetsPath = path.join(process.env.APP_ROOT || "", "src/assets/images");
      console.log("Scanning for images in:", assetsPath);
      for (const category of categories) {
        const categoryPath = path.join(assetsPath, category);
        if (fs.existsSync(categoryPath) && fs.statSync(categoryPath).isDirectory()) {
          const files = fs.readdirSync(categoryPath).filter((file) => {
            const ext = path.extname(file).toLowerCase();
            return [".jpg", ".jpeg", ".png", ".gif", ".svg"].includes(ext);
          }).map((file) => {
            return `src/assets/images/${category}/${file}`;
          });
          result[category] = files;
          console.log(`Found ${files.length} images for ${category}`);
        } else {
          result[category] = [];
          console.log(`No directory found for ${category}`);
        }
      }
      return { success: true, paths: result };
    } catch (error) {
      console.error("Failed to get image paths:", error);
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
