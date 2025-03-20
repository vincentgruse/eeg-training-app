import { ipcMain, app } from 'electron'
import fs from 'fs'
import path from 'path'

// Set up IPC handlers for the application
export const setupIpcHandlers = () => {
  // Handler to save session data to a file
  ipcMain.handle('stimulus:saveSessionData', async (_, data, filename) => {
    try {
      // Save to a fixed data directory within the project
      const projectRoot = process.env.APP_ROOT || app.getAppPath();
      const dataDir = path.join(projectRoot, 'data');
      
      // Create the data directory if it doesn't exist
      if (!fs.existsSync(dataDir)) {
        fs.mkdirSync(dataDir, { recursive: true });
      }
      
      // Full file path
      const filePath = path.join(dataDir, filename);
      
      // Write the data to file
      fs.writeFileSync(filePath, data);
      console.log(`Session data automatically saved to: ${filePath}`);
      
      return { success: true, path: filePath };
    } catch (error: unknown) {
      console.error('Failed to save session data:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      return { success: false, message: errorMessage };
    }
  });
}