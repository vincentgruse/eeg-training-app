import { ipcMain } from 'electron'
import fs from 'fs'
import path from 'path'

// Set up IPC handlers for the application
export const setupIpcHandlers = () => {
  console.log('Setting up IPC handlers for stimulus presentation');
  
  // Handler to get image paths from the assets directory
  ipcMain.handle('stimulus:getImagePaths', async () => {
    try {
      const result: Record<string, string[]> = {};
      
      // Define the categories we want to scan
      const categories = ['left', 'right', 'forward', 'backward', 'stop'];
      
      // Base path to images directory
      const assetsPath = path.join(process.env.APP_ROOT || '', 'src/assets/images');
      console.log('Scanning for images in:', assetsPath);
      
      // Get paths for each category
      for (const category of categories) {
        const categoryPath = path.join(assetsPath, category);
        
        // Check if directory exists
        if (fs.existsSync(categoryPath) && fs.statSync(categoryPath).isDirectory()) {
          // Get all files in the directory
          const files = fs.readdirSync(categoryPath)
            .filter(file => {
              const ext = path.extname(file).toLowerCase();
              return ['.jpg', '.jpeg', '.png', '.gif', '.svg'].includes(ext);
            })
            .map(file => {
              // Use relative path for easier loading in the renderer
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
    } catch (error: unknown) {
      console.error('Failed to get image paths:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      return { success: false, message: errorMessage };
    }
  });
  
  // Add more handlers as needed
}