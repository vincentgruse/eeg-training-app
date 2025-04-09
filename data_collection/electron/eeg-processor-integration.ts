import { ipcMain, app, dialog } from 'electron';
import path from 'path';
import { spawn } from 'child_process';
import fs from 'fs';

export const setupEEGProcessorHandlers = () => {
  // Handler for browsing for an EEG file
  ipcMain.handle('eeg:browseForFile', async () => {
    try {
      const result = await dialog.showOpenDialog({
        properties: ['openFile'],
        title: 'Select OpenBCI EEG Data File',
        filters: [
          { name: 'OpenBCI Data', extensions: ['txt', 'csv'] },
          { name: 'All Files', extensions: ['*'] }
        ]
      });

      if (result.canceled || result.filePaths.length === 0) {
        return { success: false };
      }

      return { 
        success: true, 
        filePath: result.filePaths[0] 
      };
    } catch (error: unknown) {
      console.error('Error browsing for file:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      return { success: false, error: errorMessage };
    }
  });

  // Handler for processing EEG data with a specific file
  ipcMain.handle('eeg:processData', async (_, filePath) => {
    try {
      // Check if file exists
      if (!fs.existsSync(filePath)) {
        console.error(`EEG data file not found: ${filePath}`);
        return { 
          success: false, 
          error: `EEG data file not found: ${filePath}` 
        };
      }

      // Path to the Python script
      const scriptPath = path.resolve(__dirname, '../resources/eeg_processor.py');
      
      // Ensure the script exists
      if (!fs.existsSync(scriptPath)) {
        console.error(`EEG processor script not found at: ${scriptPath}`);
        return { 
          success: false, 
          error: `EEG processor script not found at: ${scriptPath}` 
        };
      }
      
      // Get the path to the most recent participant info file
      const dataDir = path.join(process.env.APP_ROOT || app.getAppPath(), 'data');
      let participantFile = findMostRecentParticipantFile(dataDir);
      
      if (!participantFile) {
        console.error(`No participant info files found in ${dataDir}`);
        return { 
          success: false, 
          error: `No participant info files found in ${dataDir}` 
        };
      }
      
      console.log('Launching EEG processor script...');
      console.log(`Script path: ${scriptPath}`);
      console.log(`EEG data file: ${filePath}`);
      console.log(`Participant info file: ${participantFile}`);
      
      // Run the Python script with the provided file paths
      const pythonProcess = spawn('python', [
        scriptPath,
        '--eeg', filePath,
        '--participant', participantFile
      ]);
      
      // Collect output from the process
      let output = '';
      let errorOutput = '';
      
      pythonProcess.stdout.on('data', (data) => {
        const text = data.toString();
        output += text;
        console.log(`EEG Processor: ${text.trim()}`);
      });
      
      pythonProcess.stderr.on('data', (data) => {
        const text = data.toString();
        errorOutput += text;
        console.error(`EEG Processor Error: ${text.trim()}`);
      });
      
      // Return a promise that resolves when the process exits
      return new Promise((resolve) => {
        pythonProcess.on('close', (code) => {
          if (code === 0) {
            console.log('EEG processing completed successfully');
            resolve({ success: true, output });
          } else {
            console.error(`EEG processing failed with code ${code}`);
            resolve({ success: false, error: errorOutput || `Process exited with code ${code}` });
          }
        });
      });
    } catch (error: unknown) {
      console.error('Failed to launch EEG processor:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      return { success: false, error: errorMessage };
    }
  });
};

function findMostRecentParticipantFile(dataDir: string): string | null {
  if (!fs.existsSync(dataDir)) {
    console.error(`Data directory ${dataDir} does not exist`);
    return null;
  }
  
  const participantFiles: [string, number][] = [];
  
  try {
    for (const file of fs.readdirSync(dataDir)) {
      if (file.startsWith('participant_') && file.endsWith('.txt')) {
        const fullPath = path.join(dataDir, file);
        participantFiles.push([fullPath, fs.statSync(fullPath).mtimeMs]);
      }
    }
  } catch (error) {
    console.error('Error reading data directory:', error);
    return null;
  }
  
  if (participantFiles.length === 0) {
    return null;
  }
  
  // Sort by modification time (most recent first)
  participantFiles.sort((a, b) => b[1] - a[1]);
  
  return participantFiles[0][0];
}