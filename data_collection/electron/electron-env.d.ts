/// <reference types="vite-plugin-electron/electron-env" />

declare namespace NodeJS {
  interface ProcessEnv {
    APP_ROOT: string
    VITE_PUBLIC: string
  }
}

// Used in Renderer process, expose in `preload.ts`
interface Window {
  ipcRenderer: import('electron').IpcRenderer
  stimulusApi: {
    saveSessionData: (data: string, filename: string) => Promise<{
      success: boolean;
      path?: string;
      message?: string;
    }>;
  };
  electronAPI: {
    onMainProcessMessage: (callback: (message: string) => void) => void;
  };
  eegProcessorApi: {
    // Browse for EEG data file
    browseForEegFile: () => Promise<{
      success: boolean;
      filePath?: string;
      error?: string;
    }>;
    
    // Process EEG data with a specific file
    processEEGData: (filePath: string) => Promise<{
      success: boolean;
      output?: string;
      error?: string;
    }>;
  };
}