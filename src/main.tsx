import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import 'bootstrap/dist/css/bootstrap.min.css'
import './assets/index.css'

declare global {
  interface Window {
    electronAPI?: {
      onMainProcessMessage: (callback: (message: string) => void) => void;
    };
  }
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)

if (window.electronAPI) {
  window.electronAPI.onMainProcessMessage((message) => {
    console.log(message);
  });
}