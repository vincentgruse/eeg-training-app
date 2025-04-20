import React, { useState, useEffect } from 'react';
import { Battery, Activity, Brain, Send, Cpu } from 'lucide-react';
import EEGDisplay from './EEGDisplay';
import ClassificationPanel from './ClassificationPanel';
import RobotStatus from './RobotStatus';
import StatusIndicator from './StatusIndicator';
import SignalQualityIndicator from './SignalQuality';
import { BCIData, ChannelData, ClassificationData, SystemStatus, SignalQualityType } from '../types';
import BCIDataService from '../utils/api';

const Dashboard: React.FC = () => {
  const [eegData, setEEGData] = useState<ChannelData[]>([]);
  const [classificationData, setClassificationData] = useState<ClassificationData>({});
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    boardConnected: false,
    esp32Connected: false,
    signalQuality: 'Fair',
    batteryLevel: 0,
    lastCommand: 'NONE',
    obstacleDetected: false,
    distance: 0,
    speed: 0,
    commandConfidence: 0,
  });
  
  // WebSocket connection
  useEffect(() => {
    // Connect to the BCI data service
    BCIDataService.connect();
    
    // Set up event handlers
    BCIDataService.onData((data: BCIData) => {
      setEEGData(data.eegData);
      setClassificationData(data.classificationData);
      setSystemStatus(data.systemStatus);
    });
    
    BCIDataService.onConnect(() => {
      console.log('Connected to BCI system');
    });
    
    BCIDataService.onDisconnect(() => {
      console.log('Disconnected from BCI system');
    });
    
    // Clean up on unmount
    return () => {
      BCIDataService.disconnect();
    };
  }, []);
  
  // Handle manual command sending
  const sendCommand = (direction: string) => {
    BCIDataService.sendCommand('send_command', { direction });
  };
  
  return (
    <div className="flex justify-center bg-zinc-900 min-h-screen w-screen">
      <div className="w-full max-w-6xl px-4">
        <header className="bg-zinc-800 shadow rounded-lg p-4 my-2 flex justify-between items-center">
          <div className="flex items-center">
            <Brain size={28} className="text-zinc-400 mr-3" />
            <h1 className="text-2xl font-bold text-zinc-100 me-4">BCI System Dashboard</h1>
          </div>
          <div className="flex items-center">
            <StatusIndicator 
              label="OpenBCI" 
              connected={systemStatus.boardConnected} 
              icon={<Cpu size={18} />} 
            />
            <StatusIndicator 
              label="ESP32" 
              connected={systemStatus.esp32Connected} 
              icon={<Send size={18} />} 
            />
            <div className="flex items-center text-zinc-300 ml-4">
              <Battery size={18} className="mr-1" />
              <span>{systemStatus.batteryLevel}%</span>
            </div>
          </div>
        </header>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <div className="bg-zinc-800 shadow rounded-lg p-4 mb-2">
              <h2 className="text-lg font-semibold text-zinc-200 mb-4 flex items-center">
                <Activity size={20} className="text-zinc-400 mr-2" />
                EEG Signal Monitor
              </h2>
              <EEGDisplay eegData={eegData} />
            </div>
        
            <div className="bg-zinc-800 shadow rounded-lg p-4 mb-2">
              <h2 className="text-lg font-semibold text-zinc-200 mb-4">Manual Control</h2>
              <div className="grid grid-cols-3 gap-2">
                <div></div>
                <button 
                  className="bg-zinc-600 hover:bg-zinc-700 text-white py-2 px-4 rounded"
                  onClick={() => sendCommand('FORWARD')}
                >
                  Forward
                </button>
                <div></div>
                
                <button 
                  className="bg-zinc-600 hover:bg-zinc-700 text-white py-2 px-4 rounded"
                  onClick={() => sendCommand('LEFT')}
                >
                  Left
                </button>
                <button 
                  className="bg-red-600 hover:bg-red-700 text-white py-2 px-4 rounded"
                  onClick={() => sendCommand('STOP')}
                >
                  Stop
                </button>
                <button 
                  className="bg-zinc-600 hover:bg-zinc-700 text-white py-2 px-4 rounded"
                  onClick={() => sendCommand('RIGHT')}
                >
                  Right
                </button>
                
                <div></div>
                <button 
                  className="bg-zinc-600 hover:bg-zinc-700 text-white py-2 px-4 rounded"
                  onClick={() => sendCommand('BACKWARD')}
                >
                  Backward
                </button>
                <div></div>
              </div>
            </div>
          </div>
          
          <div className="space-y-2">
            <ClassificationPanel 
              classificationData={classificationData} 
              lastCommand={systemStatus.lastCommand} 
            />
            
            <RobotStatus status={systemStatus} />
            
            <div className="bg-zinc-800 shadow rounded-lg p-4">
              <h2 className="text-lg font-semibold text-zinc-200 mb-2 flex items-center">
                <Activity size={20} className="text-zinc-400 mr-2" />
                Signal Quality
              </h2>
              <div className="space-y-3">
                {eegData.map(channel => (
                  <SignalQualityIndicator 
                    key={channel.channel}
                    label={channel.channel} 
                    quality={systemStatus.signalQuality as SignalQualityType} 
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;