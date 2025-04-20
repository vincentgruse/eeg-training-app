import React, { useState, useEffect } from 'react';
import { Battery, Activity, Brain, Send, Cpu } from 'lucide-react';
import EEGDisplay from './EEGDisplay';
import ClassificationPanel from './ClassificationPanel';
import RobotStatus from './RobotStatus';
import StatusIndicator from './StatusIndicator';
import SignalQualityIndicator from './SignalQuality';
import { BCIData, ChannelData, ClassificationData, SystemStatus, SignalQualityType } from '../types';
import BCIDataService from '../utils/api';

// Placeholder data for initial render
const generateInitialEEGData = (): ChannelData[] => {
  const now = Date.now();
  const channelNames = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2'];
  
  return channelNames.map(channel => ({
    channel,
    values: Array(50).fill(0).map((_, i) => ({ 
      time: now - (49-i) * 20, 
      value: Math.sin(i/5 + channelNames.indexOf(channel)) * 20 + Math.random() * 8 - 4 
    }))
  }));
};

const generateInitialClassificationData = (): ClassificationData => {
  const directions = ['FORWARD', 'BACKWARD', 'LEFT', 'RIGHT', 'STOP'];
  const result: ClassificationData = {};
  
  directions.forEach(dir => {
    result[dir] = 0.2; // Equal probability initially
  });
  
  return result;
};

const generateInitialSystemStatus = (): SystemStatus => ({
  boardConnected: false,
  esp32Connected: false,
  signalQuality: 'Fair',
  batteryLevel: 100,
  lastCommand: 'NONE',
  obstacleDetected: false,
  distance: 45,
  speed: 150,
  commandConfidence: 0.2,
});

const Dashboard: React.FC = () => {
  const [eegData, setEEGData] = useState<ChannelData[]>(generateInitialEEGData());
  const [classificationData, setClassificationData] = useState<ClassificationData>(generateInitialClassificationData());
  const [systemStatus, setSystemStatus] = useState<SystemStatus>(generateInitialSystemStatus());
  
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
    <div className="bg-gray-100 min-h-screen p-4">
      <div className="w-full mx-auto">
        <header className="bg-white shadow rounded-lg p-4 mb-6 flex justify-between items-center">
          <div className="flex items-center">
            <Brain size={28} className="text-blue-600 mr-2" />
            <h1 className="text-2xl font-bold text-gray-800">BCI System Dashboard</h1>
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
            <div className="flex items-center text-gray-700">
              <Battery size={18} className="mr-1" />
              <span>{systemStatus.batteryLevel}%</span>
            </div>
          </div>
        </header>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <div className="bg-white shadow rounded-lg p-4 mb-6">
            <h2 className="text-lg font-semibold text-gray-700 mb-4 flex items-center">
                <Activity size={20} className="text-blue-500 mr-2" />
                EEG Signal Monitor
              </h2>
              <EEGDisplay eegData={eegData} />
            </div>
        
            <div className="bg-white shadow rounded-lg p-4 mb-6">
              <h2 className="text-lg font-semibold text-gray-700 mb-4">Manual Control</h2>
              <div className="grid grid-cols-3 gap-2">
                <div></div>
                <button 
                  className="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded"
                  onClick={() => sendCommand('FORWARD')}
                >
                  Forward
                </button>
                <div></div>
                
                <button 
                  className="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded"
                  onClick={() => sendCommand('LEFT')}
                >
                  Left
                </button>
                <button 
                  className="bg-red-500 hover:bg-red-600 text-white py-2 px-4 rounded"
                  onClick={() => sendCommand('STOP')}
                >
                  Stop
                </button>
                <button 
                  className="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded"
                  onClick={() => sendCommand('RIGHT')}
                >
                  Right
                </button>
                
                <div></div>
                <button 
                  className="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded"
                  onClick={() => sendCommand('BACKWARD')}
                >
                  Backward
                </button>
                <div></div>
              </div>
            </div>
          </div>
          
          <div className="space-y-6">
            <ClassificationPanel 
              classificationData={classificationData} 
              lastCommand={systemStatus.lastCommand} 
            />
            
            <RobotStatus status={systemStatus} />
            
            <div className="bg-white shadow rounded-lg p-4">
              <h2 className="text-lg font-semibold text-gray-700 mb-2 flex items-center">
                <Activity size={20} className="text-blue-500 mr-2" />
                Signal Quality
              </h2>
              <div className="space-y-3">
                <SignalQualityIndicator 
                  label="Fp1" 
                  quality={Math.random() > 0.3 ? 'good' : (Math.random() > 0.5 ? 'fair' : 'poor') as SignalQualityType} 
                />
                <SignalQualityIndicator 
                  label="Fp2" 
                  quality={Math.random() > 0.3 ? 'good' : (Math.random() > 0.5 ? 'fair' : 'poor') as SignalQualityType} 
                />
                <SignalQualityIndicator 
                  label="C3" 
                  quality={Math.random() > 0.2 ? 'good' : (Math.random() > 0.6 ? 'fair' : 'poor') as SignalQualityType} 
                />
                <SignalQualityIndicator 
                  label="C4" 
                  quality={Math.random() > 0.2 ? 'good' : (Math.random() > 0.6 ? 'fair' : 'poor') as SignalQualityType} 
                />
                <SignalQualityIndicator 
                  label="P7" 
                  quality={Math.random() > 0.3 ? 'good' : (Math.random() > 0.5 ? 'fair' : 'poor') as SignalQualityType} 
                />
                <SignalQualityIndicator 
                  label="P8" 
                  quality={Math.random() > 0.3 ? 'good' : (Math.random() > 0.5 ? 'fair' : 'poor') as SignalQualityType} 
                />
                <SignalQualityIndicator 
                  label="O1" 
                  quality={Math.random() > 0.3 ? 'good' : (Math.random() > 0.5 ? 'fair' : 'poor') as SignalQualityType} 
                />
                <SignalQualityIndicator 
                  label="O2" 
                  quality={Math.random() > 0.3 ? 'good' : (Math.random() > 0.5 ? 'fair' : 'poor') as SignalQualityType} 
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;