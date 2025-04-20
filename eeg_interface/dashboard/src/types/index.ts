import React from 'react';

export interface TimeValuePair {
    time: number;
    value: number;
  }
  
  export interface ChannelData {
    channel: string;
    values: TimeValuePair[];
  }
  
  export interface ClassificationData {
    [direction: string]: number; // e.g., 'FORWARD': 0.75
  }
  
  export interface SystemStatus {
    boardConnected: boolean;
    esp32Connected: boolean;
    signalQuality: string;
    batteryLevel: number;
    lastCommand: string;
    obstacleDetected: boolean;
    distance: number;
    speed: number;
    commandConfidence: number;
  }
  
  export interface BCIData {
    eegData: ChannelData[];
    classificationData: ClassificationData;
    systemStatus: SystemStatus;
  }
  
  export type SignalQualityType = 'good' | 'fair' | 'poor';
  
  export interface StatusIndicatorProps {
    label: string;
    connected: boolean;
    icon: React.ReactNode;
  }
  
  export interface StatusCardProps {
    label: string;
    value: string | number;
    valueColor: string;
  }
  
  export interface SignalQualityIndicatorProps {
    label: string;
    quality: SignalQualityType;
  }
  
  export interface EEGDisplayProps {
    eegData: ChannelData[];
  }
  
  export interface ClassificationPanelProps {
    classificationData: ClassificationData;
    lastCommand: string;
  }
  
  export interface RobotStatusProps {
    status: SystemStatus;
  }