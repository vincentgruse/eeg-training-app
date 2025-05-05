import React from 'react';
import { StatusIndicatorProps } from '../types';

const StatusIndicator: React.FC<StatusIndicatorProps> = ({ label, connected, icon }) => (
  <div className="flex items-center mr-4">
    <div className={`p-1 ${connected ? 'text-green-400' : 'text-red-400'}`}>
      {icon}
    </div>
    <div className="flex items-center">
      <span className="text-sm font-medium mr-1 text-zinc-300">{label}</span>
      <span className={`text-xs px-2 py-0.5 rounded-full ${
        connected 
          ? 'bg-green-900 text-green-300' 
          : 'bg-red-900 text-red-300'
      }`}>
        {connected ? 'Connected' : 'Disconnected'}
      </span>
    </div>
  </div>
);

export default StatusIndicator;