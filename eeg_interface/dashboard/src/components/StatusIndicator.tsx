import React from 'react';
import { StatusIndicatorProps } from '../types';

const StatusIndicator: React.FC<StatusIndicatorProps> = ({ label, connected, icon }) => (
  <div className="flex items-center mr-4">
    <div className={`p-1 ${connected ? 'text-green-600' : 'text-red-500'}`}>
      {icon}
    </div>
    <div className="flex items-center">
      <span className="text-sm font-medium mr-1">{label}</span>
      <span className={`text-xs px-2 py-0.5 rounded-full ${connected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
        {connected ? 'Connected' : 'Disconnected'}
      </span>
    </div>
  </div>
);

export default StatusIndicator;