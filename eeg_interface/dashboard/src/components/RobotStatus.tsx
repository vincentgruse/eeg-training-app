import React from 'react';
import { Navigation, AlertTriangle, Check } from 'lucide-react';
import { RobotStatusProps } from '../types';
import StatusCard from './StatusCard';

const RobotStatus: React.FC<RobotStatusProps> = ({ status }) => {
  return (
    <div className="bg-white shadow rounded-lg p-4">
      <h2 className="text-lg font-semibold text-gray-700 mb-4 flex items-center">
        <Navigation size={20} className="text-blue-500 mr-2" />
        Robot Status
      </h2>
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <StatusCard 
            label="Last Command" 
            value={status.lastCommand}
            valueColor="text-blue-600"
          />
          <StatusCard 
            label="Confidence" 
            value={`${(status.commandConfidence * 100).toFixed(1)}%`}
            valueColor={status.commandConfidence > 0.75 ? 'text-green-600' : 'text-yellow-600'}
          />
          <StatusCard 
            label="Distance" 
            value={`${status.distance.toFixed(1)} cm`}
            valueColor={status.obstacleDetected ? 'text-red-600' : 'text-gray-800'}
          />
          <StatusCard 
            label="Speed" 
            value={status.speed}
            valueColor="text-gray-800"
          />
        </div>
        
        <div className="flex items-center p-3 rounded-lg bg-gray-50 border border-gray-200">
          {status.obstacleDetected ? (
            <>
              <AlertTriangle size={24} className="text-red-500 mr-3" />
              <div>
                <div className="font-medium text-red-600">Obstacle Detected</div>
                <div className="text-sm text-gray-600">Avoidance maneuver in progress</div>
              </div>
            </>
          ) : (
            <>
              <Check size={24} className="text-green-500 mr-3" />
              <div>
                <div className="font-medium text-green-600">Path Clear</div>
                <div className="text-sm text-gray-600">No obstacles detected</div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default RobotStatus;