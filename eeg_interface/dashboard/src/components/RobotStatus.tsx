import React from 'react';
import { Navigation, AlertTriangle, Check } from 'lucide-react';
import { RobotStatusProps } from '../types';
import StatusCard from './StatusCard';

const RobotStatus: React.FC<RobotStatusProps> = ({ status }) => {
  return (
    <div className="bg-zinc-800 shadow rounded-lg p-4">
      <h2 className="text-lg font-semibold text-zinc-200 mb-4 flex items-center">
        <Navigation size={20} className="text-zinc-400 mr-2" />
        Robot Status
      </h2>
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <StatusCard 
            label="Last Command" 
            value={status.lastCommand}
            valueColor="text-zinc-300"
          />
          <StatusCard 
            label="Confidence" 
            value={`${(status.commandConfidence * 100).toFixed(1)}%`}
            valueColor={status.commandConfidence > 0.75 ? 'text-green-400' : 'text-yellow-400'}
          />
          <StatusCard 
            label="Distance" 
            value={`${status.distance.toFixed(1)} cm`}
            valueColor={status.obstacleDetected ? 'text-red-400' : 'text-zinc-200'}
          />
          <StatusCard 
            label="Speed" 
            value={status.speed}
            valueColor="text-zinc-200"
          />
        </div>
        
        <div className="flex items-center p-3 rounded-lg bg-zinc-700 border border-zinc-600">
          {status.obstacleDetected ? (
            <>
              <AlertTriangle size={24} className="text-red-400 mr-3" />
              <div>
                <div className="font-medium text-red-400">Obstacle Detected</div>
                <div className="text-sm text-zinc-300">Avoidance maneuver in progress</div>
              </div>
            </>
          ) : (
            <>
              <Check size={24} className="text-green-400 mr-3" />
              <div>
                <div className="font-medium text-green-400">Path Clear</div>
                <div className="text-sm text-zinc-300">No obstacles detected</div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default RobotStatus;