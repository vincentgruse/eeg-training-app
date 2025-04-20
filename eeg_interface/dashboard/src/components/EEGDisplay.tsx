import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { EEGDisplayProps } from '../types';

const EEGDisplay: React.FC<EEGDisplayProps> = ({ eegData }) => {
  return (
    <div className="space-y-4">
      {eegData.map((channel) => (
        <div key={channel.channel} className="h-20">
          <div className="text-sm text-gray-600 mb-1">{channel.channel}</div>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={channel.values}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="time" 
                type="number"
                domain={['dataMin', 'dataMax']}
                tickFormatter={(tick) => ''}
                hide
              />
              <YAxis domain={[-50, 50]} />
              <Tooltip 
                formatter={(value: number) => [Number(value).toFixed(2) + 'μV', 'Signal']}
                labelFormatter={() => ''}
              />
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke="#3b82f6" 
                dot={false} 
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ))}
    </div>
  );
};

export default EEGDisplay;