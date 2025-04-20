import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { EEGDisplayProps } from '../types';

const EEGDisplay: React.FC<EEGDisplayProps> = ({ eegData }) => {
  if (!eegData || eegData.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-zinc-400">
        Waiting for EEG data...
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {eegData.map((channel) => (
        <div key={channel.channel} className="h-20">
          <div className="text-sm text-zinc-400 mb-1">{channel.channel}</div>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={channel.values}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="time" 
                type="number"
                domain={['dataMin', 'dataMax']}
                tickFormatter={(tick) => ''}
                hide
              />
              <YAxis domain={[-50, 50]} stroke="#9CA3AF" />
              <Tooltip 
                formatter={(value: number) => [Number(value).toFixed(2) + 'μV', 'Signal']}
                labelFormatter={() => ''}
                contentStyle={{ backgroundColor: '#1F2937', borderColor: '#374151', color: '#D1D5DB' }}
              />
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke="#9CA3AF" 
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