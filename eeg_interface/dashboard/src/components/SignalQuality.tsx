import React from 'react';
import { SignalQualityIndicatorProps } from '../types';

const SignalQualityIndicator: React.FC<SignalQualityIndicatorProps> = ({ label, quality }) => {
  const getQualityColor = (quality: string) => {
    switch (quality) {
      case 'good': return 'bg-green-500';
      case 'fair': return 'bg-yellow-500';
      case 'poor': return 'bg-red-500';
      default: return 'bg-zinc-500';
    }
  };
  
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center">
        <div className={`w-3 h-3 rounded-full ${getQualityColor(quality)} mr-2`}></div>
        <span className="text-zinc-300">{label}</span>
      </div>
      <span className="text-sm capitalize text-zinc-400">{quality}</span>
    </div>
  );
};

export default SignalQualityIndicator;