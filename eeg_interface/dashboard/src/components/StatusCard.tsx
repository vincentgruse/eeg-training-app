import React from 'react';
import { StatusCardProps } from '../types';

const StatusCard: React.FC<StatusCardProps> = ({ label, value, valueColor }) => (
  <div className="bg-zinc-700 p-3 rounded-lg">
    <div className="text-sm text-zinc-400">{label}</div>
    <div className={`font-bold text-lg ${valueColor}`}>{value}</div>
  </div>
);

export default StatusCard;