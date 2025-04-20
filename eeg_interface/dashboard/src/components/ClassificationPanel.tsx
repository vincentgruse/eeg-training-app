import React from 'react';
import { Brain } from 'lucide-react';
import { ClassificationPanelProps } from '../types';

const ClassificationPanel: React.FC<ClassificationPanelProps> = ({ classificationData, lastCommand }) => {
  // Sort directions by confidence (descending)
  const sortedDirections = Object.entries(classificationData)
    .sort((a, b) => b[1] as number - (a[1] as number));
  
  // Highest confidence direction
  const highestConfidenceDirection = sortedDirections.length > 0 ? sortedDirections[0][0] : '';
  
  return (
    <div className="bg-white shadow rounded-lg p-4">
      <h2 className="text-lg font-semibold text-gray-700 mb-4 flex items-center">
        <Brain size={20} className="text-blue-500 mr-2" />
        Command Classification
      </h2>
      <div className="space-y-4">
        <div className="font-medium text-gray-800">
          Predicted Command: 
          <span className="ml-2 font-bold text-blue-600">
            {highestConfidenceDirection}
          </span>
        </div>
        
        {sortedDirections.map(([direction, confidence]) => (
          <div key={direction} className="space-y-1">
            <div className="flex justify-between text-sm">
              <span>{direction}</span>
              <span>{((confidence as number) * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className={`h-2 rounded-full ${direction === lastCommand ? 'bg-blue-600' : 'bg-gray-400'}`}
                style={{ width: `${(confidence as number) * 100}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ClassificationPanel;