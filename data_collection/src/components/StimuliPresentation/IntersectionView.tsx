import React, { useRef, useState, useEffect } from 'react';
import { IntersectionViewProps } from './types';
import { EMOJI_CHAR, REWARD_CHAR, STOP_CHAR, TRIGGER_SIZE } from './constants';

const IntersectionView: React.FC<IntersectionViewProps> = ({
  emojiPosition,
  rewardPosition,
  showTrigger,
  rewardCollected,
  isStopTrial,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 800 });

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const { width, height } = containerRef.current.getBoundingClientRect();
        const minDimension = Math.min(width, height) * 0.9;
        setDimensions({
          width: minDimension,
          height: minDimension,
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);

    return () => {
      window.removeEventListener('resize', updateDimensions);
    };
  }, []);

  const { width, height } = dimensions;
  const roadWidth = width * 0.15;
  const centerX = width / 2;
  const centerY = height / 2;

  const scaledEmojiX = (emojiPosition.x / 600) * width;
  const scaledEmojiY = (emojiPosition.y / 600) * height;
  const scaledRewardX = (rewardPosition.x / 600) * width;
  const scaledRewardY = (rewardPosition.y / 600) * height;

  const roadColor = '#888';
  const lineColor = '#fff';

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#f0f0f0',
        padding: '20px',
        position: 'relative',
      }}
    >
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        style={{ maxWidth: '100%', maxHeight: '100%' }}
      >
        {/* Grass background */}
        <rect x="0" y="0" width={width} height={height} fill="#7ab86a" />

        {/* Horizontal road */}
        <rect
          x="0"
          y={centerY - roadWidth / 2}
          width={width}
          height={roadWidth}
          fill={roadColor}
        />

        {/* Vertical road */}
        <rect
          x={centerX - roadWidth / 2}
          y="0"
          width={roadWidth}
          height={height}
          fill={roadColor}
        />

        {/* Horizontal dividing line */}
        <line
          x1="0"
          y1={centerY}
          x2={width}
          y2={centerY}
          stroke={lineColor}
          strokeWidth="4"
          strokeDasharray="10,10"
        />

        {/* Vertical dividing line */}
        <line
          x1={centerX}
          y1="0"
          x2={centerX}
          y2={height}
          stroke={lineColor}
          strokeWidth="4"
          strokeDasharray="10,10"
        />

        {!rewardCollected && (
          <text
            x={scaledRewardX}
            y={scaledRewardY}
            textAnchor="middle"
            dominantBaseline="middle"
            fontSize={width * 0.06}
            fill={isStopTrial ? 'red' : 'inherit'}
            style={{ userSelect: 'none' }}
          >
            {isStopTrial ? STOP_CHAR : REWARD_CHAR}
          </text>
        )}

        <text
          x={scaledEmojiX}
          y={scaledEmojiY}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize={width * 0.06}
          style={{ userSelect: 'none' }}
        >
          {EMOJI_CHAR}
        </text>
      </svg>

      {showTrigger && (
        <div
          style={{
            position: 'absolute',
            bottom: '0px',
            right: '0px',
            width: `${TRIGGER_SIZE}px`,
            height: `${TRIGGER_SIZE}px`,
            backgroundColor: 'black',
            zIndex: 1000,
          }}
          aria-hidden="true"
        />
      )}
    </div>
  );
};

export default IntersectionView;
