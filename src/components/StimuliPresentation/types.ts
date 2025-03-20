export type Direction = 'left' | 'right' | 'forward' | 'backward' | 'stop';

export interface Position {
  x: number;
  y: number;
}

export interface IntersectionStimulus {
  id: string;
  direction: Direction;
  completed: boolean;
}

export interface ConfigurationPanelProps {
  stimuliPerDirection: number;
  setStimuliPerDirection: (count: number) => void;
  delayBeforeMovement: number;
  setDelayBeforeMovement: (delay: number) => void;
  participantNumber: string;
  setParticipantNumber: (id: string) => void;
  onStart: () => void;
  error: string | null;
  setError: (error: string | null) => void;
}

export interface InstructionsViewProps {
  // No props currently needed
}

export interface CountdownViewProps {
  value: number;
}

export interface CompletionViewProps {
  onRestart: () => void;
  totalTrials: number;
}

export interface IntersectionViewProps {
  emojiPosition: Position;
  rewardPosition: Position;
  showTrigger: boolean;
  rewardCollected: boolean;
  isStopTrial: boolean;
  hazardPosition?: Position;
}
