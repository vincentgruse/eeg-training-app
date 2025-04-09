import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Card, Button } from 'react-bootstrap';
import ConfigurationPanel from './ConfigurationPanel';
import InstructionsView from './InstructionsView';
import CountdownView from './CountdownView';
import CompletionView from './CompletionView';
import IntersectionView from './IntersectionView';
import { IntersectionStimulus, Position, Direction } from './types';
import {
  DIRECTIONS,
  DIRECTION_DISPLAY_DURATION,
  EMOJI_MOVEMENT_DURATION,
  POST_REWARD_PAUSE,
  SIMULATION_CENTER_X,
  SIMULATION_CENTER_Y,
  SIMULATION_REWARD_DISTANCE,
} from './constants';

// Updated trigger duration to 1000ms (1 second)
const TRIGGER_DURATION = 1000;

interface TrialData {
  trialIndex: number;
  direction: Direction;
  isStopTrial: boolean;
  startTime: Date;
  endTime?: Date;
}

const StimulusDisplay: React.FC = () => {
  // Configuration state
  const [stimuliPerDirection, setStimuliPerDirection] = useState(5);
  const [delayBeforeMovement, setDelayBeforeMovement] = useState(2000);
  const [participantNumber, setParticipantNumber] = useState('');
  const [error, setError] = useState<string | null>(null);

  // Session tracking state
  const [sessionStartTime, setSessionStartTime] = useState<Date | null>(null);
  const [sessionEndTime, setSessionEndTime] = useState<Date | null>(null);
  const [trialData, setTrialData] = useState<TrialData[]>([]);
  const [hasExportedData, setHasExportedData] = useState(false);

  // Presentation flow state
  const [isPresenting, setIsPresenting] = useState(false);
  const [showDirections, setShowDirections] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [currentStimulusIndex, setCurrentStimulusIndex] = useState(0);
  const [showTrigger, setShowTrigger] = useState(false);
  const [sessionComplete, setSessionComplete] = useState(false);

  // Visual state for intersection simulation
  const [emojiPosition, setEmojiPosition] = useState<Position>({
    x: SIMULATION_CENTER_X,
    y: SIMULATION_CENTER_Y,
  });
  const [rewardPosition, setRewardPosition] = useState<Position>({
    x: SIMULATION_CENTER_X,
    y: SIMULATION_CENTER_Y - SIMULATION_REWARD_DISTANCE,
  });
  const [isStopTrial, setIsStopTrial] = useState(false);
  const [showStopSign, setShowStopSign] = useState(false);
  const [rewardCollected, setRewardCollected] = useState(false);

  // Refs for maintaining state across renders
  const stimuliRef = useRef<IntersectionStimulus[]>([]);
  const animationRef = useRef<number | null>(null);
  const currentTrialRef = useRef<TrialData | null>(null);

  // Generate randomized stimuli based on configuration
  const generateStimuli = useCallback(() => {
    let generatedStimuli: IntersectionStimulus[] = [];

    DIRECTIONS.forEach((direction) => {
      for (let i = 0; i < stimuliPerDirection; i++) {
        generatedStimuli.push({
          id: `${direction}-${i}`,
          direction: direction as Direction,
          completed: false,
        });
      }
    });

    // Randomize order of presentation
    return generatedStimuli.sort(() => Math.random() - 0.5);
  }, [stimuliPerDirection]);

  // Calculate reward position based on direction
  const calculateRewardPosition = (direction: string): Position => {
    switch (direction) {
      case 'left':
        return {
          x: SIMULATION_CENTER_X - SIMULATION_REWARD_DISTANCE,
          y: SIMULATION_CENTER_Y,
        };
      case 'right':
        return {
          x: SIMULATION_CENTER_X + SIMULATION_REWARD_DISTANCE,
          y: SIMULATION_CENTER_Y,
        };
      case 'forward':
        return {
          x: SIMULATION_CENTER_X,
          y: SIMULATION_CENTER_Y - SIMULATION_REWARD_DISTANCE,
        };
      case 'backward':
        return {
          x: SIMULATION_CENTER_X,
          y: SIMULATION_CENTER_Y + SIMULATION_REWARD_DISTANCE,
        };
      default:
        return {
          x: SIMULATION_CENTER_X,
          y: SIMULATION_CENTER_Y,
        };
    }
  };

  // Calculate position to stop
  const calculateStopPosition = useCallback(
    (startPos: Position, endPos: Position): Position => {
      return {
        x: startPos.x + (endPos.x - startPos.x) * 0.8,
        y: startPos.y + (endPos.y - startPos.y) * 0.8,
      };
    },
    []
  );

  // Export session data to a file
  const exportSessionData = useCallback(async () => {
    if (
      !sessionStartTime ||
      !sessionEndTime ||
      stimuliRef.current.length === 0
    ) {
      console.error('No session data available to export');
      return false;
    }

    // Generate content and filename for export
    const content = createExportContent(
      participantNumber,
      sessionStartTime,
      sessionEndTime
    );

    const fileName = createFileName(participantNumber);

    try {
      return await saveDataToFile(content, fileName);
    } catch (err) {
      const error = err as Error;
      console.error('Error exporting session data:', error);
      alert(`Error exporting data: ${error.message || 'Unknown error'}`);
      return false;
    }
  }, [sessionStartTime, sessionEndTime, stimuliRef, participantNumber]);

  // Handle early termination of presentation
  const handleStopPresentation = useCallback(() => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }

    // Show trigger for emergency stop
    setShowTrigger(true);
    setTimeout(() => {
      setShowTrigger(false);

      if (isPresenting && !sessionEndTime) {
        setSessionEndTime(new Date());
      }

      if (currentTrialRef.current && !currentTrialRef.current.endTime) {
        currentTrialRef.current.endTime = new Date();
        setTrialData((prev) => [...prev]);
      }

      setIsPresenting(false);
      setShowDirections(false);
      setCountdown(null);
      setSessionComplete(false);
      setCurrentStimulusIndex(0);
      setEmojiPosition({
        x: SIMULATION_CENTER_X,
        y: SIMULATION_CENTER_Y,
      });
      setShowTrigger(false);
      setRewardCollected(false);
      setIsStopTrial(false);
      setShowStopSign(false);
    }, TRIGGER_DURATION);
  }, [isPresenting, sessionEndTime]);

  // Animate the emoji movement from start to end position
  const animateEmojiMovement = useCallback(
    (
      startPos: Position,
      endPos: Position,
      duration: number,
      markRewardCollected: boolean,
      onComplete: () => void
    ) => {
      const startTime = performance.now();

      const animate = (currentTime: number) => {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Linear interpolation between start and end positions
        const newX = startPos.x + (endPos.x - startPos.x) * progress;
        const newY = startPos.y + (endPos.y - startPos.y) * progress;

        setEmojiPosition({ x: newX, y: newY });

        if (progress < 1) {
          // Continue animation if not complete
          animationRef.current = requestAnimationFrame(animate);
        } else {
          // Animation complete
          if (markRewardCollected) {
            setRewardCollected(true);
          }

          if (currentTrialRef.current) {
            currentTrialRef.current.endTime = new Date();
            setTrialData((prev) => [...prev]);
          }

          onComplete();
        }
      };

      animationRef.current = requestAnimationFrame(animate);
    },
    []
  );

  // Present the next stimulus in the sequence
  const presentNextStimulus = useCallback(() => {
    // Avoid duplicate processing of the same stimulus
    const alreadyProcessed = trialData.some(
      (trial) => trial.trialIndex === currentStimulusIndex
    );
    if (alreadyProcessed) {
      return;
    }

    // Check if we've presented all stimuli
    if (currentStimulusIndex >= stimuliRef.current.length) {
      console.log('All stimuli presented, ending session');
      setSessionEndTime(new Date());
      setIsPresenting(false);
      setSessionComplete(true);
      return;
    }

    // Get the current stimulus and determine trial type
    const currentStimulus = stimuliRef.current[currentStimulusIndex];
    const currentIsStopTrial = currentStimulus.direction === 'stop';

    // Create trial data record
    const newTrial: TrialData = {
      trialIndex: currentStimulusIndex,
      direction: currentStimulus.direction,
      isStopTrial: currentIsStopTrial,
      startTime: new Date(),
    };

    currentTrialRef.current = newTrial;
    setTrialData((prev) => [...prev, newTrial]);

    // Reset for the new stimulus
    setIsStopTrial(currentIsStopTrial);
    setShowStopSign(false);
    setEmojiPosition({
      x: SIMULATION_CENTER_X,
      y: SIMULATION_CENTER_Y,
    });
    setRewardCollected(false);

    // For stop trials, pick a random direction for visual display
    let effectiveDirection = currentStimulus.direction;
    if (currentIsStopTrial) {
      const directions = DIRECTIONS.filter((d) => d !== 'stop');
      effectiveDirection = directions[
        Math.floor(Math.random() * directions.length)
      ] as Direction;
    }

    // Position the reward based on direction
    const newRewardPosition = calculateRewardPosition(effectiveDirection);
    setRewardPosition(newRewardPosition);

    // Show initial trigger for 1 second
    setShowTrigger(true);
    setTimeout(() => {
      setShowTrigger(false);
    }, TRIGGER_DURATION);

    // After delay, start movement
    setTimeout(
      () => {
        if (currentIsStopTrial) {
          // For stop trials, calculate stopping point and show stop sign
          const startPos = { x: SIMULATION_CENTER_X, y: SIMULATION_CENTER_Y };
          const stopPosition = calculateStopPosition(
            startPos,
            newRewardPosition
          );

          setShowTrigger(true);
          setShowStopSign(true);
          setTimeout(() => {
            setShowTrigger(false);
          }, TRIGGER_DURATION);

          // Animate to stopping point
          animateEmojiMovement(
            startPos,
            stopPosition,
            EMOJI_MOVEMENT_DURATION + delayBeforeMovement,
            false,
            () => {
              setTimeout(() => {
                setCurrentStimulusIndex((prev) => prev + 1);
              }, POST_REWARD_PAUSE);
            }
          );
        } else {
          // For normal trials, move to reward
          animateEmojiMovement(
            { x: SIMULATION_CENTER_X, y: SIMULATION_CENTER_Y },
            newRewardPosition,
            EMOJI_MOVEMENT_DURATION,
            true,
            () => {
              setTimeout(() => {
                setCurrentStimulusIndex((prev) => prev + 1);
              }, POST_REWARD_PAUSE);
            }
          );
        }
      },
      currentIsStopTrial ? 0 : delayBeforeMovement
    );
  }, [
    currentStimulusIndex,
    delayBeforeMovement,
    animateEmojiMovement,
    calculateStopPosition,
    trialData,
  ]);

  // Handle return to configuration screen
  const handleReturnToConfig = useCallback(async () => {
    // Show single trigger for session end
    setShowTrigger(true);

    // Wait for the trigger to complete before proceeding
    setTimeout(async () => {
      setShowTrigger(false);

      if (!sessionEndTime && sessionStartTime) {
        const newEndTime = new Date();
        setSessionEndTime(newEndTime);

        if (sessionStartTime && stimuliRef.current.length > 0) {
          const exportContent = createExportContent(
            participantNumber,
            sessionStartTime,
            newEndTime
          );
          const fileName = createFileName(participantNumber);

          try {
            await saveDataToFile(exportContent, fileName);
            setHasExportedData(true);
          } catch (err) {
            console.error('Failed to export data:', err);
          }
        }
      } else if (
        !hasExportedData &&
        sessionStartTime &&
        sessionEndTime &&
        stimuliRef.current.length > 0
      ) {
        await exportSessionData();
      }

      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }

      // Reset all state
      setIsPresenting(false);
      setShowDirections(false);
      setCountdown(null);
      setSessionComplete(false);
      setCurrentStimulusIndex(0);
      setEmojiPosition({
        x: SIMULATION_CENTER_X,
        y: SIMULATION_CENTER_Y,
      });
      setShowTrigger(false);
      setRewardCollected(false);
      setIsStopTrial(false);
      setShowStopSign(false);

      setSessionStartTime(null);
      setSessionEndTime(null);
      setTrialData([]);
      setHasExportedData(false);
    }, TRIGGER_DURATION);
  }, [
    hasExportedData,
    sessionStartTime,
    sessionEndTime,
    exportSessionData,
    participantNumber,
  ]);

  // Create export file content
  const createExportContent = (
    partNum: string,
    startTime: Date,
    endTime: Date
  ): string => {
    const startTimeStr = startTime.toISOString();
    const endTimeStr = endTime.toISOString();
    const allStimuli = stimuliRef.current;

    let content = `Participant ID: ${partNum}\n`;
    content += `Session Start: ${startTimeStr}\n`;
    content += `Session End: ${endTimeStr}\n`;
    content += `Total Stimuli: ${allStimuli.length}\n`;
    content += `Session Duration: ${(endTime.getTime() - startTime.getTime()) / 1000} seconds\n\n`;

    // Updated trigger pattern information for EEG processing
    content += `Trigger Pattern Information:\n`;
    content += `- Stimulus Trigger: Single flash (${TRIGGER_DURATION}ms)\n\n`;

    content += `Stimulus Order:\n`;

    allStimuli.forEach((stimulus, index) => {
      content += `Stimulus ${index + 1}: ${stimulus.direction.toUpperCase()}\n`;
    });

    // Include trial timing information if available
    if (trialData.length > 0) {
      content += `\nTrial Timing Data:\n`;
      trialData.forEach((trial) => {
        const endTimeStr = trial.endTime
          ? trial.endTime.toISOString()
          : 'incomplete';
        const duration = trial.endTime
          ? `${(trial.endTime.getTime() - trial.startTime.getTime()) / 1000} seconds`
          : 'unknown';

        content += `Trial ${trial.trialIndex + 1}: Direction=${trial.direction}, Start=${trial.startTime.toISOString()}, End=${endTimeStr}, Duration=${duration}\n`;
      });
    }

    return content;
  };

  // Generate standardized filename for export
  const createFileName = (partNum: string): string => {
    const date = new Date();
    const dateStr = date.toISOString().split('T')[0];
    const timeStr = date.toTimeString().split(' ')[0].replace(/:/g, '');
    return `participant_${partNum.padStart(3, '0')}_${dateStr}_${timeStr}.txt`;
  };

  // Save data to file
  const saveDataToFile = async (
    content: string,
    fileName: string
  ): Promise<boolean> => {
    const stimulusApi = (window as any).stimulusApi;

    if (stimulusApi && typeof stimulusApi.saveSessionData === 'function') {
      const result = await stimulusApi.saveSessionData(content, fileName);

      if (result && result.success) {
        return true;
      } else {
        console.error(
          `Failed to save data:`,
          result ? result.message : 'No result returned'
        );
        return false;
      }
    } else {
      console.warn(
        'Electron API not available, using browser download fallback'
      );
      const blob = new Blob([content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);

      const a = document.createElement('a');
      a.href = url;
      a.download = fileName;
      document.body.appendChild(a);
      a.click();

      setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }, 100);

      return true;
    }
  };

  // Start the stimulus presentation sequence
  const handleStartPresentation = useCallback(() => {
    if (!participantNumber.trim()) {
      setError('Please enter a participant number');
      return;
    }

    if (stimuliPerDirection < 1) {
      setError('Number of stimuli per direction must be at least 1');
      return;
    }

    if (delayBeforeMovement < 500) {
      setError('Delay must be at least 500ms');
      return;
    }

    // Show single trigger for session start
    setShowTrigger(true);

    setTimeout(() => {
      setShowTrigger(false);

      setSessionStartTime(new Date());
      setSessionEndTime(null);
      setTrialData([]);
      setHasExportedData(false);
      setCurrentStimulusIndex(0);

      stimuliRef.current = generateStimuli();

      if (stimuliRef.current.length === 0) {
        setError(
          'No stimuli to present. Please set the number of stimuli per direction to at least 1.'
        );
        return;
      }

      // Show instructions and then start countdown
      setShowDirections(true);

      const directionsTimer = setTimeout(() => {
        setShowDirections(false);
        setCountdown(5);

        let count = 5;
        const countdownInterval = setInterval(() => {
          count--;
          if (count <= 0) {
            clearInterval(countdownInterval);
            setCountdown(null);
            setIsPresenting(true);
          } else {
            setCountdown(count);
          }
        }, 1000);

        return () => {
          clearInterval(countdownInterval);
        };
      }, DIRECTION_DISPLAY_DURATION);

      return () => {
        clearTimeout(directionsTimer);
      };
    }, TRIGGER_DURATION);
  }, [
    generateStimuli,
    participantNumber,
    stimuliPerDirection,
    delayBeforeMovement,
  ]);

  // Effect to present next stimulus when current index changes
  useEffect(() => {
    if (isPresenting) {
      if (currentStimulusIndex < stimuliRef.current.length) {
        presentNextStimulus();
      } else {
        setIsPresenting(false);
        setSessionComplete(true);
      }
    }
  }, [currentStimulusIndex, isPresenting, presentNextStimulus]);

  // Keyboard shortcut effect
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        e.key === 'Escape' &&
        (isPresenting || showDirections || countdown !== null)
      ) {
        handleStopPresentation();
      } else if (
        e.key === 's' &&
        !isPresenting &&
        !showDirections &&
        countdown === null &&
        !sessionComplete
      ) {
        handleStartPresentation();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [
    handleStopPresentation,
    handleStartPresentation,
    isPresenting,
    showDirections,
    countdown,
    sessionComplete,
  ]);

  // Cleanup effect
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  return (
    <div
      style={{
        height: '100vh',
        width: '100vw',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        margin: 0,
        padding: 0,
        overflow: 'hidden',
        position: 'fixed',
        top: 0,
        left: 0,
        backgroundColor: '#f8f9fa',
      }}
    >
      {isPresenting ||
      showDirections ||
      countdown !== null ||
      sessionComplete ? (
        <div className="stimulus-presentation w-full h-full">
          <Card
            className="h-full w-full"
            style={{
              height: '100vh',
              width: '100vw',
              margin: 0,
              borderRadius: 0,
            }}
          >
            <Card.Header className="d-flex justify-content-between align-items-center">
              <span>
                {showDirections
                  ? 'Instructions'
                  : countdown !== null
                    ? 'Starting Soon...'
                    : sessionComplete
                      ? `Session Complete - Participant ${participantNumber}`
                      : isPresenting &&
                          currentStimulusIndex < stimuliRef.current.length
                        ? `Stimulus ${currentStimulusIndex + 1} of ${stimuliRef.current.length} (${stimuliRef.current[currentStimulusIndex].direction})`
                        : ''}
              </span>
              {!sessionComplete && (
                <Button
                  variant="danger"
                  onClick={handleStopPresentation}
                  aria-label="Stop presentation"
                >
                  Stop (ESC)
                </Button>
              )}
            </Card.Header>

            <Card.Body
              style={{
                height: 'calc(100vh - 126px)',
                width: '100%',
                position: 'relative',
                padding: 0,
                overflow: 'hidden',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                backgroundColor: '#fff',
              }}
            >
              {showDirections ? (
                <InstructionsView />
              ) : countdown !== null ? (
                <CountdownView value={countdown} />
              ) : sessionComplete ? (
                <CompletionView
                  onRestart={handleReturnToConfig}
                  totalTrials={stimuliRef.current.length}
                />
              ) : isPresenting &&
                currentStimulusIndex < stimuliRef.current.length ? (
                <IntersectionView
                  emojiPosition={emojiPosition}
                  rewardPosition={rewardPosition}
                  showTrigger={showTrigger}
                  rewardCollected={rewardCollected}
                  isStopTrial={isStopTrial && showStopSign}
                />
              ) : null}
            </Card.Body>
          </Card>
        </div>
      ) : (
        <ConfigurationPanel
          stimuliPerDirection={stimuliPerDirection}
          setStimuliPerDirection={setStimuliPerDirection}
          delayBeforeMovement={delayBeforeMovement}
          setDelayBeforeMovement={setDelayBeforeMovement}
          participantNumber={participantNumber}
          setParticipantNumber={setParticipantNumber}
          onStart={handleStartPresentation}
          error={error}
          setError={setError}
        />
      )}
    </div>
  );
};

export default StimulusDisplay;
