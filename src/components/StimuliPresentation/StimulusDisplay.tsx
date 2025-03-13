import { useState, useEffect, useRef, useCallback } from 'react';
import { Card, Button, Form, Row, Col, Alert } from 'react-bootstrap';

const DIRECTIONS = ['left', 'right', 'forward', 'backward', 'stop'];
const DIRECTION_DISPLAY_DURATION = 5000;
const TRIGGER_DISPLAY_DURATION = 100;

interface Stimulus {
  id: string;
  type: 'text' | 'image';
  content: string;
  duration: number;
  category: string;
}

interface ImagePathsState {
  loaded: boolean;
  paths: Record<string, string[]>;
}

declare global {
  interface Window {
    stimulusApi: {
      getImagePaths: () => Promise<{
        success: boolean;
        paths?: Record<string, string[]>;
        message?: string;
      }>;
    };
  }
}

const ConfigurationPanel = ({ 
  stimulusType, 
  setType, 
  numStimuli, 
  setNumStimuli,
  stimulusDuration, 
  setDuration,
  onStart,
  error,
  setError
}: {
  stimulusType: 'text' | 'image';
  setType: (type: 'text' | 'image') => void;
  numStimuli: number;
  setNumStimuli: (num: number) => void;
  stimulusDuration: number;
  setDuration: (duration: number) => void;
  onStart: () => void;
  error: string | null;
  setError: (error: string | null) => void;
}) => {
  const handleNumStimuliChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value);
    if (isNaN(value) || value < 1) {
      setError('Number of stimuli must be at least 1');
      return;
    }
    setNumStimuli(value);
  };

  const handleDurationChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value);
    if (isNaN(value) || value < 100) {
      setError('Duration must be at least 100ms');
      return;
    }
    setDuration(value);
  };

  return (
    <div className="stimulus-configuration w-full px-4">
      <Card className="mb-4 mx-auto" style={{ maxWidth: '1200px' }}>
        <Card.Header>
          <h4 className="mb-0">Configure Stimuli</h4>
        </Card.Header>
        <Card.Body>
          {error && <Alert variant="danger" dismissible onClose={() => setError(null)}>{error}</Alert>}
          
          <Form className="p-3">
            <Row className="mb-4">
              <Col md={4}>
                <Form.Group>
                  <Form.Label><strong>Stimulus Type</strong></Form.Label>
                  <Form.Select 
                    value={stimulusType} 
                    onChange={(e) => setType(e.target.value as 'text' | 'image')}
                    className="form-control-lg"
                    aria-label="Select stimulus type"
                  >
                    <option value="text">All Text</option>
                    <option value="image">All Images</option>
                  </Form.Select>
                  <Form.Text>
                    {stimulusType === 'image' ? 
                      'Presents only image stimuli from your assets folder' : 
                      'Presents directional text (LEFT, RIGHT, etc.)'}
                  </Form.Text>
                </Form.Group>
              </Col>
              <Col md={4}>
                <Form.Group>
                  <Form.Label><strong>Number of Stimuli</strong></Form.Label>
                  <Form.Control 
                    type="number" 
                    value={numStimuli}
                    onChange={handleNumStimuliChange}
                    min={1}
                    max={100}
                    className="form-control-lg"
                    aria-label="Number of stimuli"
                  />
                  <Form.Text>
                    Total number of stimuli to present in sequence
                  </Form.Text>
                </Form.Group>
              </Col>
              <Col md={4}>
                <Form.Group>
                  <Form.Label><strong>Stimulus Duration (ms)</strong></Form.Label>
                  <Form.Control 
                    type="number" 
                    value={stimulusDuration}
                    onChange={handleDurationChange}
                    min={100}
                    step={100}
                    className="form-control-lg"
                    aria-label="Stimulus duration in milliseconds"
                  />
                  <Form.Text>
                    How long each stimulus appears on screen
                  </Form.Text>
                </Form.Group>
              </Col>
            </Row>
            
            <div className="text-center mt-4">
              <Button 
                variant="success" 
                size="lg"
                onClick={onStart}
                className="px-5"
              >
                Start Presentation
              </Button>
            </div>
          </Form>
        </Card.Body>
      </Card>
    </div>
  );
};

const Instructions = () => (
  <div style={{
    width: '80%',
    height: '80%',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'white',
    borderRadius: '5px',
    padding: '2rem'
  }}>
    <div className="text-center">
      <h2>Stimulus Presentation</h2>
      <p>You will be shown a series of direction stimuli.</p>
      <p>Please focus on the center of the screen.</p>
      <p>The presentation will begin after a 5-second countdown.</p>
    </div>
  </div>
);

const Countdown = ({ value }: { value: number }) => (
  <div style={{
    width: '100%',
    height: '100%',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'white'
  }}>
    <h1 style={{ fontSize: '10rem', margin: 0 }}>{value}</h1>
  </div>
);

const StimulusView = ({ stimulus }: { stimulus: Stimulus }) => (
  <div style={{ 
    width: '100%', 
    height: '100%',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center'
  }}>
    {stimulus.type === 'text' ? (
      <h1 style={{ 
        fontSize: '15rem', 
        color: 'black',
        textAlign: 'center',
        margin: 0
      }}>{stimulus.content}</h1>
    ) : (
      <div style={{ 
        width: '100%',
        height: '100%',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        overflow: 'hidden'
      }}>
        <img 
          src={stimulus.content} 
          alt={`${stimulus.category} stimulus`} 
          style={{ 
            objectFit: 'contain',
            maxWidth: '100%',
            maxHeight: '100%'
          }} 
        />
      </div>
    )}
  </div>
);

const StimulusDisplay = () => {
  const [stimulusType, setStimulusType] = useState<'text' | 'image'>('image');
  const [numStimuli, setNumStimuli] = useState(10);
  const [stimulusDuration, setStimulusDuration] = useState(1000);
  const [isPresenting, setIsPresenting] = useState(false);
  const [currentStimulus, setCurrentStimulus] = useState<Stimulus | null>(null);
  const [showDirections, setShowDirections] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showTrigger, setShowTrigger] = useState(false);
  const [imagePaths, setImagePaths] = useState<ImagePathsState>({
    loaded: false,
    paths: {}
  });
  
  const stimuliRef = useRef<Stimulus[]>([]);
  const currentIndexRef = useRef<number>(0);
  
  // Load available images from the file system
  useEffect(() => {
    const loadImagePaths = async () => {
      try {
        const result = await window.stimulusApi.getImagePaths();
        
        if (result.success && result.paths) {
          setImagePaths({
            loaded: true,
            paths: result.paths
          });
          setError(null);
        } else {
          setError(`Failed to load images: ${result.message}`);
          setImagePaths({
            loaded: true,
            paths: {}
          });
        }
      } catch (err) {
        setError('Failed to load images. Using text fallback.');
        setImagePaths({
          loaded: true,
          paths: {}
        });
      }
    };
    
    loadImagePaths();
  }, []);
  
  // Generate stimuli array based on settings
  const generateStimuli = () => {
    let generatedStimuli: Stimulus[] = [];
    
    // For "All Images" mode, first collect all available images across categories
    const allAvailableImages: {path: string, category: string}[] = [];
    if (stimulusType === 'image') {
      DIRECTIONS.forEach(direction => {
        const directionImages = imagePaths.paths[direction] || [];
        directionImages.forEach(imagePath => {
          allAvailableImages.push({
            path: imagePath,
            category: direction
          });
        });
      });
      
      if (allAvailableImages.length === 0) {
        setError('No images found in any category. Please add images or switch to text mode.');
        return [];
      }
    }
    
    for (let i = 0; i < numStimuli; i++) {
      const randomDirection = DIRECTIONS[Math.floor(Math.random() * DIRECTIONS.length)];
      
      if (stimulusType === 'text') {
        generatedStimuli.push({
          id: Date.now().toString() + i,
          type: 'text',
          content: randomDirection.toUpperCase(),
          duration: stimulusDuration,
          category: randomDirection,
        });
      } else {
        const directionImages = imagePaths.paths[randomDirection] || [];
        
        if (directionImages.length === 0) {
          const randomImageData = allAvailableImages[Math.floor(Math.random() * allAvailableImages.length)];
          generatedStimuli.push({
            id: Date.now().toString() + i,
            type: 'image',
            content: randomImageData.path,
            duration: stimulusDuration,
            category: randomImageData.category,
          });
        } else {
          const randomImage = directionImages[Math.floor(Math.random() * directionImages.length)];
          generatedStimuli.push({
            id: Date.now().toString() + i,
            type: 'image',
            content: randomImage,
            duration: stimulusDuration,
            category: randomDirection,
          });
        }
      }
    }
    
    return generatedStimuli;
  };
  
  const handleStopPresentation = useCallback(() => {
    setIsPresenting(false);
    setCurrentStimulus(null);
    setShowTrigger(false);
    setCountdown(null);
    setShowDirections(false);
    currentIndexRef.current = 0;
  }, []);
  
  const presentNextStimulus = useCallback(() => {
    if (currentIndexRef.current >= stimuliRef.current.length) {
      handleStopPresentation();
      return;
    }
    
    const stimulus = stimuliRef.current[currentIndexRef.current];
    setCurrentStimulus(stimulus);
    setShowTrigger(true);
    
    // Hide trigger after 100ms
    const triggerTimer = setTimeout(() => {
      setShowTrigger(false);
    }, TRIGGER_DISPLAY_DURATION);
    
    // Schedule the next stimulus
    const nextTimer = setTimeout(() => {
      currentIndexRef.current++;
      presentNextStimulus();
    }, stimulus.duration);
    
    return () => {
      clearTimeout(triggerTimer);
      clearTimeout(nextTimer);
    };
  }, [handleStopPresentation]);
  
  const handleStartPresentation = useCallback(() => {
    stimuliRef.current = generateStimuli();
    
    if (stimuliRef.current.length === 0) {
      setError('No stimuli to present');
      return;
    }
    
    setShowDirections(true);
    
    // After a delay, start the countdown
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
          currentIndexRef.current = 0;
          presentNextStimulus();
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
  }, [generateStimuli, presentNextStimulus]);
  
  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && (isPresenting || showDirections || countdown !== null)) {
        handleStopPresentation();
      } else if (e.key === 's' && !isPresenting && !showDirections && countdown === null) {
        handleStartPresentation();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleStopPresentation, handleStartPresentation, isPresenting, showDirections, countdown]);
  
  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      if (isPresenting) {
        // Adjust any size-dependent variables here
      }
    };
    
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [isPresenting]);
  
  if (!imagePaths.loaded) {
    return (
      <div className="h-screen w-screen flex items-center justify-center">
        <div className="text-center p-5">
          <div className="spinner-border" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
          <p className="mt-3">Loading image resources...</p>
        </div>
      </div>
    );
  }
  
  return (
    <div style={{ 
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
      backgroundColor: '#f8f9fa'
    }}>
      {isPresenting || showDirections || countdown !== null ? (
        <div className="stimulus-presentation w-full h-full">
          <Card className="h-full w-full" style={{ 
            height: '100vh', 
            width: '100vw', 
            margin: 0, 
            borderRadius: 0 
          }}>
            <Card.Header className="d-flex justify-content-between align-items-center">
              <span>
                {showDirections ? 'Instructions' : 
                 countdown !== null ? 'Starting Soon...' : 
                 `Presenting Stimulus: ${currentIndexRef.current + 1} of ${stimuliRef.current.length}`}
              </span>
              <Button 
                variant="danger" 
                onClick={handleStopPresentation}
                aria-label="Stop presentation"
              >
                Stop (ESC)
              </Button>
            </Card.Header>
            
            <Card.Body style={{ 
                height: 'calc(100vh - 126px)', 
                width: '100%',
                position: 'relative',
                padding: 0,
                overflow: 'hidden',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                backgroundColor: '#fff'
              }}>
              {showTrigger && currentStimulus && !showDirections && countdown === null && (
                <div 
                  style={{ 
                    position: 'absolute', 
                    bottom: '0px', 
                    right: '0px', 
                    width: '50px', 
                    height: '50px', 
                    backgroundColor: 'black',
                    zIndex: 1000
                  }} 
                  aria-hidden="true"
                />
              )}
              
              {showDirections ? (
                <Instructions />
              ) : countdown !== null ? (
                <Countdown value={countdown} />
              ) : currentStimulus && (
                <StimulusView stimulus={currentStimulus} />
              )}
            </Card.Body>
          </Card>
        </div>
      ) : (
        <ConfigurationPanel 
          stimulusType={stimulusType}
          setType={setStimulusType}
          numStimuli={numStimuli}
          setNumStimuli={setNumStimuli}
          stimulusDuration={stimulusDuration}
          setDuration={setStimulusDuration}
          onStart={handleStartPresentation}
          error={error}
          setError={setError}
        />
      )}
    </div>
  );
};

export default StimulusDisplay;