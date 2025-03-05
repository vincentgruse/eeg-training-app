import { useState, useEffect, useRef } from 'react';
import { Card, Button, Form, Row, Col, ListGroup, Alert } from 'react-bootstrap';
import { addMarker } from '../../utils/brainflowHelper';

interface Stimulus {
  id: string;
  type: 'text' | 'image';
  content: string;
  duration: number; // in milliseconds
  marker: string;
}

const StimulusDisplay = () => {
  const [stimuli, setStimuli] = useState<Stimulus[]>([]);
  const [currentStimulus, setCurrentStimulus] = useState<Stimulus | null>(null);
  const [isPresenting, setIsPresenting] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [newStimulus, setNewStimulus] = useState<Partial<Stimulus>>({
    type: 'text',
    content: '',
    duration: 1000,
    marker: '',
  });
  
  // Reference to the presentation interval
  const presentationTimerRef = useRef<number | null>(null);
  const currentIndexRef = useRef<number>(0);
  
  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (presentationTimerRef.current) {
        clearTimeout(presentationTimerRef.current);
      }
    };
  }, []);
  
  const handleAddStimulus = () => {
    if (!newStimulus.content) {
      setError('Stimulus content is required');
      return;
    }
    
    if (!newStimulus.marker) {
      setError('Marker is required');
      return;
    }
    
    const stimulus: Stimulus = {
      id: Date.now().toString(),
      type: newStimulus.type as 'text' | 'image',
      content: newStimulus.content,
      duration: newStimulus.duration || 1000,
      marker: newStimulus.marker,
    };
    
    setStimuli([...stimuli, stimulus]);
    setError(null);
    
    // Reset the form
    setNewStimulus({
      type: 'text',
      content: '',
      duration: 1000,
      marker: '',
    });
  };
  
  const handleRemoveStimulus = (id: string) => {
    setStimuli(stimuli.filter(stim => stim.id !== id));
  };
  
  const handleStartPresentation = () => {
    if (stimuli.length === 0) {
      setError('No stimuli to present');
      return;
    }
    
    setIsPresenting(true);
    setIsPaused(false);
    currentIndexRef.current = 0;
    presentNextStimulus();
  };
  
  const handlePausePresentation = () => {
    setIsPaused(true);
    if (presentationTimerRef.current) {
      clearTimeout(presentationTimerRef.current);
      presentationTimerRef.current = null;
    }
  };
  
  const handleResumePresentation = () => {
    setIsPaused(false);
    presentNextStimulus();
  };
  
  const handleStopPresentation = () => {
    setIsPresenting(false);
    setIsPaused(false);
    setCurrentStimulus(null);
    
    if (presentationTimerRef.current) {
      clearTimeout(presentationTimerRef.current);
      presentationTimerRef.current = null;
    }
  };
  
  const presentNextStimulus = async () => {
    if (currentIndexRef.current >= stimuli.length) {
      // End of sequence
      handleStopPresentation();
      return;
    }
    
    const stimulus = stimuli[currentIndexRef.current];
    setCurrentStimulus(stimulus);
    
    // Send marker to the EEG data stream
    try {
      await addMarker(stimulus.marker);
    } catch (err) {
      console.error('Failed to add marker:', err);
      // Continue presentation even if marker fails
    }
    
    // Schedule the next stimulus
    presentationTimerRef.current = window.setTimeout(() => {
      currentIndexRef.current++;
      presentNextStimulus();
    }, stimulus.duration);
  };
  
  return (
    <div>
      {isPresenting ? (
        <div className="stimulus-presentation">
          <Card className="mb-4">
            <Card.Header className="d-flex justify-content-between align-items-center">
              <span>Presenting Stimulus: {currentIndexRef.current + 1} of {stimuli.length}</span>
              <div>
                {isPaused ? (
                  <Button variant="success" onClick={handleResumePresentation}>Resume</Button>
                ) : (
                  <Button variant="warning" onClick={handlePausePresentation}>Pause</Button>
                )}
                <Button variant="danger" className="ms-2" onClick={handleStopPresentation}>Stop</Button>
              </div>
            </Card.Header>
            <Card.Body className="d-flex justify-content-center align-items-center" style={{ minHeight: '300px' }}>
              {currentStimulus && (
                currentStimulus.type === 'text' ? (
                  <h1>{currentStimulus.content}</h1>
                ) : (
                  <img 
                    src={currentStimulus.content} 
                    alt="Visual stimulus" 
                    style={{ maxWidth: '100%', maxHeight: '300px' }} 
                  />
                )
              )}
            </Card.Body>
            <Card.Footer className="text-muted">
              Current marker: {currentStimulus?.marker}, Duration: {currentStimulus?.duration}ms
            </Card.Footer>
          </Card>
        </div>
      ) : (
        <div className="stimulus-configuration">
          <Card className="mb-4">
            <Card.Header>Configure Stimuli</Card.Header>
            <Card.Body>
              {error && <Alert variant="danger">{error}</Alert>}
              
              <Form>
                <Row>
                  <Col md={3}>
                    <Form.Group className="mb-3">
                      <Form.Label>Type</Form.Label>
                      <Form.Select 
                        value={newStimulus.type} 
                        onChange={(e) => setNewStimulus({...newStimulus, type: e.target.value as 'text' | 'image'})}
                      >
                        <option value="text">Text</option>
                        <option value="image">Image</option>
                      </Form.Select>
                    </Form.Group>
                  </Col>
                  <Col md={4}>
                    <Form.Group className="mb-3">
                      <Form.Label>Content</Form.Label>
                      <Form.Control 
                        type="text" 
                        placeholder={newStimulus.type === 'text' ? "Enter text to display" : "Enter image URL"} 
                        value={newStimulus.content}
                        onChange={(e) => setNewStimulus({...newStimulus, content: e.target.value})}
                      />
                    </Form.Group>
                  </Col>
                  <Col md={2}>
                    <Form.Group className="mb-3">
                      <Form.Label>Duration (ms)</Form.Label>
                      <Form.Control 
                        type="number" 
                        value={newStimulus.duration}
                        onChange={(e) => setNewStimulus({...newStimulus, duration: parseInt(e.target.value)})}
                        min={100}
                      />
                    </Form.Group>
                  </Col>
                  <Col md={3}>
                    <Form.Group className="mb-3">
                      <Form.Label>Marker</Form.Label>
                      <Form.Control 
                        type="text" 
                        placeholder="e.g., TARGET, DISTRACTOR" 
                        value={newStimulus.marker}
                        onChange={(e) => setNewStimulus({...newStimulus, marker: e.target.value})}
                      />
                    </Form.Group>
                  </Col>
                </Row>
                
                <Button variant="primary" onClick={handleAddStimulus}>
                  Add Stimulus
                </Button>
              </Form>
              
              <hr />
              
              <h5>Stimulus Sequence ({stimuli.length} items)</h5>
              {stimuli.length > 0 ? (
                <ListGroup>
                  {stimuli.map((stim, index) => (
                    <ListGroup.Item key={stim.id} className="d-flex justify-content-between align-items-center">
                      <div>
                        <strong>{index + 1}.</strong> {stim.type === 'text' ? `"${stim.content}"` : `Image: ${stim.content}`} 
                        <span className="text-muted ms-2">({stim.duration}ms, Marker: {stim.marker})</span>
                      </div>
                      <Button variant="outline-danger" size="sm" onClick={() => handleRemoveStimulus(stim.id)}>
                        Remove
                      </Button>
                    </ListGroup.Item>
                  ))}
                </ListGroup>
              ) : (
                <Alert variant="info">No stimuli added yet</Alert>
              )}
              
              {stimuli.length > 0 && (
                <Button 
                  variant="success" 
                  className="mt-3" 
                  onClick={handleStartPresentation}
                >
                  Start Presentation
                </Button>
              )}
            </Card.Body>
          </Card>
        </div>
      )}
    </div>
  );
};

export default StimulusDisplay;