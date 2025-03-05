import { useState, useRef, useEffect } from 'react';
import { Card, Button, Form, Row, Col, Alert, ProgressBar, ListGroup } from 'react-bootstrap';
import { addMarker } from '../../utils/brainflowHelper';

interface RecordingPanelProps {
  isStreaming?: boolean;
  boardInfo?: {
    sampleRate: number;
    eegChannels: number[];
  };
}

interface RecordingSession {
  id: string;
  name: string;
  participantId: string;
  startTime: Date;
  endTime?: Date;
  markers: Array<{ time: Date; label: string }>;
  filePath?: string;
}

const RecordingPanel: React.FC<RecordingPanelProps> = ({ 
  isStreaming = false,
  boardInfo
}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [sessionName, setSessionName] = useState('');
  const [participantId, setParticipantId] = useState('');
  const [newMarker, setNewMarker] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [recordedSessions, setRecordedSessions] = useState<RecordingSession[]>([]);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [estimatedSize, setEstimatedSize] = useState(0);
  
  // Current recording session
  const currentSessionRef = useRef<RecordingSession | null>(null);
  
  // Timer for updating elapsed time
  const timerRef = useRef<number | null>(null);
  
  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);
  
  // Calculate estimated file size based on sample rate, channels, and time
  useEffect(() => {
    if (boardInfo && isRecording) {
      // Each data point is a 4-byte float
      const bytesPerSecond = 
        (boardInfo.sampleRate * boardInfo.eegChannels.length * 4) + 
        (boardInfo.sampleRate * 8); // Add timestamp (8 bytes)
      
      // Convert to MB
      const mbPerSecond = bytesPerSecond / (1024 * 1024);
      setEstimatedSize(mbPerSecond * elapsedTime);
    }
  }, [elapsedTime, boardInfo, isRecording]);
  
  const startRecording = () => {
    if (!isStreaming) {
      setError('Cannot start recording: Device is not streaming data');
      return;
    }
    
    if (!sessionName) {
      setError('Please enter a session name');
      return;
    }
    
    // Create a new recording session
    const newSession: RecordingSession = {
      id: Date.now().toString(),
      name: sessionName,
      participantId: participantId || 'unknown',
      startTime: new Date(),
      markers: []
    };
    
    currentSessionRef.current = newSession;
    setIsRecording(true);
    setError(null);
    setElapsedTime(0);
    
    // Start elapsed time timer
    timerRef.current = window.setInterval(() => {
      setElapsedTime(prev => prev + 1);
    }, 1000);
    
    console.log('Recording started:', newSession);
  };
  
  const stopRecording = () => {
    if (!isRecording) return;
    
    // Clear the timer
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    
    if (currentSessionRef.current) {
      // Complete the session
      const completedSession = {
        ...currentSessionRef.current,
        endTime: new Date(),
        filePath: `recordings/${currentSessionRef.current.name.replace(/\s+/g, '_')}_${Date.now()}.csv`
      };
      
      // Add to recorded sessions list
      setRecordedSessions(prev => [...prev, completedSession]);
      
      console.log('Recording stopped:', completedSession);
    }
    
    setIsRecording(false);
    currentSessionRef.current = null;
  };
  
  const addMarkerToRecording = async () => {
    if (!isRecording || !newMarker) return;
    
    try {
      // Add marker to the EEG data stream
      await addMarker(newMarker);
      
      // Add to the current session
      if (currentSessionRef.current) {
        const marker = {
          time: new Date(),
          label: newMarker
        };
        
        currentSessionRef.current.markers.push(marker);
        
        // Force a re-render to show the new marker
        setRecordedSessions(prev => [...prev]);
      }
      
      // Clear the marker input
      setNewMarker('');
    } catch (err) {
      console.error('Failed to add marker:', err);
      setError('Failed to add marker');
    }
  };
  
  const formatTime = (seconds: number): string => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    const parts = [];
    if (hrs > 0) parts.push(`${hrs}h`);
    if (mins > 0 || hrs > 0) parts.push(`${mins}m`);
    parts.push(`${secs}s`);
    
    return parts.join(' ');
  };
  
  return (
    <Card>
      <Card.Header>EEG Recording</Card.Header>
      <Card.Body>
        {error && <Alert variant="danger">{error}</Alert>}
        
        <Row className="mb-4">
          <Col md={6}>
            <Form.Group className="mb-3">
              <Form.Label>Session Name</Form.Label>
              <Form.Control 
                type="text" 
                placeholder="e.g., Visual P300 Experiment" 
                value={sessionName}
                onChange={(e) => setSessionName(e.target.value)}
                disabled={isRecording}
              />
            </Form.Group>
            
            <Form.Group className="mb-3">
              <Form.Label>Participant ID</Form.Label>
              <Form.Control 
                type="text" 
                placeholder="e.g., P001" 
                value={participantId}
                onChange={(e) => setParticipantId(e.target.value)}
                disabled={isRecording}
              />
            </Form.Group>
            
            <div className="d-grid gap-2">
              {!isRecording ? (
                <Button 
                  variant="success" 
                  onClick={startRecording}
                  disabled={!isStreaming || !sessionName}
                >
                  Start Recording
                </Button>
              ) : (
                <Button 
                  variant="danger" 
                  onClick={stopRecording}
                >
                  Stop Recording
                </Button>
              )}
            </div>
          </Col>
          
          <Col md={6}>
            {isRecording && (
              <div className="recording-status p-3 border rounded">
                <div className="d-flex align-items-center mb-2">
                  <div className="recording-indicator me-2" 
                    style={{ 
                      width: '12px', 
                      height: '12px', 
                      borderRadius: '50%', 
                      backgroundColor: 'red',
                      animation: 'pulse 1s infinite' 
                    }} 
                  />
                  <h5 className="mb-0">Recording in Progress</h5>
                </div>
                
                <p>
                  <strong>Session:</strong> {currentSessionRef.current?.name}<br />
                  <strong>Participant:</strong> {currentSessionRef.current?.participantId}<br />
                  <strong>Started:</strong> {currentSessionRef.current?.startTime.toLocaleTimeString()}<br />
                  <strong>Duration:</strong> {formatTime(elapsedTime)}<br />
                  <strong>Markers Added:</strong> {currentSessionRef.current?.markers.length || 0}<br />
                  <strong>Estimated Size:</strong> {estimatedSize.toFixed(2)} MB
                </p>
                
                <Form.Group className="mb-2">
                  <Form.Label>Add Marker</Form.Label>
                  <div className="d-flex">
                    <Form.Control 
                      type="text" 
                      placeholder="e.g., Target Displayed" 
                      value={newMarker}
                      onChange={(e) => setNewMarker(e.target.value)}
                    />
                    <Button 
                      variant="primary" 
                      className="ms-2"
                      onClick={addMarkerToRecording}
                      disabled={!newMarker}
                    >
                      Add
                    </Button>
                  </div>
                </Form.Group>
                
                <div className="mt-3">
                  <ProgressBar animated now={100} label="Recording..." />
                </div>
              </div>
            )}
            
            {!isRecording && !isStreaming && (
              <Alert variant="warning">
                Device is not streaming data. Please start streaming before recording.
              </Alert>
            )}
          </Col>
        </Row>
        
        {recordedSessions.length > 0 && (
          <div className="mt-4">
            <h5>Recorded Sessions</h5>
            <ListGroup>
              {recordedSessions.map(session => {
                const duration = session.endTime && session.startTime
                  ? Math.floor((session.endTime.getTime() - session.startTime.getTime()) / 1000)
                  : 0;
                
                return (
                  <ListGroup.Item key={session.id}>
                    <div className="d-flex justify-content-between align-items-center">
                      <div>
                        <strong>{session.name}</strong> ({session.participantId})
                        <br />
                        <small className="text-muted">
                          {session.startTime.toLocaleString()} • 
                          Duration: {formatTime(duration)} • 
                          Markers: {session.markers.length}
                        </small>
                      </div>
                      <div>
                        <Button variant="outline-secondary" size="sm">
                          Export Data
                        </Button>
                      </div>
                    </div>
                  </ListGroup.Item>
                );
              })}
            </ListGroup>
          </div>
        )}
      </Card.Body>
    </Card>
  );
};

export default RecordingPanel;