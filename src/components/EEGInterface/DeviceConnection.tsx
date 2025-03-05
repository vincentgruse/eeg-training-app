import { useState, useEffect } from 'react'
import { Button, Form, Card, Alert, Spinner, Badge, Row, Col, ProgressBar } from 'react-bootstrap'
import { 
  connectToDevice, 
  startDataStream, 
  stopDataStream,
  disconnectDevice 
} from '../../utils/brainflowHelper'

interface BoardInfo {
  sampleRate: number;
  eegChannels: number[];
  totalChannels: number;
}

interface DeviceConnectionProps {
  onStreamingChange?: (isStreaming: boolean, boardInfo?: BoardInfo) => void;
}

const DeviceConnection: React.FC<DeviceConnectionProps> = ({ onStreamingChange }) => {
  const [connecting, setConnecting] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const [connected, setConnected] = useState(false)
  const [boardInfo, setBoardInfo] = useState<BoardInfo | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)
  const [comPort, setComPort] = useState('')
  const [useMockMode, setUseMockMode] = useState(false)
  
  // Notify parent component when streaming state changes
  useEffect(() => {
    if (onStreamingChange) {
      onStreamingChange(streaming, boardInfo || undefined);
    }
  }, [streaming, boardInfo, onStreamingChange]);

  // Clear success message after 5 seconds
  useEffect(() => {
    if (successMessage) {
      const timer = setTimeout(() => {
        setSuccessMessage(null);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [successMessage]);
  
  const handleConnect = async () => {
    setConnecting(true)
    setError(null)
    setSuccessMessage(null)
    
    console.log("Attempting to connect to:", comPort, "Mock mode:", useMockMode);
    
    try {
      const result = await connectToDevice(comPort)
      
      console.log("Connection result:", result);
      
      if (result.success) {
        setConnected(true)
        setSuccessMessage(`Successfully connected to device on ${comPort}`)
        if (result.boardInfo) {
          setBoardInfo(result.boardInfo)
        }
      } else {
        setError(result.message || "Unknown connection error")
      }
    } catch (err) {
      console.error('Failed to connect to device:', err)
      setError('Failed to connect to device: ' + (err instanceof Error ? err.message : String(err)))
    } finally {
      setConnecting(false)
    }
  }
  
  const handleStartStream = async () => {
    setError(null)
    setSuccessMessage(null)
    
    try {
      const result = await startDataStream()
      
      console.log("Stream start result:", result);
      
      if (result.success) {
        setStreaming(true)
        setSuccessMessage('EEG data streaming started successfully')
      } else {
        setError(result.message)
      }
    } catch (err) {
      console.error('Failed to start data stream:', err)
      setError('Failed to start data stream: ' + (err instanceof Error ? err.message : String(err)))
    }
  }
  
  const handleStopStream = async () => {
    setError(null)
    setSuccessMessage(null)
    
    try {
      const result = await stopDataStream()
      
      console.log("Stream stop result:", result);
      
      if (result.success) {
        setStreaming(false)
        setSuccessMessage('EEG data streaming stopped')
      } else {
        setError(result.message)
      }
    } catch (err) {
      console.error('Failed to stop data stream:', err)
      setError('Failed to stop data stream: ' + (err instanceof Error ? err.message : String(err)))
    }
  }
  
  const handleDisconnect = async () => {
    setError(null)
    setSuccessMessage(null)
    
    try {
      // If streaming, stop it first
      if (streaming) {
        await handleStopStream()
      }
      
      const result = await disconnectDevice()
      
      console.log("Disconnect result:", result);
      
      if (result.success) {
        setConnected(false)
        setBoardInfo(null)
        setSuccessMessage('Successfully disconnected from device')
      } else {
        setError(result.message)
      }
    } catch (err) {
      console.error('Failed to disconnect from device:', err)
      setError('Failed to disconnect from device: ' + (err instanceof Error ? err.message : String(err)))
    }
  }

  // Get status badge
  const getStatusBadge = () => {
    if (connecting) {
      return <Badge bg="info" className="pulse-animation">Connecting...</Badge>;
    } else if (streaming) {
      return <Badge bg="success" className="pulse-animation">Streaming</Badge>;
    } else if (connected) {
      return <Badge bg="primary">Connected</Badge>;
    } else {
      return <Badge bg="secondary">Disconnected</Badge>;
    }
  };
  
  return (
    <Card>
      <Card.Header className="d-flex justify-content-between align-items-center">
        <span>OpenBCI Cyton + Daisy Connection</span>
        {getStatusBadge()}
      </Card.Header>
      <Card.Body>
        {/* Error message */}
        {error && (
          <Alert variant="danger" dismissible onClose={() => setError(null)}>
            <Alert.Heading>Connection Error</Alert.Heading>
            <p>{error}</p>
          </Alert>
        )}
        
        {/* Success message */}
        {successMessage && (
          <Alert variant="success" dismissible onClose={() => setSuccessMessage(null)}>
            <p className="mb-0">{successMessage}</p>
          </Alert>
        )}
        
        {!connected ? (
          <Form>
            <Form.Group className="mb-3">
              <Form.Label>Serial Port</Form.Label>
              <Form.Control 
                type="text" 
                placeholder="Enter serial port (e.g., COM3 on Windows, /dev/ttyUSB0 on Linux)" 
                value={comPort}
                onChange={(e) => setComPort(e.target.value)}
              />
              <Form.Text className="text-muted">
                Specify the port where your OpenBCI Cyton+Daisy device is connected
              </Form.Text>
            </Form.Group>
            
            <Form.Group className="mb-3">
              <Form.Check 
                type="checkbox" 
                label="Use Mock Mode (for testing without hardware)" 
                checked={useMockMode}
                onChange={(e) => setUseMockMode(e.target.checked)}
              />
              <Form.Text className="text-muted">
                This allows you to test the app functionality without an actual device
              </Form.Text>
            </Form.Group>
            
            <Button 
              variant="primary" 
              onClick={handleConnect} 
              disabled={connecting || !comPort}
              className="me-2"
            >
              {connecting ? (
                <>
                  <Spinner as="span" animation="border" size="sm" role="status" aria-hidden="true" />
                  <span className="ms-2">Connecting...</span>
                </>
              ) : (
                'Connect to Device'
              )}
            </Button>
            
            <Button 
              variant="outline-secondary"
              onClick={() => {
                if (navigator.platform.indexOf('Win') !== -1) {
                  setComPort('COM3');
                } else if (navigator.platform.indexOf('Mac') !== -1) {
                  setComPort('/dev/cu.usbserial-DM00CXN8');
                } else {
                  setComPort('/dev/ttyUSB0');
                }
              }}
            >
              Suggest Port
            </Button>
          </Form>
        ) : (
          <div>
            <Row className="connection-status mb-4 p-3 border rounded">
              <Col md={6}>
                <h5 className="mb-3 border-bottom pb-2">Device Information</h5>
                <p className="mb-2">
                  <strong>Status:</strong>{' '}
                  {streaming ? (
                    <span className="text-success">●</span>
                  ) : (
                    <span className="text-primary">●</span>
                  )}{' '}
                  {streaming ? 'Streaming' : 'Connected'}
                </p>
                <p className="mb-2"><strong>Device:</strong> OpenBCI Cyton+Daisy (16 channels)</p>
                <p className="mb-2"><strong>Port:</strong> {comPort}</p>
                
                {boardInfo && (
                  <>
                    <p className="mb-2"><strong>Sample Rate:</strong> {boardInfo.sampleRate} Hz</p>
                    <p className="mb-2"><strong>EEG Channels:</strong> {boardInfo.eegChannels.length}</p>
                    <p className="mb-2"><strong>Total Data Channels:</strong> {boardInfo.totalChannels}</p>
                  </>
                )}
              </Col>
              
              <Col md={6}>
                <h5 className="mb-3 border-bottom pb-2">Device Controls</h5>
                
                {streaming ? (
                  <div className="mb-3">
                    <p className="mb-2">
                      <strong>Streaming Status:</strong> Active
                    </p>
                    <div className="mb-2">
                      <ProgressBar animated now={100} label="Receiving data..." />
                    </div>
                    <Button 
                      variant="warning" 
                      onClick={handleStopStream}
                      className="me-2"
                    >
                      Stop Streaming
                    </Button>
                  </div>
                ) : (
                  <Button 
                    variant="success" 
                    onClick={handleStartStream}
                    className="me-2 mb-3"
                  >
                    Start Streaming
                  </Button>
                )}
                
                <div className="mt-2">
                  <Button 
                    variant="outline-danger" 
                    onClick={handleDisconnect}
                  >
                    Disconnect
                  </Button>
                </div>
              </Col>
            </Row>
            
            {/* Connection tips */}
            <Alert variant="info">
              <Alert.Heading>Connection Tips</Alert.Heading>
              <p>
                If you're having trouble with the connection:
              </p>
              <ul className="mb-0">
                <li>Make sure the device is powered on and has charged batteries</li>
                <li>Verify the USB dongle is properly connected to your computer</li>
                <li>Check if another application is using the same port</li>
                <li>Try restarting the device or reconnecting the dongle</li>
              </ul>
            </Alert>
          </div>
        )}
      </Card.Body>
      <Card.Footer className="text-muted">
        <small>OpenBCI Ultracortex Mark IV - 16 Channel EEG System</small>
      </Card.Footer>
      
      {/* Add some CSS for the pulsing animation */}
      <style>{`
        .pulse-animation {
          animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
          0% {
            opacity: 1;
          }
          50% {
            opacity: 0.6;
          }
          100% {
            opacity: 1;
          }
        }
        
        .text-success, .text-primary {
          display: inline-block;
          animation: pulse 1.5s infinite;
        }
      `}</style>
    </Card>
  )
}

export default DeviceConnection