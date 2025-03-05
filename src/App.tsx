import { useState } from 'react'
import { Container, Row, Col, Navbar, Nav } from 'react-bootstrap'
import './App.css'
import DeviceConnection from './components/EEGInterface/DeviceConnection'
import StimulusDisplay from './components/StimuliPresentation/StimulusDisplay'
import RecordingPanel from './components/RecordingControl/RecordingPanel'
import EEGViewer from './components/Visualization/EEGViewer'

interface BoardInfo {
  sampleRate: number;
  eegChannels: number[];
  totalChannels: number;
}

function App() {
  const [activeTab, setActiveTab] = useState('setup')
  const [isStreaming, setIsStreaming] = useState(false)
  const [boardInfo, setBoardInfo] = useState<BoardInfo | null>(null)

  // Handle streaming state change from DeviceConnection
  const handleStreamingChange = (streaming: boolean, info: BoardInfo | null = null) => {
    setIsStreaming(streaming);
    if (info) {
      setBoardInfo(info);
    }
  };

  return (
    <div className="App">
      <Navbar bg="dark" variant="dark" expand="lg">
        <Container>
          <Navbar.Brand>EEG Training App</Navbar.Brand>
          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="me-auto">
              <Nav.Link onClick={() => setActiveTab('setup')}>Setup</Nav.Link>
              <Nav.Link onClick={() => setActiveTab('stimuli')}>Stimuli</Nav.Link>
              <Nav.Link onClick={() => setActiveTab('recording')}>Recording</Nav.Link>
              <Nav.Link onClick={() => setActiveTab('visualization')}>Visualization</Nav.Link>
            </Nav>
            {isStreaming && (
              <div className="d-flex align-items-center text-light">
                <div className="me-2" 
                  style={{ 
                    width: '10px', 
                    height: '10px', 
                    borderRadius: '50%', 
                    backgroundColor: '#28a745',
                    animation: 'pulse 1s infinite' 
                  }} 
                />
                Streaming
              </div>
            )}
          </Navbar.Collapse>
        </Container>
      </Navbar>

      <Container fluid className="mt-3">
        {activeTab === 'setup' && (
          <Row>
            <Col>
              <h2>Device Setup</h2>
              <DeviceConnection 
                onStreamingChange={handleStreamingChange}
              />
            </Col>
          </Row>
        )}

        {activeTab === 'stimuli' && (
          <Row>
            <Col>
              <h2>Stimulus Presentation</h2>
              <StimulusDisplay />
            </Col>
          </Row>
        )}

        {activeTab === 'recording' && (
          <Row>
            <Col>
              <h2>Recording Controls</h2>
              <RecordingPanel 
                isStreaming={isStreaming}
                boardInfo={boardInfo || undefined}
              />
            </Col>
          </Row>
        )}

        {activeTab === 'visualization' && (
          <Row>
            <Col>
              <h2>EEG Visualization</h2>
              <EEGViewer 
                isStreaming={isStreaming}
              />
            </Col>
          </Row>
        )}
      </Container>
    </div>
  )
}

export default App