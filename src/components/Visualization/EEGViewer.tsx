import { useState, useEffect, useRef } from 'react';
import { Card, Button, Form, Row, Col, Alert } from 'react-bootstrap';
import { getLatestData } from '../../utils/brainflowHelper';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface EEGViewerProps {
  isStreaming?: boolean;
}

interface ChartDataPoint {
  time: number;
  [key: string]: number; // For channel data (ch1, ch2, etc.)
}

const EEGViewer: React.FC<EEGViewerProps> = ({ isStreaming = false }) => {
  const [eegData, setEegData] = useState<{ timestamps: number[], channels: number[][] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedChannels, setSelectedChannels] = useState<number[]>([0, 1, 2, 3, 4, 5, 6, 7]); // Default to first 8 channels
  const [timeWindow, setTimeWindow] = useState<number>(5); // Display 5 seconds of data by default
  const [autoScale, setAutoScale] = useState<boolean>(true);
  const [yScale, setYScale] = useState<number>(200); // μV
  const [updateInterval, setUpdateInterval] = useState<number>(250); // Update every 250ms
  
  // Total number of channels (Cyton + Daisy = 16 channels)
  const totalChannels = 16;
  
  // Reference for the update interval
  const updateTimerRef = useRef<number | null>(null);
  // Store data buffer
  const dataBufferRef = useRef<{ timestamps: number[], channels: number[][] }>({
    timestamps: [],
    channels: Array(totalChannels).fill([]).map(() => [])  // Initialize 16 channels
  });
  
  // Setup the update interval when streaming starts/stops
  useEffect(() => {
    if (isStreaming) {
      startDataFetching();
    } else {
      stopDataFetching();
    }
    
    return () => {
      stopDataFetching();
    };
  }, [isStreaming, updateInterval]);
  
  const startDataFetching = () => {
    if (updateTimerRef.current) {
      clearInterval(updateTimerRef.current);
    }
    
    // Start periodic update
    updateTimerRef.current = window.setInterval(fetchLatestData, updateInterval);
    
    fetchLatestData();
  };
  
  const stopDataFetching = () => {
    if (updateTimerRef.current) {
      clearInterval(updateTimerRef.current);
      updateTimerRef.current = null;
    }
  };
  
  const fetchLatestData = async () => {
    try {
      const latestData = await getLatestData();
      
      if (latestData) {
        // Add new data to our buffer
        updateDataBuffer(latestData);
        
        // Update state with the buffered data (limited to timeWindow seconds)
        setEegData({
          timestamps: [...dataBufferRef.current.timestamps],
          channels: [...dataBufferRef.current.channels]
        });
        
        setError(null);
      }
    } catch (err) {
      console.error('Error fetching EEG data:', err);
      setError('Failed to fetch EEG data');
    }
  };
  
  const updateDataBuffer = (newData: { timestamps: number[], channels: number[][] }) => {
    const { timestamps, channels } = dataBufferRef.current;
    
    // Append new timestamps
    dataBufferRef.current.timestamps = [...timestamps, ...newData.timestamps];
    
    // Append new channel data
    for (let i = 0; i < channels.length; i++) {
      if (i < newData.channels.length) {
        dataBufferRef.current.channels[i] = [...channels[i], ...newData.channels[i]];
      }
    }
    
    // Trim buffer to keep only data from the last timeWindow seconds
    const now = Date.now();
    const cutoffTime = now - (timeWindow * 1000);
    
    // Find the index where we should start keeping data
    const startIndex = dataBufferRef.current.timestamps.findIndex(ts => ts >= cutoffTime);
    
    if (startIndex > 0) {
      dataBufferRef.current.timestamps = dataBufferRef.current.timestamps.slice(startIndex);
      dataBufferRef.current.channels = dataBufferRef.current.channels.map(channel => 
        channel.slice(startIndex)
      );
    }
  };
  
  const clearData = () => {
    dataBufferRef.current = {
      timestamps: [],
      channels: Array(totalChannels).fill([]).map(() => [])
    };
    setEegData(null);
  };
  
  const formatDataForChart = (): ChartDataPoint[] => {
    if (!eegData || !eegData.timestamps.length) return [];
    
    // Convert the raw data into a format that Recharts can use
    return eegData.timestamps.map((_timestamp, idx) => {
      const point: ChartDataPoint = { 
        time: idx, // Use index as the x-axis for smoother display
      };
      
      // Add only selected channels
      selectedChannels.forEach(chIdx => {
        if (eegData.channels[chIdx] && eegData.channels[chIdx][idx] !== undefined) {
          point[`ch${chIdx + 1}`] = eegData.channels[chIdx][idx];
        } else {
          point[`ch${chIdx + 1}`] = 0;
        }
      });
      
      return point;
    });
  };
  
  // Get min/max values for auto-scaling
  const getDataRange = () => {
    if (!eegData || !eegData.channels.length) return { min: -yScale, max: yScale };
    
    let min = 0, max = 0;
    
    selectedChannels.forEach(chIdx => {
      if (eegData.channels[chIdx]) {
        const channelMin = Math.min(...eegData.channels[chIdx]);
        const channelMax = Math.max(...eegData.channels[chIdx]);
        
        if (channelMin < min) min = channelMin;
        if (channelMax > max) max = channelMax;
      }
    });
    
    // Add 10% padding
    const padding = (max - min) * 0.1;
    return {
      min: min - padding,
      max: max + padding
    };
  };
  
  const dataRange = autoScale ? getDataRange() : { min: -yScale, max: yScale };
  const chartData = formatDataForChart();
  
  // Channel colors for the chart
  const channelColors = [
    '#FF5733', '#33FF57', '#3357FF', '#F033FF', // Channels 1-4
    '#FF33F0', '#33FFF0', '#F0FF33', '#8033FF', // Channels 5-8
    '#FF8C33', '#33FF8C', '#8C33FF', '#FF338C', // Channels 9-12
    '#338CFF', '#8CFF33', '#FF5733', '#33FF57'  // Channels 13-16
  ];
  
  const tooltipFormatter = (value: number) => {
    return [`${value.toFixed(2)} μV`, ''];
  };
  
  return (
    <Card>
      <Card.Header>EEG Data Visualization (16 Channels)</Card.Header>
      <Card.Body>
        {error && <Alert variant="danger">{error}</Alert>}
        
        <Row className="mb-3">
          <Col md={3}>
            <Form.Group>
              <Form.Label>Time Window (seconds)</Form.Label>
              <Form.Control 
                type="number" 
                value={timeWindow}
                min={1}
                max={60}
                onChange={(e) => setTimeWindow(parseInt(e.target.value))}
              />
            </Form.Group>
          </Col>
          
          <Col md={3}>
            <Form.Group>
              <Form.Label>Update Interval (ms)</Form.Label>
              <Form.Control 
                type="number" 
                value={updateInterval}
                min={100}
                max={1000}
                step={50}
                onChange={(e) => setUpdateInterval(parseInt(e.target.value))}
              />
            </Form.Group>
          </Col>
          
          <Col md={3}>
            <Form.Group>
              <Form.Label>Y-Axis Scale (μV)</Form.Label>
              <Form.Control 
                type="number" 
                value={yScale}
                min={10}
                max={1000}
                step={10}
                onChange={(e) => setYScale(parseInt(e.target.value))}
                disabled={autoScale}
              />
            </Form.Group>
          </Col>
          
          <Col md={3}>
            <Form.Group className="mt-4">
              <Form.Check 
                type="checkbox" 
                label="Auto Scale Y-Axis" 
                checked={autoScale}
                onChange={(e) => setAutoScale(e.target.checked)}
              />
            </Form.Group>
          </Col>
        </Row>
        
        <Row className="mb-3">
          <Col>
            <Form.Group>
              <Form.Label>Channels to Display</Form.Label>
              <div className="d-flex flex-wrap">
                {/* First 8 channels (Cyton) */}
                <div className="me-4">
                  <p className="mb-1 fw-bold">Cyton (1-8):</p>
                  {[0, 1, 2, 3, 4, 5, 6, 7].map(chIdx => (
                    <Form.Check 
                      key={chIdx}
                      inline
                      type="checkbox" 
                      id={`channel-${chIdx}`}
                      label={`Ch ${chIdx + 1}`}
                      checked={selectedChannels.includes(chIdx)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedChannels([...selectedChannels, chIdx]);
                        } else {
                          setSelectedChannels(selectedChannels.filter(ch => ch !== chIdx));
                        }
                      }}
                      style={{ color: channelColors[chIdx] }}
                    />
                  ))}
                </div>
                
                {/* Next 8 channels (Daisy) */}
                <div>
                  <p className="mb-1 fw-bold">Daisy (9-16):</p>
                  {[8, 9, 10, 11, 12, 13, 14, 15].map(chIdx => (
                    <Form.Check 
                      key={chIdx}
                      inline
                      type="checkbox" 
                      id={`channel-${chIdx}`}
                      label={`Ch ${chIdx + 1}`}
                      checked={selectedChannels.includes(chIdx)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedChannels([...selectedChannels, chIdx]);
                        } else {
                          setSelectedChannels(selectedChannels.filter(ch => ch !== chIdx));
                        }
                      }}
                      style={{ color: channelColors[chIdx] }}
                    />
                  ))}
                </div>
              </div>
            </Form.Group>
          </Col>
        </Row>
        
        <Row>
          <Col>
            <div style={{ width: '100%', height: 400 }}>
              {chartData.length > 0 ? (
                <ResponsiveContainer>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="time" 
                      label={{ value: 'Time', position: 'insideBottomRight', offset: -10 }} 
                    />
                    <YAxis 
                      domain={[dataRange.min, dataRange.max]} 
                      label={{ value: 'Amplitude (μV)', angle: -90, position: 'insideLeft' }} 
                    />
                    <Tooltip formatter={tooltipFormatter} />
                    <Legend />
                    
                    {selectedChannels.map(chIdx => (
                      <Line 
                        key={`ch${chIdx + 1}`}
                        type="monotone" 
                        dataKey={`ch${chIdx + 1}`} 
                        name={`Channel ${chIdx + 1}`} 
                        stroke={channelColors[chIdx]}
                        dot={false}
                        isAnimationActive={false}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="d-flex justify-content-center align-items-center h-100">
                  <p className="text-muted">
                    {isStreaming 
                      ? 'Waiting for EEG data...' 
                      : 'Start streaming to view EEG data'
                    }
                  </p>
                </div>
              )}
            </div>
          </Col>
        </Row>
        
        {/* Extra tools */}
        <Row className="mt-3">
          <Col>
            <Button variant="secondary" onClick={clearData}>
              Clear Data
            </Button>
            
            <Button 
              variant="outline-primary" 
              className="ms-2"
              onClick={() => setSelectedChannels(Array.from({ length: totalChannels }, (_, i) => i))}
            >
              Select All Channels
            </Button>
            
            <Button 
              variant="outline-secondary" 
              className="ms-2"
              onClick={() => setSelectedChannels([])}
            >
              Deselect All
            </Button>
          </Col>
        </Row>
      </Card.Body>
    </Card>
  );
};

export default EEGViewer;