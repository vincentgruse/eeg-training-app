import React from 'react';
import { Card, Button, Form, Row, Col, Alert } from 'react-bootstrap';
import { ConfigurationPanelProps } from './types';

const ConfigurationPanel: React.FC<ConfigurationPanelProps> = ({
  stimuliPerDirection,
  setStimuliPerDirection,
  delayBeforeMovement,
  setDelayBeforeMovement,
  participantNumber,
  setParticipantNumber,
  onStart,
  error,
  setError,
}) => {
  const handleStimuliCountChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const value = parseInt(e.target.value);
    setStimuliPerDirection(isNaN(value) ? 0 : value);
  };

  const handleDelayChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const value = parseInt(e.target.value);
    setDelayBeforeMovement(isNaN(value) ? 0 : value);
  };

  const handleParticipantChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    setParticipantNumber(e.target.value);
  };

  return (
    <div className="stimulus-configuration w-full px-4">
      <Card className="mb-4 mx-auto" style={{ maxWidth: '1200px' }}>
        <Card.Header>
          <h4 className="mb-0">Configure Intersection Stimuli</h4>
        </Card.Header>
        <Card.Body>
          {error && (
            <Alert variant="danger" dismissible onClose={() => setError(null)}>
              {error}
            </Alert>
          )}

          <Form className="p-3">
            <Row className="mb-4">
              <Col md={4} className="mx-auto">
                <Form.Group>
                  <Form.Label>
                    <strong>Participant Number</strong>
                  </Form.Label>
                  <Form.Control
                    type="text"
                    value={participantNumber}
                    onChange={handleParticipantChange}
                    className="form-control-lg"
                    aria-label="Participant number"
                    placeholder="e.g. 001"
                  />
                  <Form.Text>Enter the participant's ID number</Form.Text>
                </Form.Group>
              </Col>
              <Col md={4} className="mx-auto">
                <Form.Group>
                  <Form.Label>
                    <strong>Number of Stimuli Per Direction</strong>
                  </Form.Label>
                  <Form.Control
                    type="number"
                    value={stimuliPerDirection}
                    onChange={handleStimuliCountChange}
                    min={1}
                    className="form-control-lg"
                    aria-label="Number of stimuli per direction"
                  />
                  <Form.Text>
                    Each direction will be shown this many times (randomly
                    ordered)
                  </Form.Text>
                </Form.Group>
              </Col>
              <Col md={4} className="mx-auto">
                <Form.Group>
                  <Form.Label>
                    <strong>Delay Before Movement (ms)</strong>
                  </Form.Label>
                  <Form.Control
                    type="number"
                    value={delayBeforeMovement}
                    onChange={handleDelayChange}
                    min={500}
                    step={100}
                    className="form-control-lg"
                    aria-label="Delay before movement in milliseconds"
                  />
                  <Form.Text>
                    Time to wait before character moves to collect the reward
                  </Form.Text>
                </Form.Group>
              </Col>
            </Row>

            <div className="d-flex justify-content-between align-items-center mt-4">
              <div>
                <strong>Total Stimuli: {stimuliPerDirection * 5}</strong>
              </div>
              <Button
                variant="success"
                size="lg"
                onClick={onStart}
                className="px-5"
                disabled={!participantNumber.trim()}
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

export default ConfigurationPanel;
