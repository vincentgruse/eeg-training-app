import React, { useState } from 'react';
import { Button, Spinner, Alert, Form } from 'react-bootstrap';
import { CompletionViewProps } from './types';

const CompletionView: React.FC<CompletionViewProps> = ({
  onRestart,
  totalTrials,
}) => {
  const [processing, setProcessing] = useState(false);
  const [processingSuccess, setProcessingSuccess] = useState<boolean | null>(
    null
  );
  const [processingMessage, setProcessingMessage] = useState<string>('');
  const [selectedEegFile, setSelectedEegFile] = useState<string>('');

  const handleBrowseFile = async () => {
    if (window.eegProcessorApi && window.eegProcessorApi.browseForEegFile) {
      try {
        const result = await window.eegProcessorApi.browseForEegFile();
        if (result.success && result.filePath) {
          setSelectedEegFile(result.filePath);
          setProcessingMessage('');
          setProcessingSuccess(null);
        }
      } catch (error: unknown) {
        console.error('Error selecting file:', error);
      }
    } else {
      setProcessingMessage('File browsing not available in this environment');
      setProcessingSuccess(false);
    }
  };

  const handleReturn = async () => {
    // Return directly if no EEG processor API available
    if (!window.eegProcessorApi) {
      onRestart();
      return;
    }

    // If no file selected, show error
    if (!selectedEegFile) {
      setProcessingSuccess(false);
      setProcessingMessage('Please select an EEG data file first');
      return;
    }

    setProcessing(true);
    setProcessingSuccess(null);
    setProcessingMessage('');

    try {
      const result =
        await window.eegProcessorApi.processEEGData(selectedEegFile);

      if (result.success) {
        setProcessingSuccess(true);
        setProcessingMessage('EEG data processed successfully!');
      } else {
        setProcessingSuccess(false);
        setProcessingMessage(
          `Error processing EEG data: ${result.error || 'Unknown error'}`
        );
      }
    } catch (error: unknown) {
      setProcessingSuccess(false);
      const errorMessage =
        error instanceof Error ? error.message : 'Unknown error occurred';
      setProcessingMessage(`Error: ${errorMessage}`);
    } finally {
      // Slight delay before returning to show the success/error message
      setTimeout(() => {
        setProcessing(false);
        onRestart();
      }, 1500);
    }
  };

  return (
    <div
      style={{
        width: '80%',
        height: '80%',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: 'white',
        borderRadius: '10px',
        padding: '2rem',
        textAlign: 'center',
        boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
      }}
    >
      <h2 style={{ fontSize: '2rem', marginBottom: '1.5rem' }}>
        Session Complete
      </h2>
      <p style={{ fontSize: '1.2rem', marginBottom: '1rem' }}>
        Thank you for completing all {totalTrials} stimulus trials.
      </p>
      <p style={{ fontSize: '1.2rem', marginBottom: '2rem' }}>
        Your data has been recorded successfully and will be saved
        automatically.
      </p>

      <div className="w-100 mb-4" style={{ maxWidth: '500px' }}>
        <Form.Group className="mb-3">
          <Form.Label>
            <strong>Select EEG Data File</strong>
          </Form.Label>
          <div className="d-flex">
            <Form.Control
              type="text"
              placeholder="No file selected"
              value={selectedEegFile}
              readOnly
              className="me-2"
            />
            <Button variant="secondary" onClick={handleBrowseFile}>
              Browse...
            </Button>
          </div>
          <Form.Text>Select the OpenBCI-RAW-*.txt file to process</Form.Text>
        </Form.Group>
      </div>

      {processingSuccess === true && (
        <Alert variant="success" className="mb-3">
          {processingMessage}
        </Alert>
      )}

      {processingSuccess === false && (
        <Alert variant="danger" className="mb-3">
          {processingMessage}
        </Alert>
      )}

      <Button
        variant="primary"
        size="lg"
        onClick={handleReturn}
        className="px-5 py-2"
        disabled={processing || !selectedEegFile}
      >
        {processing ? (
          <>
            <Spinner
              as="span"
              animation="border"
              size="sm"
              role="status"
              aria-hidden="true"
              className="me-2"
            />
            Processing EEG Data...
          </>
        ) : (
          'Process Data & Return'
        )}
      </Button>
    </div>
  );
};

export default CompletionView;
