import React from 'react';
import { Button } from 'react-bootstrap';
import { CompletionViewProps } from './types';

const CompletionView: React.FC<CompletionViewProps> = ({
  onRestart,
  totalTrials,
}) => {
  const handleReturn = () => {
    onRestart();
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
        Your data has been saved successfully.
      </p>

      <Button
        variant="primary"
        size="lg"
        onClick={handleReturn}
        className="px-5 py-3"
      >
        Return to Start
      </Button>
    </div>
  );
};

export default CompletionView;
