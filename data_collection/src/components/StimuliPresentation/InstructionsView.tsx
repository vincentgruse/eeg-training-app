import React from 'react';
import { InstructionsViewProps } from './types';

const InstructionsView: React.FC<InstructionsViewProps> = () => (
  <div
    style={{
      width: '80%',
      height: '80%',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center',
      backgroundColor: 'white',
      borderRadius: '5px',
      padding: '2rem',
    }}
  >
    <div className="text-center">
      <h2>Intersection Stimulus Presentation</h2>
      <p>
        You will be shown a 4-way intersection with rewards appearing in
        different directions.
      </p>
      <p>
        Please focus on the center of the screen where the character will
        appear.
      </p>
      <p>
        Concentrate hard, and think about the direction the character needs to
        move to obtain the reward.
      </p>
      <p>
        <strong>Important:</strong> Sometimes the reward will change to a stop
        sign (⛔). When this happens, you need to think STOP. You do NOT want
        the character to collect the ⛔.
      </p>
      <p>The presentation will begin after a 5-second countdown.</p>
    </div>
  </div>
);

export default InstructionsView;
