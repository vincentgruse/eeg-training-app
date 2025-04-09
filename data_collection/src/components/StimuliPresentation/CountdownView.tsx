import React from 'react';
import { CountdownViewProps } from './types';

const CountdownView: React.FC<CountdownViewProps> = ({ value }) => (
  <div
    style={{
      width: '100%',
      height: '100%',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      backgroundColor: 'white',
    }}
  >
    <h1 style={{ fontSize: '10rem', margin: 0 }}>{value}</h1>
  </div>
);

export default CountdownView;
