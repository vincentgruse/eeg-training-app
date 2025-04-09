# EEG Training Data Collection Application

## Table of Contents

- [Overview](#overview)
- [Neuroscientific Basis](#neuroscientific-basis)
- [Features](#features)
- [System Architecture](#system-architecture)
  - [Electron Application Structure](#electron-application-structure)
  - [Key Components](#key-components)
- [EEG Data Collection Process](#eeg-data-collection-process)
  - [Experiment Flow](#experiment-flow)
  - [Visual Stimuli](#visual-stimuli)
  - [Trigger System](#trigger-system)
- [EEG Data Processing](#eeg-data-processing)
  - [Integration with OpenBCI](#integration-with-openbci)
  - [Trigger Detection Circuit](#trigger-detection-circuit)
  - [File Processing Pipeline](#file-processing-pipeline)
  - [Data File Structure](#data-file-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
  - [Data Collection Protocol](#data-collection-protocol)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
  - [EEG Processing Notes](#eeg-processing-notes)
- [Development](#development)
  - [Project Structure](#project-structure)
  - [Adding New Features](#adding-new-features)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This application is designed to collect EEG (Electroencephalogram) data for brain-computer interface development. It presents visual stimuli to participants and records their brain activity while they think about directional commands (forward, backward, left, right, or stop). The collected data will be used for transfer learning to analyze brain activity in real-time and control a small robot.

The application uses Electron for cross-platform desktop support and integrates with OpenBCI hardware for EEG data collection. The goal is to create a dataset that can be used to train machine learning models to interpret movement intentions directly from brain signals.

## Neuroscientific Basis

### Motor Imagery vs. Language Processing

The application is deliberately designed to engage brain regions associated with motor planning rather than language processing. This is why the stimuli are presented as visual scenarios requiring directional thinking, rather than using text instructions like "think about moving forward."

#### Key Neural Areas Targeted:

- **Primary Motor Cortex**: Located in the precentral gyrus, this area is involved in the execution of voluntary movements and is active during motor imagery (imagining movements without executing them).
  
- **Premotor Cortex and Supplementary Motor Area (SMA)**: These regions are crucial for motor planning and preparation, showing activity when participants are preparing to move or imagining movement.
  
- **Posterior Parietal Cortex**: Involved in spatial processing and sensorimotor integration, this area helps in planning spatially guided movements.

#### Neural Areas Avoided:

- **Wernicke's Area**: Located in the posterior section of the superior temporal gyrus, this area is involved in language comprehension and would be more active if text-based instructions were used.
  
- **Broca's Area**: Found in the inferior frontal gyrus, this region is involved in speech production and language processing.

### Why Visual Scenarios?

The intersection scenario with rewards in different directions encourages participants to:

1. **Engage in Spatial Thinking**: Activating the posterior parietal cortex and dorsolateral prefrontal cortex
   
2. **Imagine Movement**: Triggering activity in the premotor cortex and SMA even without actual physical movement
   
3. **Form Motor Intentions**: Creating distinct neural patterns in the motor planning regions that can be detected via EEG

These neural patterns are more distinct and consistent across individuals when elicited through spatial-motor imagery rather than through verbal or text-based instructions, making them better candidates for machine learning classification.

![Application screenshot](public\app_screenshot.png)

## Features

- **Visual Stimulus Presentation**: Displays a 4-way intersection with directional cues
- **Configurable Experiment Parameters**: Adjustable number of trials and timing
- **EEG Data Processing**: Integration with OpenBCI hardware and data processing
- **Automatic Data Synchronization**: Visual triggers for EEG data alignment
- **Data Organization**: Automated file management and organization by stimulus type

## System Architecture

### Electron Application Structure

The application is built using Electron, React, and TypeScript with the following components:

- **Main Process** (`main.ts`): Manages application lifecycle, window creation, and IPC handlers
- **Renderer Process** (`*.tsx` files): React components for the user interface
- **IPC Communication**: Handles data flow between UI and system operations
- **Python Integration**: Processes raw EEG data files using the external `eeg_processor.py` script

#### Why Electron?

Electron was chosen as the application framework for several important reasons:

1. **Cross-Platform Compatibility**: The application needs to run on Windows, macOS, and Linux environments often found in research settings.

2. **Native System Access**: Unlike web applications, Electron allows direct access to the file system for saving participant data and processing large EEG files.

3. **Python Integration**: The application requires running a Python script for sophisticated EEG data processing. Electron's ability to spawn child processes makes this integration seamless.

4. **Precise Timing Control**: EEG research requires precise stimulus timing. Electron provides more reliable timing control than web applications running in browsers.

5. **Offline Operation**: The application needs to function in laboratory environments that may have limited or no internet connectivity.

The separation between main and renderer processes enhances the application's stability, preventing UI freezes during computationally intensive operations like EEG file processing.

### Key Components

1. **Configuration Panel**: Set up participant information and experiment parameters
2. **Stimulus Display**: Present visual cues for directional commands
3. **EEG Data Processor**: Process raw EEG data and align with stimulus timing
4. **Data Export**: Save organized session data for later analysis

## EEG Data Collection Process

### Experiment Flow

1. **Setup**: Enter participant ID and configure stimulus parameters
2. **Instructions**: Participant reads instructions for the experiment
3. **Countdown**: 5-second countdown before experiment begins
4. **Stimulus Presentation**: Series of randomized directional stimuli
5. **Data Processing**: Select and process the OpenBCI EEG data file
6. **Completion**: Summary and preparation for next participant

### Visual Stimuli

The application displays an intersection scene with:
- A character (🧑‍🦲) in the center
- A reward (🏆) appearing in one of the four directions (or a stop sign ⛔)
- Visual movement animation that aligns with EEG data collection

#### Neuroscientific Design Considerations

The visual stimuli were carefully designed to optimize EEG signal quality and neural specificity:

1. **Character & Reward System**: This design leverages the brain's reward-oriented attention systems. The basal ganglia and dopaminergic pathways are activated when participants anticipate the character obtaining the reward, creating stronger neural signatures.

2. **Four-Way Intersection**: This layout was chosen to create distinct spatial representations in the right parietal lobe, which processes spatial relationships. The clear orthogonal directions (forward, backward, left, right) produce more distinguishable neural patterns than would arbitrary or diagonal directions.

3. **Stop Condition**: The stop sign (⛔) engages inhibitory control networks in the prefrontal cortex that are distinct from directional motor planning, providing a clear fifth neural state that can be detected in the EEG data.

4. **Animation Timing**: The delay before movement (configurable in the UI) allows collection of pre-movement EEG data that contains the neural signature of motor planning in the premotor cortex and supplementary motor area, which occurs before the visual feedback of motion.

5. **Clean Visual Design**: The simple, high-contrast visuals minimize visual processing load in the occipital lobe, reducing noise in the EEG signal that would otherwise obscure the motor planning signals of interest.

### Trigger System

The application uses a visual trigger system for synchronizing EEG data with stimuli:

- **Black Square Triggers**: A black square appears for 1 second in the bottom-right corner at specific timing events

#### Why Use Visual Triggers?

Visual triggers serve a critical purpose in EEG data collection:

1. **Hardware Synchronization**: The black square is detected by a photosensor connected to the OpenBCI's analog input channel, creating precise voltage spikes in the data stream.

2. **Temporal Precision**: EEG data is collected continuously at high sampling rates (250Hz), making it challenging to determine exactly when a stimulus was presented. Triggers provide precise temporal markers.

3. **Automated Processing**: The distinctive patterns allow the Python processing script to automatically identify session boundaries and individual trials without manual intervention.

4. **Non-invasive Integration**: This method doesn't require modifying the OpenBCI hardware or firmware, making it compatible with standard equipment.

## EEG Data Processing

### Integration with OpenBCI

The application is designed to work with OpenBCI hardware for EEG data collection. The raw data files (`.txt` or `.csv`) are processed after the experiment.

### Trigger Detection Circuit

A custom Light Dependent Resistor (LDR) circuit is used to capture the visual triggers from the screen:

#### Hardware Components

- LDR (Light Dependent Resistor)
- 220 ohm pull-up resistor
- Connection to OpenBCI analog input channel

#### Circuit Configuration

The LDR circuit connects to the OpenBCI board's analog input channel to detect the black square trigger displayed on screen.

![LDR schematic](public\LDR_schematic.png)

#### Circuit Operation

1. When the black square is not displayed (screen area is bright), the LDR has low resistance, causing the analog input to read a low voltage.
2. When the black square appears (screen area is dark), the LDR resistance increases, causing the analog input to read a higher voltage through the pull-up resistor.
3. This voltage change creates distinct spikes in the analog channel that are detected by the EEG processing script.

#### Physical Setup

The LDR should be attached to the bottom-right corner of the screen (where the trigger square appears) and covered with shrink tubing or another light shield to prevent ambient light interference.

![LDR with shield](public/LDR.png)

#### Why This Approach?

The LDR circuit was chosen for several reasons:

1. **Non-invasive**: Requires no modification to the OpenBCI hardware
2. **Low cost**: Uses inexpensive, widely available components
3. **Reliable detection**: Creates clear voltage spikes that are easily distinguishable from background noise
4. **No electrical connection**: Maintains electrical isolation between the computer and the EEG equipment

### File Processing Pipeline

1. **Data Collection**: OpenBCI records raw EEG data during the experiment
2. **File Selection**: User selects the OpenBCI data file after experiment completion
3. **Trigger Detection**: Python script identifies triggers in the analog channel
4. **Data Segmentation**: EEG data is segmented based on stimulus events
5. **Organization**: Processed data is organized by direction (LEFT, RIGHT, FORWARD, BACKWARD, STOP)

### Data File Structure

#### Participant Info Files

The application generates a participant info file with the following structure:

```
Participant ID: [ID]
Session Start: [ISO timestamp]
Session End: [ISO timestamp]
Total Stimuli: [count]
Session Duration: [seconds]

Trigger Pattern Information:
- Session Start Pattern: 3 quick flashes (150ms) followed by 1 long flash (450ms)
- Session End Pattern: 1 long flash (450ms) followed by 3 quick flashes (150ms)
- Emergency Stop Pattern: 5 equal flashes (100ms on, 100ms off)
- Normal Stimulus Trigger: Single flash (500ms)

Stimulus Order:
Stimulus 1: [DIRECTION]
Stimulus 2: [DIRECTION]
...

Trial Timing Data:
Trial 1: Direction=[direction], Start=[timestamp], End=[timestamp], Duration=[seconds]
...
```

#### Processed EEG Data

The Python processor organizes data into directories by direction:
```
processed_data/
├── FORWARD/
│   └── participant_001_FORWARD_1.csv
│   └── participant_001_FORWARD_2.csv
├── BACKWARD/
│   └── participant_001_BACKWARD_1.csv
├── LEFT/
│   └── participant_001_LEFT_1.csv
├── RIGHT/
│   └── participant_001_RIGHT_1.csv
└── STOP/
    └── participant_001_STOP_1.csv
```

Each CSV file contains the EEG data for a specific stimulus with added columns for direction and trial number.

## Getting Started

### Prerequisites

- Node.js and npm
- Python 3.6+
- OpenBCI hardware and software

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/eeg-training-app.git
cd eeg-training-app
```

2. Install dependencies
```bash
npm install
```

3. Install Python dependencies
```bash
pip install pandas numpy
```

### Running the Application

1. Start the development server:
```bash
npm run dev
```

2. Build for production:
```bash
npm run build
```

### Data Collection Protocol

1. Connect the OpenBCI headset according to the manufacturer's instructions
2. Start recording in the OpenBCI software
3. Launch the EEG Training Application
4. Enter participant information and configure trial settings
5. Run through the stimulus presentation
6. After completion, select the recorded EEG file for processing
7. Check the `processed_data` directory for the organized EEG data files

## Troubleshooting

### Common Issues

- **File Not Found Errors**: Ensure the OpenBCI file is properly saved and accessible
- **Trigger Detection Issues**: Check that the analog channel is properly connected
- **Application Crashes**: Verify that all dependencies are installed correctly

### EEG Processing Notes

- The processor automatically attempts to detect the delimiter in the OpenBCI file
- It uses the Analog Channel 1 for trigger detection
- If the expected number of triggers isn't detected, it will estimate trigger locations

#### Why These Processing Decisions Matter

1. **Automatic Delimiter Detection**: OpenBCI software can export data with different delimiters (commas, tabs, spaces) depending on the version and export settings. Automatic detection increases robustness across different file formats.

2. **Analog Channel Usage**: While OpenBCI has digital trigger inputs, using the analog channel provides better temporal resolution and allows for capturing varying trigger intensities, not just on/off states.

3. **Fallback Estimation**: The fallback mechanism for estimating trigger positions is crucial for salvaging data from sessions where some triggers might have been missed due to hardware issues. This prevents complete data loss when minor hardware problems occur.

4. **Data Segmentation Strategy**: Organizing data by direction (LEFT, RIGHT, etc.) facilitates machine learning training by allowing the model to learn distinct patterns associated with each directional intention.

5. **Participant-Centric File Naming**: Including participant IDs in filenames allows for both individual analysis and aggregation across participants, supporting both personalized and generalized model development.

## Development

### Project Structure

```
eeg-training-app/
├── electron/
│   ├── main.ts                  # Main Electron process
│   ├── preload.ts               # Preload script for IPC
│   ├── eeg-processor-integration.ts  # EEG processor integration
│   └── ipc.ts                   # IPC handlers
├── src/
│   ├── StimulusDisplay.tsx      # Main stimulus presentation component
│   ├── ConfigurationPanel.tsx   # Experiment configuration UI
│   ├── IntersectionView.tsx     # Visual intersection display
│   ├── CompletionView.tsx       # Session completion screen
│   └── ...                      # Other React components
├── resources/
│   └── eeg_processor.py         # Python EEG data processing script
└── data/                        # Directory for participant data files
```

#### Architecture Design Rationale

The project structure follows specific design principles for EEG research applications:

1. **Separation of Concerns**: 
   - **Electron/** contains all system-level operations, isolating them from the UI
   - **src/** contains purely UI components, making them testable in isolation
   - **resources/** contains the Python processing script, kept separate from the JavaScript codebase

2. **Component-Based Design**: Each UI element is a separate component with clear responsibilities, allowing for easier maintenance and extension for future experiments.

3. **Data Flow Architecture**: The application follows a unidirectional data flow pattern, where:
   - Configuration settings flow down from parent components
   - Events flow up through callbacks
   - This prevents state synchronization issues that could affect timing precision

4. **Security Considerations**: The preload script creates a controlled bridge between renderer and main processes to prevent unauthorized access to system resources.

### Adding New Features

- **New Stimulus Types**: Modify the `DIRECTIONS` constant in `constants.ts`
- **Additional EEG Analysis**: Extend the Python processing script
- **UI Customization**: Update the React components in the `src` directory

#### Transfer Learning Pipeline

This application is part of a larger transfer learning pipeline for brain-computer interfaces:

1. **Data Collection (This App)**: Collects structured, labeled EEG data samples
2. **Preprocessing**: The collected data undergoes additional cleaning and feature extraction
3. **Model Training**: A base model is trained on the larger dataset
4. **Transfer Learning**: The base model is fine-tuned with individual participant data
5. **Real-time Control**: The fine-tuned model is deployed for real-time robot control

When extending this application, consider how changes will affect downstream components in this pipeline. For example:

- Adding new directions would require retraining the base model
- Changing the EEG processing parameters could invalidate compatibility with existing models
- Modifications to the visual stimuli might alter the neural patterns being detected

#### Adding New EEG Analysis Features

When extending the Python EEG processor, consider implementing:

1. **Additional Frequency Band Analysis**: Add extraction of standard EEG bands (alpha, beta, theta, delta, gamma)
2. **Spatial Filtering**: Implement Common Spatial Pattern (CSP) filtering for better signal isolation
3. **Artifact Rejection**: Enhance the automatic detection and removal of eye blinks and muscle artifacts

## Contributing

Contributions to improve the application are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the **MIT License**:

```
MIT License

Copyright (c) [Year] [Your Name/Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Acknowledgments

- OpenBCI for EEG hardware and GUI software
- Electron and React for application framework
