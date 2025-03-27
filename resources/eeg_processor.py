import os
import pandas as pd
import numpy as np
import sys
import argparse
from pathlib import Path

# Define column names explicitly - make sure there are no duplicates
COLUMN_NAMES = [
    'Sample Index', 'EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3', 
    'EXG Channel 4', 'EXG Channel 5', 'EXG Channel 6', 'EXG Channel 7', 'EXG Channel 8', 
    'EXG Channel 9', 'EXG Channel 10', 'EXG Channel 11', 'EXG Channel 12', 'EXG Channel 13', 
    'EXG Channel 14', 'EXG Channel 15', 'Accel Channel 0', 'Accel Channel 1', 'Accel Channel 2', 
    'Not Used 1', 'Digital Channel 0 (D11)', 'Digital Channel 1 (D12)', 'Digital Channel 2 (D13)', 
    'Digital Channel 3 (D17)', 'Not Used 2', 'Digital Channel 4 (D18)', 'Analog Channel 0', 
    'Analog Channel 1', 'Analog Channel 2', 'Timestamp', 'Marker Channel', 'Timestamp (Formatted)'
]

def process_eeg_data(eeg_file_path, participant_info_file, output_dir='processed_data'):
    """
    Process EEG data and separate it based on stimulus events triggered by the photoresistor.
    Using Analog Channel 1 as the trigger.
    """
    # Check if files exist
    if not os.path.exists(eeg_file_path):
        print(f"ERROR: EEG data file does not exist: {eeg_file_path}")
        return False
        
    if not os.path.exists(participant_info_file):
        print(f"ERROR: Participant info file does not exist: {participant_info_file}")
        return False
            
    print(f"Processing EEG data from: {eeg_file_path}")
    print(f"Using participant info from: {participant_info_file}")
    
    # Create output directory structure
    base_dir = Path(output_dir)
    for direction in ['STOP', 'BACKWARD', 'FORWARD', 'RIGHT', 'LEFT']:
        os.makedirs(base_dir / direction, exist_ok=True)
    
    # Read stimulus order from participant info file
    stimulus_order = read_stimulus_order(participant_info_file)
    print(f"Stimulus order: {stimulus_order}")
    
    # Filter out any empty stimulus entries
    stimulus_order = [s for s in stimulus_order if s.strip()]
    if not stimulus_order:
        print("ERROR: No valid stimulus directions found in participant info file")
        return False
        
    print(f"Filtered stimulus order: {stimulus_order}")
    
    # Try to infer the file format by examining the file
    delimiter = infer_delimiter(eeg_file_path)
    print(f"Inferred delimiter: '{delimiter}'")
    
    # Read the CSV file
    try:
        # Try reading with the inferred delimiter
        eeg_data = pd.read_csv(eeg_file_path, delimiter=delimiter, header=None)
        
        # Check if we have the right number of columns
        if len(eeg_data.columns) != len(COLUMN_NAMES):
            print(f"Warning: Found {len(eeg_data.columns)} columns but expected {len(COLUMN_NAMES)}")
            
            # If we have too many columns, truncate them
            if len(eeg_data.columns) > len(COLUMN_NAMES):
                eeg_data = eeg_data.iloc[:, :len(COLUMN_NAMES)]
                print(f"Truncated to {len(eeg_data.columns)} columns")
            
            # If we have too few columns, we'll just name the ones we have
            column_names_to_use = COLUMN_NAMES[:len(eeg_data.columns)]
        else:
            column_names_to_use = COLUMN_NAMES
        
        # Rename the columns
        eeg_data.columns = column_names_to_use
        
        print(f"Successfully loaded data: {len(eeg_data)} rows")
    except Exception as e:
        print(f"Error reading EEG data: {e}")
        
        try:
            print("Trying alternative parsing approach...")
            data_array = []
            
            with open(eeg_file_path, 'r') as f:
                for line in f:
                    values = line.strip().split(delimiter)
                    # Convert string values to numeric where possible
                    processed_values = []
                    for val in values:
                        try:
                            processed_values.append(float(val))
                        except ValueError:
                            processed_values.append(val)
                    data_array.append(processed_values)
            
            if not data_array:
                print("No data found in file")
                return False
                
            # Create DataFrame with maximum number of columns
            max_cols = max(len(row) for row in data_array)
            # Only use as many column names as we have columns
            temp_column_names = [f"Col_{i}" for i in range(max_cols)]
            
            eeg_data = pd.DataFrame(data_array, columns=temp_column_names)
            print(f"Successfully loaded data with alternative method: {len(eeg_data)} rows, {len(eeg_data.columns)} columns")
            
            # Now map to our expected column names
            column_mapping = {}
            for i, col in enumerate(eeg_data.columns):
                if i < len(COLUMN_NAMES):
                    column_mapping[col] = COLUMN_NAMES[i]
            
            eeg_data = eeg_data.rename(columns=column_mapping)
            
        except Exception as e2:
            print(f"Alternative parsing also failed: {e2}")
            return False
    
    # Confirm we have the Analog Channel 1 column
    if 'Analog Channel 1' not in eeg_data.columns:
        # Try to find the column index that would correspond to Analog Channel 1
        analog_col_index = 28  # This is the typical index for Analog Channel 1 based on your column list
        
        if analog_col_index < len(eeg_data.columns):
            print(f"'Analog Channel 1' not found in column names, using column at index {analog_col_index}")
            
            # Create a temporary name for the column
            temp_col_name = f"Col_{analog_col_index}"
            
            # If the column doesn't already have this name, rename it
            if temp_col_name not in eeg_data.columns:
                eeg_data = eeg_data.rename(columns={eeg_data.columns[analog_col_index]: temp_col_name})
            
            # Now we can use this column for trigger detection
            analog_channel_1 = eeg_data[temp_col_name]
        else:
            print("ERROR: Could not find 'Analog Channel 1' in the data and no suitable column index exists")
            print("Available columns:", eeg_data.columns.tolist())
            return False
    else:
        analog_channel_1 = eeg_data['Analog Channel 1']
    
    print(f"Using Analog Channel 1 for trigger detection")
    
    # Show some sample values
    print(f"First 5 values from Analog Channel 1: {analog_channel_1.head().tolist()}")
    
    # Detect trigger events
    trigger_on_indices = detect_triggers(analog_channel_1)
    
    if not trigger_on_indices or len(trigger_on_indices) < len(stimulus_order):
        print(f"Warning: Detected {len(trigger_on_indices)} triggers, expected {len(stimulus_order)}")
        
        # If not enough triggers detected, use fixed intervals
        estimated_interval = len(eeg_data) // (len(stimulus_order) + 1)
        trigger_on_indices = [i * estimated_interval for i in range(1, len(stimulus_order) + 1)]
        print(f"Using estimated intervals at samples: {trigger_on_indices}")
    else:
        print(f"Detected {len(trigger_on_indices)} trigger events")
    
    # Get stimulus periods
    stimulus_periods = get_stimulus_periods(trigger_on_indices, len(eeg_data))
    
    # Verify we have the right number of periods for the stimulus order
    if len(stimulus_periods) != len(stimulus_order):
        print(f"Warning: Number of detected stimulus periods ({len(stimulus_periods)}) doesn't match expected number ({len(stimulus_order)})")
        # Use the minimum of the two to avoid index errors
        num_periods = min(len(stimulus_periods), len(stimulus_order))
    else:
        num_periods = len(stimulus_periods)
    
    # Process data for each stimulus period
    for i in range(num_periods):
        # Get the stimulus direction from the order
        direction = stimulus_order[i]
        
        # Get the start and end indices for this stimulus period
        start_idx, end_idx = stimulus_periods[i]
        
        # Extract data for this stimulus period
        stimulus_data = eeg_data.iloc[start_idx:end_idx].copy()
        
        # Add a column indicating which stimulus this is
        stimulus_data['Stimulus_Direction'] = direction
        stimulus_data['Stimulus_Number'] = i + 1
        
        # Save data for this stimulus
        participant_id = extract_participant_id(participant_info_file)
        output_file = base_dir / direction / f"participant_{participant_id}_{direction}_{i+1}.csv"
        stimulus_data.to_csv(output_file, index=False)
        print(f"Saved {direction} stimulus {i+1} data to {output_file}")
    
    print("Processing complete!")
    return True

def infer_delimiter(file_path):
    """
    Try to infer the delimiter used in the file by examining the first few lines.
    """
    # Common delimiters to check
    delimiters = ['\t', ',', ' ']
    delimiter_counts = {d: 0 for d in delimiters}
    
    # Read the first few lines of the file
    try:
        with open(file_path, 'r') as f:
            lines = [f.readline() for _ in range(5)]
        
        # Count occurrences of each delimiter in the first few lines
        for line in lines:
            if line:
                for d in delimiters:
                    delimiter_counts[d] += line.count(d)
        
        # Find the delimiter with the highest count
        best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
        
        # If the best delimiter isn't found much, default to comma
        if delimiter_counts[best_delimiter] < 5:
            return ','
        
        return best_delimiter
    except Exception:
        # If anything goes wrong, default to comma
        return ','

def read_stimulus_order(participant_info_file):
    """Read stimulus order from participant info file."""
    with open(participant_info_file, 'r') as f:
        lines = f.readlines()
    
    stimulus_order = []
    for line in lines:
        if line.strip().startswith("Stimulus"):
            parts = line.strip().split(":")
            if len(parts) == 2:
                stimulus_order.append(parts[1].strip())
    
    return stimulus_order

def extract_participant_id(participant_info_file):
    """Extract the participant ID from the info file"""
    with open(participant_info_file, 'r') as f:
        for line in f:
            if line.startswith("Participant ID:"):
                return line.split(":")[1].strip()
    return "unknown"

def detect_triggers(analog_channel):
    """
    Detect triggers based on significant changes in the analog channel.
    """
    # Convert to numeric values in case they're stored as strings
    values = pd.to_numeric(analog_channel, errors='coerce')
    
    # Get basic statistics of the values
    mean_val = values.mean()
    std_val = values.std()
    min_val = values.min()
    max_val = values.max()
    
    print(f"Analog Channel 1 stats - Min: {min_val}, Max: {max_val}, Mean: {mean_val}, Std: {std_val}")
    
    # Calculate absolute differences between consecutive values
    diffs = values.diff().abs()
    
    # Detect significant jumps (3 standard deviations above mean difference)
    threshold = 3 * diffs.std()
    print(f"Difference threshold for trigger detection: {threshold}")
    
    # Find indices where differences exceed threshold
    trigger_indices = np.where(diffs > threshold)[0]
    
    # Ensure minimum spacing between triggers
    min_spacing = 50  # Minimum samples between triggers
    filtered_indices = []
    
    if len(trigger_indices) > 0:
        filtered_indices.append(trigger_indices[0])
        for idx in trigger_indices[1:]:
            if idx - filtered_indices[-1] >= min_spacing:
                filtered_indices.append(idx)
    
    return filtered_indices

def get_stimulus_periods(trigger_on_indices, data_length):
    """Determine the periods between consecutive triggers."""
    periods = []
    
    # Each stimulus period starts at a trigger and ends before the next trigger
    for i in range(len(trigger_on_indices)):
        start_idx = trigger_on_indices[i]
        
        # If this is the last trigger, the period ends at the end of the data
        if i == len(trigger_on_indices) - 1:
            end_idx = data_length
        else:
            # Otherwise, it ends just before the next trigger
            end_idx = trigger_on_indices[i + 1]
        
        periods.append((start_idx, end_idx))
    
    return periods

def main():
    """Main function to run the EEG data processing script."""
    parser = argparse.ArgumentParser(description="Process OpenBCI EEG data based on stimulus events")
    parser.add_argument("--eeg", required=True, help="Path to EEG data file (.csv)")
    parser.add_argument("--participant", required=True, help="Path to participant info file (.txt)")
    parser.add_argument("--output", default="processed_data", help="Output directory")
    
    args = parser.parse_args()
    
    # Process the data with the provided file paths
    success = process_eeg_data(args.eeg, args.participant, args.output)
    
    return success

if __name__ == "__main__":
    success = main()
    # Return exit code (0 for success, 1 for failure)
    sys.exit(0 if success else 1)