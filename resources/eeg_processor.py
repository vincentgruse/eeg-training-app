import os
import pandas as pd
import numpy as np
import sys
import argparse
from pathlib import Path

# OpenBCI column names
COLUMN_NAMES = [
    'Sample Index', 'EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2',
    'EXG Channel 3', 'EXG Channel 4', 'EXG Channel 5', 'EXG Channel 6',
    'EXG Channel 7', 'EXG Channel 8', 'EXG Channel 9', 'EXG Channel 10',
    'EXG Channel 11', 'EXG Channel 12', 'EXG Channel 13', 'EXG Channel 14',
    'EXG Channel 15', 'Accel Channel 0', 'Accel Channel 1', 'Accel Channel 2',
    'Not Used 1', 'Digital Channel 0 (D11)', 'Digital Channel 1 (D12)',
    'Digital Channel 2 (D13)', 'Digital Channel 3 (D17)', 'Not Used 2',
    'Digital Channel 4 (D18)', 'Analog Channel 0', 'Analog Channel 1',
    'Analog Channel 2', 'Timestamp', 'Marker Channel', 'Timestamp (Formatted)'
]


def infer_delimiter(file_path):
    # Try to detect which delimiter is used in the file
    delimiters = ['\t', ',', ' ']
    delimiter_counts = {d: 0 for d in delimiters}

    try:
        with open(file_path, 'r') as f:
            lines = [f.readline() for _ in range(5)]

        for line in lines:
            if line:
                for d in delimiters:
                    delimiter_counts[d] += line.count(d)

        best_delimiter = max(delimiter_counts, key=delimiter_counts.get)

        if delimiter_counts[best_delimiter] < 5:
            return ','

        return best_delimiter
    except Exception:
        return ','


def read_stimulus_order(participant_info_file):
    # Extract stimulus directions from participant info file
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
    # Get participant ID from info file
    with open(participant_info_file, 'r') as f:
        for line in f:
            if line.startswith("Participant ID:"):
                return line.split(":")[1].strip()
    return "unknown"


def detect_triggers(analog_channel):
    # Find triggers in analog channel, looking for special patterns
    values = pd.to_numeric(analog_channel, errors='coerce')

    mean_val = values.mean()
    std_val = values.std()
    min_val = values.min()
    max_val = values.max()

    print(f"Analog Channel 1 stats - Min: {min_val}, Max: {max_val}, "
          f"Mean: {mean_val}, Std: {std_val}")

    diffs = values.diff().abs()
    threshold = 3 * diffs.std()
    print(f"Difference threshold for trigger detection: {threshold}")

    trigger_indices = np.where(diffs > threshold)[0]

    session_boundaries = identify_session_boundaries(
        values, trigger_indices, threshold)
    if session_boundaries:
        session_start_idx, session_end_idx = session_boundaries
        print(f"Detected session boundaries: Start at {session_start_idx}, "
              f"End at {session_end_idx}")

        session_triggers = [idx for idx in trigger_indices
                           if session_start_idx < idx < session_end_idx]

        min_spacing = 50
        filtered_indices = []

        if len(session_triggers) > 0:
            filtered_indices.append(session_triggers[0])
            for idx in session_triggers[1:]:
                if idx - filtered_indices[-1] >= min_spacing:
                    filtered_indices.append(idx)

        return filtered_indices

    print("Session boundaries not detected, falling back to standard trigger detection")

    min_spacing = 50
    filtered_indices = []

    if len(trigger_indices) > 0:
        filtered_indices.append(trigger_indices[0])
        for idx in trigger_indices[1:]:
            if idx - filtered_indices[-1] >= min_spacing:
                filtered_indices.append(idx)

    return filtered_indices


def identify_session_boundaries(values, trigger_indices, threshold):
    # Find session start (3 quick + 1 long flash) and end (1 long + 3 quick flash) patterns
    if len(trigger_indices) < 10:
        return None

    start_idx = None
    end_idx = None

    expected_interval_short = 30
    expected_interval_long = 75
    pattern_tolerance = 20

    # Look for start pattern
    for i in range(len(trigger_indices) - 8):
        interval_matches = True

        for j in range(6):
            if j % 2 == 0:
                if not (expected_interval_short - pattern_tolerance <
                        trigger_indices[i+j+1] - trigger_indices[i+j] <
                        expected_interval_short + pattern_tolerance):
                    interval_matches = False
                    break

        if interval_matches:
            if (expected_interval_long - pattern_tolerance <
                    trigger_indices[i+7] - trigger_indices[i+6] <
                    expected_interval_long + pattern_tolerance):
                start_idx = trigger_indices[i+8]
                print(f"Found start pattern at index {start_idx}")
                break

    # Look for end pattern
    if start_idx:
        mid_point = start_idx + (len(values) - start_idx) // 2
        mid_point_idx = min(range(len(trigger_indices)),
                           key=lambda i: abs(trigger_indices[i] - mid_point))

        for i in range(mid_point_idx, len(trigger_indices) - 8):
            if not (expected_interval_long - pattern_tolerance <
                    trigger_indices[i+1] - trigger_indices[i] <
                    expected_interval_long + pattern_tolerance):
                continue

            interval_matches = True
            for j in range(6):
                if not (expected_interval_short - pattern_tolerance <
                        trigger_indices[i+j+3] - trigger_indices[i+j+2] <
                        expected_interval_short + pattern_tolerance):
                    interval_matches = False
                    break

            if interval_matches:
                end_idx = trigger_indices[i]
                print(f"Found end pattern at index {end_idx}")
                break

    if start_idx and end_idx and start_idx < end_idx:
        return (start_idx, end_idx)
    else:
        return None


def get_stimulus_periods(trigger_on_indices, data_length):
    # Determine sample periods between triggers
    periods = []

    for i in range(len(trigger_on_indices)):
        start_idx = trigger_on_indices[i]

        if i == len(trigger_on_indices) - 1:
            end_idx = data_length
        else:
            end_idx = trigger_on_indices[i + 1]

        periods.append((start_idx, end_idx))

    return periods


def process_eeg_data(eeg_file_path, participant_info_file, output_dir='processed_data'):
    # Main function to process EEG data using Analog Channel 1 as trigger
    if not os.path.exists(eeg_file_path):
        print(f"ERROR: EEG data file does not exist: {eeg_file_path}")
        return False

    if not os.path.exists(participant_info_file):
        print(f"ERROR: Participant info file does not exist: {participant_info_file}")
        return False

    print(f"Processing EEG data from: {eeg_file_path}")
    print(f"Using participant info from: {participant_info_file}")

    base_dir = Path(output_dir)
    for direction in ['STOP', 'BACKWARD', 'FORWARD', 'RIGHT', 'LEFT']:
        os.makedirs(base_dir / direction, exist_ok=True)

    stimulus_order = read_stimulus_order(participant_info_file)
    print(f"Stimulus order: {stimulus_order}")

    stimulus_order = [s for s in stimulus_order if s.strip()]
    if not stimulus_order:
        print("ERROR: No valid stimulus directions found in participant info file")
        return False

    print(f"Filtered stimulus order: {stimulus_order}")

    delimiter = infer_delimiter(eeg_file_path)
    print(f"Inferred delimiter: '{delimiter}'")

    try:
        eeg_data = pd.read_csv(eeg_file_path, delimiter=delimiter, header=None)

        if len(eeg_data.columns) != len(COLUMN_NAMES):
            print(f"Warning: Found {len(eeg_data.columns)} columns "
                  f"but expected {len(COLUMN_NAMES)}")

            if len(eeg_data.columns) > len(COLUMN_NAMES):
                eeg_data = eeg_data.iloc[:, :len(COLUMN_NAMES)]
                print(f"Truncated to {len(eeg_data.columns)} columns")

            column_names_to_use = COLUMN_NAMES[:len(eeg_data.columns)]
        else:
            column_names_to_use = COLUMN_NAMES

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

            max_cols = max(len(row) for row in data_array)
            temp_column_names = [f"Col_{i}" for i in range(max_cols)]

            eeg_data = pd.DataFrame(data_array, columns=temp_column_names)
            print(f"Successfully loaded data with alternative method: "
                  f"{len(eeg_data)} rows, {len(eeg_data.columns)} columns")

            column_mapping = {}
            for i, col in enumerate(eeg_data.columns):
                if i < len(COLUMN_NAMES):
                    column_mapping[col] = COLUMN_NAMES[i]

            eeg_data = eeg_data.rename(columns=column_mapping)

        except Exception as e2:
            print(f"Alternative parsing also failed: {e2}")
            return False

    if 'Analog Channel 1' not in eeg_data.columns:
        analog_col_index = 28

        if analog_col_index < len(eeg_data.columns):
            print(f"'Analog Channel 1' not found in column names, using "
                  f"column at index {analog_col_index}")

            temp_col_name = f"Col_{analog_col_index}"

            if temp_col_name not in eeg_data.columns:
                eeg_data = eeg_data.rename(
                    columns={eeg_data.columns[analog_col_index]: temp_col_name})

            analog_channel_1 = eeg_data[temp_col_name]
        else:
            print("ERROR: Could not find 'Analog Channel 1' in the data "
                  "and no suitable column index exists")
            print("Available columns:", eeg_data.columns.tolist())
            return False
    else:
        analog_channel_1 = eeg_data['Analog Channel 1']

    print(f"Using Analog Channel 1 for trigger detection")
    print(f"First 5 values from Analog Channel 1: {analog_channel_1.head().tolist()}")

    trigger_on_indices = detect_triggers(analog_channel_1)

    if not trigger_on_indices or len(trigger_on_indices) < len(stimulus_order):
        print(f"Warning: Detected {len(trigger_on_indices)} triggers, "
              f"expected {len(stimulus_order)}")

        estimated_interval = len(eeg_data) // (len(stimulus_order) + 1)
        trigger_on_indices = [i * estimated_interval
                             for i in range(1, len(stimulus_order) + 1)]
        print(f"Using estimated intervals at samples: {trigger_on_indices}")
    else:
        print(f"Detected {len(trigger_on_indices)} trigger events")

    stimulus_periods = get_stimulus_periods(trigger_on_indices, len(eeg_data))

    if len(stimulus_periods) != len(stimulus_order):
        print(f"Warning: Number of detected stimulus periods ({len(stimulus_periods)}) "
              f"doesn't match expected number ({len(stimulus_order)})")
        num_periods = min(len(stimulus_periods), len(stimulus_order))
    else:
        num_periods = len(stimulus_periods)

    for i in range(num_periods):
        direction = stimulus_order[i]
        start_idx, end_idx = stimulus_periods[i]
        stimulus_data = eeg_data.iloc[start_idx:end_idx].copy()
        stimulus_data['Stimulus_Direction'] = direction
        stimulus_data['Stimulus_Number'] = i + 1

        participant_id = extract_participant_id(participant_info_file)
        output_file = base_dir / direction / f"participant_{participant_id}_{direction}_{i+1}.csv"
        stimulus_data.to_csv(output_file, index=False)
        print(f"Saved {direction} stimulus {i+1} data to {output_file}")

    print("Processing complete!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Process OpenBCI EEG data based on stimulus events")
    parser.add_argument("--eeg", required=True,
                       help="Path to EEG data file (.csv)")
    parser.add_argument("--participant", required=True,
                       help="Path to participant info file (.txt)")
    parser.add_argument("--output", default="processed_data",
                       help="Output directory")

    args = parser.parse_args()

    success = process_eeg_data(args.eeg, args.participant, args.output)

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)