import os
import pandas as pd
import numpy as np
import sys
import tkinter as tk
import argparse
from tkinter import filedialog, messagebox
from pathlib import Path

def read_stimulus_order(participant_info_file):
    with open(participant_info_file, 'r') as f:
        lines = f.readlines()

    stimulus_order = []
    for line in lines:
        if line.strip().startswith("Stimulus"):
            parts = line.strip().split(":")
            if len(parts) == 2:
                stimulus_order.append(parts[1].strip())

    return [s for s in stimulus_order if s.strip()]

def extract_participant_id(participant_info_file):
    with open(participant_info_file, 'r') as f:
        for line in f:
            if line.startswith("Participant ID:"):
                return line.split(":")[1].strip()
    return "unknown"

def parse_openbci_file(file_path):
    print(f"Parsing OpenBCI file: {file_path}")
    
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines (lines starting with %)
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('%'):
            data_start = i + 1
        else:
            break
    
    # Get column names from next line
    header_line = lines[data_start].strip()
    column_names = [col.strip() for col in header_line.split(',')]
    print(f"Found {len(column_names)} columns")
    
    # Parse data rows
    data = []
    for i in range(data_start + 1, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
            
        # Split by comma
        values = [val.strip() for val in line.split(',')]
        
        # Convert numeric values
        processed_values = []
        for val in values:
            try:
                processed_values.append(float(val))
            except ValueError:
                processed_values.append(val)
        
        # Ensure correct length
        if len(processed_values) < len(column_names):
            processed_values.extend([None] * (len(column_names) - len(processed_values)))
        elif len(processed_values) > len(column_names):
            processed_values = processed_values[:len(column_names)]
            
        data.append(processed_values)
    
    df = pd.DataFrame(data, columns=column_names)
    print(f"Successfully loaded {len(df)} rows of data")
    return df

def detect_triggers_by_value(analog_values, low_threshold=10, high_threshold=40, min_spacing=250):
    # Convert to numeric if needed
    values = pd.to_numeric(analog_values, errors='coerce')
    
    # Get statistics
    mean_val = values.mean()
    std_val = values.std()
    min_val = values.min()
    max_val = values.max()
    
    print(f"Analog Channel 1 stats - Min: {min_val:.2f}, Max: {max_val:.2f}, Mean: {mean_val:.2f}, Std: {std_val:.2f}")
    
    # Print histogram to understand the distribution better
    value_counts = values.value_counts().sort_index()
    print("Value distribution (first 10 most common values):")
    for val, count in value_counts.head(10).items():
        print(f"  {val:.2f}: {count} occurrences")
    
    # Find the first index where the value is HIGH (above high_threshold)
    # This ensures we're in a stable state before looking for triggers
    high_indices = np.where(values > high_threshold)[0]
    
    # If we found any high values, start from the first one
    start_idx = 0
    if len(high_indices) > 0:
        start_idx = high_indices[0]
        print(f"Found first HIGH value (>{high_threshold}) at index {start_idx}, starting detection from here")
    else:
        print(f"Warning: No values above {high_threshold} found, starting from beginning")
    
    # Now find LOW values (triggers) only AFTER seeing the first HIGH value
    # (when the trigger square is ON, analog values are LOW)
    low_value_indices = np.where(values.iloc[start_idx:] <= low_threshold)[0] + start_idx
    print(f"Found {len(low_value_indices)} samples with values <= {low_threshold} after first HIGH value")
    
    # Find contiguous blocks of low values
    trigger_blocks = []
    if len(low_value_indices) > 0:
        block_start = low_value_indices[0]
        prev_idx = low_value_indices[0]
        
        for idx in low_value_indices[1:]:
            # If this sample is not consecutive with the previous one
            if idx > prev_idx + 1:
                # Add the previous block
                trigger_blocks.append((block_start, prev_idx))
                # Start a new block
                block_start = idx
            prev_idx = idx
            
        # Add the final block
        trigger_blocks.append((block_start, prev_idx))
    
    print(f"Found {len(trigger_blocks)} contiguous blocks of low values")
    
    # Get the start of each trigger block (when the value first drops)
    trigger_indices = [block[0] for block in trigger_blocks]
    
    # Filter to ensure minimum spacing
    filtered_indices = []
    if len(trigger_indices) > 0:
        filtered_indices.append(trigger_indices[0])
        for idx in trigger_indices[1:]:
            if idx - filtered_indices[-1] >= min_spacing:
                filtered_indices.append(idx)
    
    print(f"Detected {len(filtered_indices)} triggers at indices: {filtered_indices}")
    
    # If we have at least one trigger, print the analog values at those points
    if filtered_indices:
        print("Trigger values at detected indices:")
        for idx in filtered_indices[:5]:  # Show first 5 triggers
            # Show values before, at, and after trigger
            before_idx = max(0, idx-5)
            after_idx = min(len(values)-1, idx+5)
            print(f"  Index {idx}: Values around trigger: {values.iloc[before_idx:after_idx+1].tolist()}")
    
    return filtered_indices

def process_eeg_data(eeg_file_path, participant_info_file, output_dir='processed_data', low_threshold=10, high_threshold=40):
    # Check if files exist
    if not os.path.exists(eeg_file_path):
        print(f"ERROR: EEG data file does not exist: {eeg_file_path}")
        return False
    if not os.path.exists(participant_info_file):
        print(f"ERROR: Participant info file does not exist: {participant_info_file}")
        return False

    # Create output directories
    base_dir = Path(output_dir)
    for direction in ['STOP', 'BACKWARD', 'FORWARD', 'RIGHT', 'LEFT']:
        os.makedirs(base_dir / direction, exist_ok=True)

    # Read stimulus order
    stimulus_order = read_stimulus_order(participant_info_file)
    if not stimulus_order:
        print("ERROR: No valid stimulus directions found")
        return False
    print(f"Stimulus order: {stimulus_order}")

    # Parse the EEG data file
    eeg_data = parse_openbci_file(eeg_file_path)
    
    # Get Analog Channel 1 values (at column index 20)
    analog_col_name = 'Analog Channel 1'
    if analog_col_name in eeg_data.columns:
        analog_channel = eeg_data[analog_col_name]
        print(f"Found '{analog_col_name}' column")
    else:
        # Use column index 20
        analog_channel = eeg_data.iloc[:, 20]
        print(f"Using column at index 20 for trigger detection")
    
    # Detect trigger points
    trigger_indices = detect_triggers_by_value(analog_channel, low_threshold, high_threshold)
    
    # If not enough triggers detected, use evenly spaced intervals
    if len(trigger_indices) < len(stimulus_order):
        print(f"Warning: Detected {len(trigger_indices)} triggers, expected {len(stimulus_order)}")
        estimated_interval = len(eeg_data) // (len(stimulus_order) + 1)
        trigger_indices = [i * estimated_interval for i in range(1, len(stimulus_order) + 1)]
        print(f"Using estimated intervals at samples: {trigger_indices}")
    elif len(trigger_indices) > len(stimulus_order):
        print(f"Warning: Detected {len(trigger_indices)} triggers, expected {len(stimulus_order)}")
        print(f"Using only the first {len(stimulus_order)} triggers")
        trigger_indices = trigger_indices[:len(stimulus_order)]
    
    # Create periods between triggers
    stimulus_periods = []
    for i in range(len(trigger_indices)):
        start_idx = trigger_indices[i]
        if i == len(trigger_indices) - 1:
            end_idx = len(eeg_data)
        else:
            end_idx = trigger_indices[i + 1]
        stimulus_periods.append((start_idx, end_idx))
    
    # Save segmented data
    num_periods = min(len(stimulus_periods), len(stimulus_order))
    participant_id = extract_participant_id(participant_info_file)
    
    for i in range(num_periods):
        direction = stimulus_order[i]
        start_idx, end_idx = stimulus_periods[i]
        stimulus_data = eeg_data.iloc[start_idx:end_idx].copy()
        stimulus_data['Stimulus_Direction'] = direction
        stimulus_data['Stimulus_Number'] = i + 1

        output_file = base_dir / direction / f"participant_{participant_id}_{direction}_{i+1}.csv"
        stimulus_data.to_csv(output_file, index=False)
        print(f"Saved {direction} stimulus {i+1} data to {output_file}")

    print("Processing complete!")
    return True


class EEGProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Data Processor")
        self.root.geometry("650x500")
        
        # Variables to store file paths and settings
        self.eeg_file_path = tk.StringVar()
        self.participant_file_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="processed_data")
        
        # Output text widget for logging
        self.output_frame = tk.Frame(root)
        self.output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.output_text = tk.Text(self.output_frame, height=20, width=75)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.scrollbar = tk.Scrollbar(self.output_frame, command=self.output_text.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=self.scrollbar.set)
        
        # Redirect stdout to the text widget
        self.stdout_original = sys.stdout
        sys.stdout = self
        
        # File selection frame
        self.file_frame = tk.Frame(root)
        self.file_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # EEG File
        tk.Label(self.file_frame, text="EEG Data File:").grid(row=0, column=0, sticky="w", pady=5)
        tk.Entry(self.file_frame, textvariable=self.eeg_file_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(self.file_frame, text="Browse...", command=self.browse_eeg_file).grid(row=0, column=2, pady=5)
        
        # Participant File
        tk.Label(self.file_frame, text="Participant Info:").grid(row=1, column=0, sticky="w", pady=5)
        tk.Entry(self.file_frame, textvariable=self.participant_file_path, width=50).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(self.file_frame, text="Browse...", command=self.browse_participant_file).grid(row=1, column=2, pady=5)
        
        # Output Directory
        tk.Label(self.file_frame, text="Output Directory:").grid(row=2, column=0, sticky="w", pady=5)
        tk.Entry(self.file_frame, textvariable=self.output_dir, width=50).grid(row=2, column=1, padx=5, pady=5)
        tk.Button(self.file_frame, text="Browse...", command=self.browse_output_dir).grid(row=2, column=2, pady=5)
        
        # Threshold settings
        tk.Label(self.file_frame, text="Low Value Threshold:").grid(row=3, column=0, sticky="w", pady=5)
        self.low_threshold_var = tk.StringVar(value="10")
        tk.Entry(self.file_frame, textvariable=self.low_threshold_var, width=10).grid(row=3, column=1, sticky="w", padx=5, pady=5)
        tk.Label(self.file_frame, text="(Values below this are considered trigger ON)").grid(row=3, column=1, sticky="e", padx=5, pady=5)
        
        tk.Label(self.file_frame, text="High Value Threshold:").grid(row=4, column=0, sticky="w", pady=5)
        self.high_threshold_var = tk.StringVar(value="40")
        tk.Entry(self.file_frame, textvariable=self.high_threshold_var, width=10).grid(row=4, column=1, sticky="w", padx=5, pady=5)
        tk.Label(self.file_frame, text="(Waits for value above this before detecting)").grid(row=4, column=1, sticky="e", padx=5, pady=5)
        
        # Button frame
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)
        
        # Process button
        self.process_button = tk.Button(button_frame, text="Process EEG Data", command=self.process_data, width=20, height=2)
        self.process_button.pack()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.log("EEG Data Processor started. Please select files and click 'Process EEG Data'.")
        self.log("NOTE: For OpenBCI data with high values (~50) and low trigger values (~3-10), use low threshold 10 and high threshold 40.")
    
    def write(self, text):
        # For redirecting stdout to text widget
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.root.update_idletasks()
    
    def flush(self):
        # Required for redirect
        pass
    
    def log(self, message):
        # Add message to log
        self.write(message + "\n")
    
    def browse_eeg_file(self):
        filename = filedialog.askopenfilename(
            title="Select EEG Data File",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.eeg_file_path.set(filename)
            self.log(f"Selected EEG data file: {filename}")
    
    def browse_participant_file(self):
        filename = filedialog.askopenfilename(
            title="Select Participant Info File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            self.participant_file_path.set(filename)
            self.log(f"Selected participant info file: {filename}")
    
    def browse_output_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)
            self.log(f"Selected output directory: {directory}")
    
    def process_data(self):
        eeg_file = self.eeg_file_path.get()
        participant_file = self.participant_file_path.get()
        output_dir = self.output_dir.get()
        
        if not eeg_file or not participant_file:
            messagebox.showerror("Error", "Please select both EEG data and participant info files.")
            return
        
        # Get custom thresholds
        try:
            low_threshold = float(self.low_threshold_var.get())
            high_threshold = float(self.high_threshold_var.get())
            
            if low_threshold <= 0:
                messagebox.showerror("Error", "Low threshold value must be positive.")
                return
                
            if high_threshold <= low_threshold:
                messagebox.showerror("Error", "High threshold must be greater than low threshold.")
                return
                
        except ValueError:
            messagebox.showerror("Error", "Invalid threshold values. Please enter numbers.")
            return
        
        self.status_var.set("Processing...")
        self.process_button.config(state=tk.DISABLED)
        self.root.update_idletasks()
        
        try:
            self.log(f"Using value-based trigger detection with low threshold {low_threshold} and high threshold {high_threshold}")
            success = process_eeg_data(eeg_file, participant_file, output_dir, low_threshold, high_threshold)
            
            if success:
                self.status_var.set("Processing completed successfully!")
                messagebox.showinfo("Success", f"EEG data has been processed successfully!\nOutput saved to: {output_dir}")
            else:
                self.status_var.set("Processing failed")
                messagebox.showerror("Error", "Processing failed. Check the log for details.")
                
        except Exception as e:
            self.status_var.set("Error occurred")
            self.log(f"ERROR: {str(e)}")
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
        finally:
            self.process_button.config(state=tk.NORMAL)


def main():
    parser = argparse.ArgumentParser(description="Process OpenBCI EEG data")
    parser.add_argument("--eeg", help="Path to OpenBCI EEG data file")
    parser.add_argument("--participant", help="Path to participant info file")
    parser.add_argument("--output", default="processed_data", help="Output directory")
    parser.add_argument("--low_threshold", type=float, default=10.0, 
                        help="Low value threshold for trigger detection (values below this are considered trigger ON)")
    parser.add_argument("--high_threshold", type=float, default=40.0,
                        help="High value threshold (waits for a value above this before detecting triggers)")
    
    args = parser.parse_args()
    
    if args.eeg and args.participant:
        # Command line mode
        print(f"Using value-based detection with low threshold {args.low_threshold} and high threshold {args.high_threshold}")
        success = process_eeg_data(args.eeg, args.participant, args.output, args.low_threshold, args.high_threshold)
        return success
    else:
        # GUI mode
        root = tk.Tk()
        app = EEGProcessorGUI(root)
        root.mainloop()
        
        # Restore stdout
        sys.stdout = app.stdout_original
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)