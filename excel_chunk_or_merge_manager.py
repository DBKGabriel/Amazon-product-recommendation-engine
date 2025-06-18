"""
Excel/CSV Chunk Manager
Split large Excel/CSV files into smaller chunks and merge them back together.
Supports chunking by file size or row count.
"""

import os
import sys
import re
import pathlib
import argparse
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import tempfile
import shutil
from typing import List, Optional, Tuple, Union
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class ChunkManager:
    """Manages splitting and merging of large Excel/CSV files."""
    
    # Configuration constants
    MAX_CHUNKS = 100  # Safety limit to prevent excessive file creation
    SAMPLE_SIZE_FOR_ESTIMATION = 1000  # Number of rows to sample for size estimation
    
    def __init__(self):
        self.chunk_count = 0
        
    @staticmethod
    def get_file_size_mb(file_path: pathlib.Path) -> float:
        """Get file size in megabytes."""
        return file_path.stat().st_size / (1024 * 1024)
    
    @staticmethod
    def get_file_size_bytes(file_path: pathlib.Path) -> int:
        """Get file size in bytes."""
        return file_path.stat().st_size
    
    @staticmethod
    def parse_size_string(size_str: str) -> int:
        """
        Takes size_str with optional unit (KB, MB, GB)
        and returns size in bytes as an integer
        """
        size_str = size_str.upper().strip()
        
        # Extract number and unit
        match = re.match(r'^(\d+(?:\.\d+)?)\s*(KB|MB|GB)?$', size_str)
        if not match:
            raise ValueError(f"Invalid size format: {size_str}. Use format like '50MB', '1.5GB', or '100KB'")
        
        number = float(match.group(1))
        unit = match.group(2) or 'MB'  # Default to MB if no unit
        
        multipliers = {
            'KB': 1024,
            'MB': 1024 * 1024,
            'GB': 1024 * 1024 * 1024
        }
        
        return int(number * multipliers[unit])
    
    def estimate_rows_for_size(self, file_path: pathlib.Path, target_size_bytes: int) -> int:
        """
       Takes 
            file_path : pathlib.Path
            -- Path to the file
            target_size_bytes : int
            -- Target size in bytes
        Returns
            Estimated number of rows to make up specified chunk size
        """
        file_extension = file_path.suffix.lower()
        
        print(f"  Analyzing file structure...")
        
        try:
            if file_extension == '.csv':
                # Read sample
                sample_df = pd.read_csv(file_path, nrows=self.SAMPLE_SIZE_FOR_ESTIMATION)
                
                # Estimate bytes per row by converting to string
                sample_string = sample_df.to_csv(index=False)
                bytes_per_row = len(sample_string.encode('utf-8')) / len(sample_df)
                
            elif file_extension in ['.xlsx', '.xls']:
                # Excel has a lot of bloat, so estimate conservatively
                sample_df = pd.read_excel(file_path, nrows=min(self.SAMPLE_SIZE_FOR_ESTIMATION, 500))
                cols = len(sample_df.columns)
                bytes_per_row = max(100, cols * 20)  # More conservative for Excel
                
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Calculate rows with safety margin
            safety_margin = 0.7 if file_extension in ['.xlsx', '.xls'] else 0.8
            estimated_rows = int((target_size_bytes * safety_margin) / bytes_per_row)
            
            # Apply reasonable bounds
            min_rows = 100
            max_rows = 500000 if file_extension == '.csv' else 100000
            estimated_rows = max(min_rows, min(estimated_rows, max_rows))
            
            print(f"  Estimated ~{bytes_per_row:.1f} bytes/row → {estimated_rows:,} rows per chunk")
            
            return estimated_rows
            
        except Exception as e:
            print(f"  Warning: Could not estimate chunk size ({e}), using defaults")
            # Fallback values
            if file_extension == '.csv':
                return min(50000, max(1000, target_size_bytes // 200))
            else:
                return min(25000, max(500, target_size_bytes // 500))
    
    def split_by_rows(self, file_path: pathlib.Path, chunk_size: int, 
                      output_dir: pathlib.Path) -> List[pathlib.Path]:
        """Split file by number of rows."""
        file_extension = file_path.suffix.lower()
        base_name = file_path.stem
        chunk_files = []
        
        print(f"Splitting {file_path.name} into chunks of {chunk_size:,} rows...")
        print(f"Original file size: {self.get_file_size_mb(file_path):.1f} MB")
        
        try:
            if file_extension == '.csv':
                # Process CSV in chunks for memory efficiency
                chunk_num = 1
                total_rows = 0
                
                for chunk_df in pd.read_csv(file_path, chunksize=chunk_size):
                    if self._check_chunk_limit(chunk_num):
                        break
                        
                    chunk_filename = f"{base_name}_part_{chunk_num:03d}.csv"
                    chunk_path = output_dir / chunk_filename
                    
                    chunk_df.to_csv(chunk_path, index=False)
                    chunk_files.append(chunk_path)
                    
                    total_rows += len(chunk_df)
                    chunk_size_mb = self.get_file_size_mb(chunk_path)
                    print(f"  Created {chunk_filename} ({len(chunk_df):,} rows, {chunk_size_mb:.1f} MB)")
                    
                    chunk_num += 1
                
            elif file_extension in ['.xlsx', '.xls']:
                # Load Excel file
                print("  Loading Excel file...")
                df = pd.read_excel(file_path)
                total_rows = len(df)
                
                print(f"  File contains {total_rows:,} rows")
                
                # Split into chunks
                chunk_num = 1
                for start_idx in range(0, total_rows, chunk_size):
                    if self._check_chunk_limit(chunk_num):
                        break
                        
                    end_idx = min(start_idx + chunk_size, total_rows)
                    chunk_df = df.iloc[start_idx:end_idx]
                    
                    chunk_filename = f"{base_name}_part_{chunk_num:03d}.xlsx"
                    chunk_path = output_dir / chunk_filename
                    
                    chunk_df.to_excel(chunk_path, index=False)
                    chunk_files.append(chunk_path)
                    
                    chunk_size_mb = self.get_file_size_mb(chunk_path)
                    print(f"  Created {chunk_filename} ({len(chunk_df):,} rows, {chunk_size_mb:.1f} MB)")
                    
                    chunk_num += 1
            
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            return chunk_files
            
        except Exception as e:
            print(f"Error during splitting: {e}")
            self._cleanup_chunks(chunk_files)
            raise
    
    def split_by_size(self, file_path: pathlib.Path, target_size_bytes: int, 
                      output_dir: pathlib.Path) -> List[pathlib.Path]:
        """Split file by target file size."""
        file_extension = file_path.suffix.lower()
        base_name = file_path.stem
        chunk_files = []
        
        print(f"Splitting {file_path.name} into chunks of ~{target_size_bytes/1024/1024:.1f} MB each...")
        
        # Estimate rows per chunk
        estimated_rows = self.estimate_rows_for_size(file_path, target_size_bytes)
        
        try:
            if file_extension == '.csv':
                # Process CSV efficiently
                chunk_num = 1
                total_rows = 0
                accumulated_dfs = []
                accumulated_rows = 0
                
                # Read in smaller chunks to accumulate
                read_chunk_size = min(5000, estimated_rows // 10)
                
                for read_chunk in pd.read_csv(file_path, chunksize=read_chunk_size):
                    accumulated_dfs.append(read_chunk)
                    accumulated_rows += len(read_chunk)
                    
                    # Write when we reach target or end of file
                    if accumulated_rows >= estimated_rows or len(read_chunk) < read_chunk_size:
                        if self._check_chunk_limit(chunk_num):
                            break
                            
                        # Combine accumulated data
                        chunk_df = pd.concat(accumulated_dfs, ignore_index=True)
                        
                        chunk_filename = f"{base_name}_part_{chunk_num:03d}.csv"
                        chunk_path = output_dir / chunk_filename
                        
                        chunk_df.to_csv(chunk_path, index=False)
                        chunk_files.append(chunk_path)
                        
                        total_rows += len(chunk_df)
                        actual_size_mb = self.get_file_size_mb(chunk_path)
                        print(f"  Created {chunk_filename} ({len(chunk_df):,} rows, {actual_size_mb:.1f} MB)")
                        
                        # Adjust estimate based on actual size
                        if chunk_num == 1 and actual_size_mb > 0:
                            actual_bytes_per_row = (actual_size_mb * 1024 * 1024) / len(chunk_df)
                            new_estimate = int((target_size_bytes * 0.8) / actual_bytes_per_row)
                            new_estimate = max(100, min(new_estimate, 500000))
                            if abs(new_estimate - estimated_rows) > estimated_rows * 0.2:
                                estimated_rows = new_estimate
                                print(f"  Adjusted to ~{estimated_rows:,} rows per chunk")
                        
                        # Reset for next chunk
                        accumulated_dfs = []
                        accumulated_rows = 0
                        chunk_num += 1
                
            elif file_extension in ['.xlsx', '.xls']:
                # Load Excel file
                print("  Loading Excel file...")
                df = pd.read_excel(file_path)
                total_rows = len(df)
                
                print(f"  File contains {total_rows:,} rows")
                
                # Split by estimated size
                chunk_num = 1
                current_row = 0
                
                while current_row < total_rows:
                    if self._check_chunk_limit(chunk_num):
                        break
                        
                    end_row = min(current_row + estimated_rows, total_rows)
                    chunk_df = df.iloc[current_row:end_row]
                    
                    chunk_filename = f"{base_name}_part_{chunk_num:03d}.xlsx"
                    chunk_path = output_dir / chunk_filename
                    
                    chunk_df.to_excel(chunk_path, index=False)
                    chunk_files.append(chunk_path)
                    
                    actual_size_mb = self.get_file_size_mb(chunk_path)
                    print(f"  Created {chunk_filename} ({len(chunk_df):,} rows, {actual_size_mb:.1f} MB)")
                    
                    # Adjust for next chunk
                    if chunk_num == 1 and actual_size_mb > 0:
                        target_mb = target_size_bytes / (1024 * 1024)
                        if actual_size_mb > target_mb * 1.2 or actual_size_mb < target_mb * 0.5:
                            ratio = target_mb / actual_size_mb
                            new_estimate = int(estimated_rows * ratio * 0.9)
                            new_estimate = max(100, min(new_estimate, total_rows - current_row))
                            estimated_rows = new_estimate
                            print(f"  Adjusted to ~{estimated_rows:,} rows per chunk")
                    
                    current_row = end_row
                    chunk_num += 1
            
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            return chunk_files
            
        except Exception as e:
            print(f"Error during splitting: {e}")
            self._cleanup_chunks(chunk_files)
            raise
    
    def merge_chunks(self, chunk_dir: pathlib.Path, output_path: Optional[pathlib.Path] = None) -> pathlib.Path:
        """
        Merge all chunk files in a directory back into a single file.
        
        Parameters
        ----------
        chunk_dir : pathlib.Path
            Directory containing chunk files
        output_path : pathlib.Path, optional
            Output file path
        
        Returns
        -------
        pathlib.Path
            Path to merged file
        """
        # Find all chunk files in directory
        chunk_files = self._find_chunk_files_in_dir(chunk_dir)
        
        if not chunk_files:
            raise ValueError(f"No chunk files found in directory: {chunk_dir}")
        
        # Determine output path if not provided
        if output_path is None:
            first_chunk = chunk_files[0]
            base_name = re.sub(r'_part_\d+$', '', first_chunk.stem)
            output_path = chunk_dir.parent / f"{base_name}_merged{first_chunk.suffix}"
        
        file_extension = chunk_files[0].suffix.lower()
        
        print(f"Merging {len(chunk_files)} chunk files from {chunk_dir.name}/")
        print("Chunks to merge (in order):")
        for i, chunk in enumerate(chunk_files[:5]):  # Show first 5
            print(f"  {i+1}. {chunk.name}")
        if len(chunk_files) > 5:
            print(f"  ... and {len(chunk_files) - 5} more files")
        
        try:
            if file_extension == '.csv':
                # Merge CSV files
                print("\nMerging CSV files...")
                dfs = []
                for i, chunk in enumerate(chunk_files):
                    print(f"  Reading {chunk.name} ({i+1}/{len(chunk_files)})")
                    dfs.append(pd.read_csv(chunk))
                
                print("  Combining dataframes...")
                merged_df = pd.concat(dfs, ignore_index=True)
                
                print("  Writing merged file...")
                merged_df.to_csv(output_path, index=False)
                
            elif file_extension in ['.xlsx', '.xls']:
                # Merge Excel files
                print("\nMerging Excel files...")
                dfs = []
                for i, chunk in enumerate(chunk_files):
                    print(f"  Reading {chunk.name} ({i+1}/{len(chunk_files)})")
                    dfs.append(pd.read_excel(chunk))
                
                print("  Combining dataframes...")
                merged_df = pd.concat(dfs, ignore_index=True)
                
                print("  Writing merged file...")
                merged_df.to_excel(output_path, index=False)
            
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            total_rows = len(merged_df) if 'merged_df' in locals() else 0
            output_size_mb = self.get_file_size_mb(output_path)
            
            print(f"\nMerge complete!")
            print(f"  Output: {output_path}")
            print(f"  Total rows: {total_rows:,}")
            print(f"  File size: {output_size_mb:.1f} MB")
            
            return output_path
            
        except Exception as e:
            print(f"Error during merging: {e}")
            if output_path.exists():
                output_path.unlink()
            raise
    
    def _check_chunk_limit(self, chunk_num: int) -> bool:
        """Check if we've hit the safety limit for chunks."""
        if chunk_num > self.MAX_CHUNKS:
            print(f"\nWarning: Reached {self.MAX_CHUNKS} chunks limit!")
            response = input("Continue splitting? This may indicate an issue. (y/n): ")
            if response.lower() != 'y':
                print("Stopping split operation.")
                return True
        return False
    
    def _cleanup_chunks(self, chunk_files: List[pathlib.Path]):
        """Clean up partial chunk files after error."""
        for chunk_file in chunk_files:
            if chunk_file.exists():
                try:
                    chunk_file.unlink()
                except:
                    pass
    
    def _find_chunk_files_in_dir(self, directory: pathlib.Path) -> List[pathlib.Path]:
        """Find all chunk files in a directory and return them sorted."""
        # Find all files matching chunk pattern
        chunk_pattern = re.compile(r'(.+)_part_(\d+)\.(csv|xlsx|xls)$', re.IGNORECASE)
        
        chunk_files = []
        base_names = set()
        
        for file_path in directory.iterdir():
            if file_path.is_file():
                match = chunk_pattern.match(file_path.name)
                if match:
                    base_name = match.group(1)
                    base_names.add(base_name)
                    chunk_files.append(file_path)
        
        if len(base_names) > 1:
            print(f"Warning: Found chunks from multiple files: {', '.join(base_names)}")
            print("Please ensure the directory contains chunks from only one split operation.")
        
        # Sort by part number
        def extract_part_number(path):
            match = chunk_pattern.match(path.name)
            return int(match.group(2)) if match else 0
        
        chunk_files.sort(key=extract_part_number)
        return chunk_files


def choose_file_gui(title: str = "Select file") -> Optional[pathlib.Path]:
    """Open GUI file chooser for files."""
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[
            ("Supported files", "*.xlsx;*.xls;*.csv"),
            ("Excel files", "*.xlsx;*.xls"),
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return pathlib.Path(file_path) if file_path else None


def choose_directory_gui(title: str = "Select directory") -> Optional[pathlib.Path]:
    """Open GUI directory chooser."""
    root = tk.Tk()
    root.withdraw()
    
    dir_path = filedialog.askdirectory(title=title)
    
    root.destroy()
    return pathlib.Path(dir_path) if dir_path else None


def get_chunk_settings_gui() -> Tuple[Optional[str], Optional[str]]:
    """Get chunk method and value from user via GUI."""
    root = tk.Tk()
    root.title("Chunk Settings")
    root.geometry("450x300")
    root.resizable(False, False)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (225)  # Half of window width
    y = (root.winfo_screenheight() // 2) - (150)  # Half of window height
    root.geometry(f"450x300+{x}+{y}")
    
    result = {'method': None, 'value': None}
    
    # Main frame with padding
    main_frame = tk.Frame(root, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Method selection
    tk.Label(main_frame, text="Choose chunking method:", font=("Arial", 11)).pack(anchor=tk.W, pady=(0, 10))
    
    method_var = tk.StringVar(value="size")
    tk.Radiobutton(main_frame, text="By file size (recommended)", 
                   variable=method_var, value="size", font=("Arial", 10)).pack(anchor=tk.W, padx=20)
    tk.Radiobutton(main_frame, text="By number of rows", 
                   variable=method_var, value="rows", font=("Arial", 10)).pack(anchor=tk.W, padx=20, pady=(5, 0))
    
    # Value input section
    tk.Label(main_frame, text="Enter chunk size:", font=("Arial", 11)).pack(anchor=tk.W, pady=(20, 5))
    
    # Input frame
    input_frame = tk.Frame(main_frame)
    input_frame.pack(anchor=tk.W, padx=20, pady=(0, 10))
    
    value_entry = tk.Entry(input_frame, width=25, font=("Arial", 10))
    value_entry.pack(side=tk.LEFT)
    value_entry.insert(0, "50MB")
    
    # Help text
    help_label = tk.Label(main_frame, text="Examples: 50MB, 1.5GB, 100KB or 50000 rows", 
                         font=("Arial", 9), fg="gray")
    help_label.pack(anchor=tk.W, padx=20)
    
    # Update placeholder based on method
    def update_placeholder(*args):
        if method_var.get() == "size":
            value_entry.delete(0, tk.END)
            value_entry.insert(0, "50MB")
            help_label.config(text="Examples: 50MB, 1.5GB, 100KB")
        else:
            value_entry.delete(0, tk.END)
            value_entry.insert(0, "50000")
            help_label.config(text="Examples: 50000, 100000, 25000")
    
    method_var.trace('w', update_placeholder)
    
    def on_ok():
        value = value_entry.get().strip()
        if not value:
            messagebox.showwarning("Input Required", "Please enter a chunk size value.")
            return
        result['method'] = method_var.get()
        result['value'] = value
        root.destroy()
    
    def on_cancel():
        root.destroy()
    
    # Button frame with proper spacing
    button_frame = tk.Frame(main_frame)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(30, 0))
    
    # Create buttons with consistent styling
    ok_button = tk.Button(button_frame, text="OK", command=on_ok, 
                         width=12, height=2, font=("Arial", 10), bg="#0078D4", fg="white")
    cancel_button = tk.Button(button_frame, text="Cancel", command=on_cancel, 
                             width=12, height=2, font=("Arial", 10))
    
    # Pack buttons with proper spacing
    ok_button.pack(side=tk.RIGHT, padx=(5, 0))
    cancel_button.pack(side=tk.RIGHT)
    
    # Bind Enter key to OK
    root.bind('<Return>', lambda e: on_ok())
    root.bind('<Escape>', lambda e: on_cancel())
    
    # Focus on entry
    value_entry.focus_set()
    
    root.mainloop()
    
    return result['method'], result['value']


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Split large Excel/CSV files into chunks or merge chunks back together",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split by file size (recommended)
  python chunk_manager.py split file.xlsx --size 50MB
  python chunk_manager.py split data.csv --size 1.5GB
  
  # Split by row count
  python chunk_manager.py split file.xlsx --rows 50000
  python chunk_manager.py split data.csv --rows 25000
  
  # Interactive mode
  python chunk_manager.py split                   # GUI to select file and settings
  
  # Merge operations  
  python chunk_manager.py merge chunk_folder/     # Merge all chunks in folder
  python chunk_manager.py merge                   # GUI to select chunk folder
        """
    )
    
    subparsers = parser.add_subparsers(dest='operation', help='Operation to perform')
    
    # Split command
    split_parser = subparsers.add_parser('split', help='Split a large file into chunks')
    split_parser.add_argument('file', nargs='?', help='File to split')
    
    # Chunking method (mutually exclusive)
    chunk_group = split_parser.add_mutually_exclusive_group()
    chunk_group.add_argument('-r', '--rows', type=int, help='Chunk by number of rows')
    chunk_group.add_argument('-s', '--size', help='Chunk by file size (e.g., 50MB, 1.5GB)')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge chunk files back together')
    merge_parser.add_argument('folder', nargs='?', help='Folder containing chunk files')
    merge_parser.add_argument('-o', '--output', help='Output file path')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    if not args.operation:
        print("Excel/CSV Chunk Manager")
        print("========================")
        print("Please specify either 'split' or 'merge' operation.")
        print("\nExamples:")
        print("  python chunk_manager.py split file.csv --size 50MB")
        print("  python chunk_manager.py merge chunk_folder/")
        print("\nUse --help for more information.")
        return 1
    
    manager = ChunkManager()
    
    try:
        if args.operation == 'split':
            # Get file to split
            if args.file:
                file_path = pathlib.Path(args.file)
                if not file_path.exists():
                    print(f"Error: File not found: {args.file}")
                    return 1
            else:
                print("Select file to split...")
                file_path = choose_file_gui("Select file to split")
                if not file_path:
                    print("No file selected.")
                    return 1
            
            # Get chunk method and value
            if args.rows:
                chunk_method = 'rows'
                chunk_value = str(args.rows)
            elif args.size:
                chunk_method = 'size'
                chunk_value = args.size
            else:
                print("\nChoose chunking method and size...")
                chunk_method, chunk_value = get_chunk_settings_gui()
                if not chunk_method or not chunk_value:
                    print("No chunking settings specified.")
                    return 1
            
            # Create output directory
            base_name = file_path.stem
            output_dir = file_path.parent / f"{base_name}_chunks"
            output_dir.mkdir(exist_ok=True)
            print(f"\nOutput directory: {output_dir}")
            
            # Perform split
            if chunk_method == 'rows':
                chunk_size = int(chunk_value)
                chunk_files = manager.split_by_rows(file_path, chunk_size, output_dir)
            else:  # size
                target_size_bytes = manager.parse_size_string(chunk_value)
                chunk_files = manager.split_by_size(file_path, target_size_bytes, output_dir)
            
            # Summary
            print(f"\nSuccess! Created {len(chunk_files)} chunk files in {output_dir.name}/")
            total_size_mb = sum(manager.get_file_size_mb(f) for f in chunk_files)
            print(f"Total size: {total_size_mb:.1f} MB")
            
        elif args.operation == 'merge':
            # Get chunk directory
            if args.folder:
                chunk_dir = pathlib.Path(args.folder)
                if not chunk_dir.exists() or not chunk_dir.is_dir():
                    print(f"Error: Directory not found: {args.folder}")
                    return 1
            else:
                print("Select folder containing chunk files...")
                chunk_dir = choose_directory_gui("Select chunk folder")
                if not chunk_dir:
                    print("No folder selected.")
                    return 1
            
            # Get output path
            output_path = pathlib.Path(args.output) if args.output else None
            
            # Perform merge
            merged_file = manager.merge_chunks(chunk_dir, output_path)
            print(f"\nSuccess! Merged file: {merged_file}")
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())