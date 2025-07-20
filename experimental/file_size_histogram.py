#!/usr/bin/env python3
"""
File Size Histogram Generator

This script creates a histogram of file sizes in bytes for all files under a given folder.
It recursively traverses the directory and plots the distribution of file sizes.

Usage:
    python file_size_histogram.py <folder_path> [options]
    python file_size_histogram.py --examples    # Show example usage
    python file_size_histogram.py --demo        # Run demo on current directory

Options:
    --bins <number>     Number of histogram bins (default: 50)
    --log-scale         Use logarithmic scale for file sizes
    --max-size <bytes>  Maximum file size to include (in bytes)
    --min-size <bytes>  Minimum file size to include (in bytes)
    --output <path>     Save plot to file (default: display only)
    --csv-output <path> Save histogram data to CSV file
    --extensions <ext>  File extensions to include (e.g., .png .jpg .txt)
    --by-type           Analyze files by type (XXXXX_<type>_YYYY.png format)
    --no-human-readable Display file sizes in raw bytes instead of human-readable format
    --examples          Show example usage
    --demo              Run demo on current directory
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def parse_file_type(filename):
    """
    Parse file type from filename in format XXXXX_<type>_YYYY.png
    
    Args:
        filename (str): Filename to parse
        
    Returns:
        str: File type or None if format doesn't match
    """
    import re
    # Pattern to match XXXXX_<type>_YYYY.png
    pattern = r'^[^_]+_([^_]+)_[^_]+\.png$'
    match = re.match(pattern, filename)
    return match.group(1) if match else None


def get_file_sizes_by_type(folder_path, min_size=0, max_size=None, extensions=None):
    """
    Recursively get file sizes from a folder, grouped by file type.
    
    Args:
        folder_path (str): Path to the folder to scan
        min_size (int): Minimum file size in bytes to include
        max_size (int): Maximum file size in bytes to include (None for no limit)
        extensions (list): List of file extensions to include (e.g., ['.png', '.jpg'])
    
    Returns:
        dict: Dictionary mapping file types to (file_sizes, size_to_files) tuples
    """
    file_types_data = {}  # Map file type to (file_sizes, size_to_files)
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return {}
    
    if not folder_path.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        return {}
    
    print(f"Scanning folder: {folder_path}")
    if extensions:
        print(f"Filtering for extensions: {', '.join(extensions)}")
    print("Collecting file sizes by type...")
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = Path(root) / file
            
            # Check file extension if specified
            if extensions:
                file_ext = file_path.suffix.lower()
                if file_ext not in extensions:
                    continue
            
            # Parse file type
            file_type = parse_file_type(file)
            if file_type is None:
                continue  # Skip files that don't match the expected format
            
            try:
                file_size = file_path.stat().st_size
                if file_size >= min_size and (max_size is None or file_size <= max_size):
                    # Initialize data structure for this file type if not exists
                    if file_type not in file_types_data:
                        file_types_data[file_type] = ([], {})
                    
                    file_sizes, size_to_files = file_types_data[file_type]
                    file_sizes.append(file_size)
                    
                    # Store file path for this size
                    if file_size not in size_to_files:
                        size_to_files[file_size] = []
                    size_to_files[file_size].append(str(file_path))
                    
            except (OSError, PermissionError) as e:
                print(f"Warning: Could not access {file_path}: {e}")
    
    # Print summary
    total_files = sum(len(data[0]) for data in file_types_data.values())
    print(f"Found {total_files} files across {len(file_types_data)} types:")
    for file_type, (file_sizes, _) in file_types_data.items():
        print(f"  {file_type}: {len(file_sizes)} files")
    
    return file_types_data


def get_file_sizes(folder_path, min_size=0, max_size=None, extensions=None):
    """
    Recursively get file sizes from a folder (legacy function for backward compatibility).
    
    Args:
        folder_path (str): Path to the folder to scan
        min_size (int): Minimum file size in bytes to include
        max_size (int): Maximum file size in bytes to include (None for no limit)
        extensions (list): List of file extensions to include (e.g., ['.png', '.jpg'])
    
    Returns:
        tuple: (list of file sizes in bytes, dict mapping sizes to file paths)
    """
    file_sizes = []
    size_to_files = {}  # Map file sizes to list of file paths
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return [], {}
    
    if not folder_path.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        return [], {}
    
    print(f"Scanning folder: {folder_path}")
    if extensions:
        print(f"Filtering for extensions: {', '.join(extensions)}")
    print("Collecting file sizes...")
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = Path(root) / file
            
            # Check file extension if specified
            if extensions:
                file_ext = file_path.suffix.lower()
                if file_ext not in extensions:
                    continue
            
            try:
                file_size = file_path.stat().st_size
                if file_size >= min_size and (max_size is None or file_size <= max_size):
                    file_sizes.append(file_size)
                    
                    # Store file path for this size
                    if file_size not in size_to_files:
                        size_to_files[file_size] = []
                    size_to_files[file_size].append(str(file_path))
                    
            except (OSError, PermissionError) as e:
                print(f"Warning: Could not access {file_path}: {e}")
    
    print(f"Found {len(file_sizes)} files")
    return file_sizes, size_to_files


def format_bytes(bytes_value):
    """Convert bytes to human readable format."""
    if bytes_value == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while bytes_value >= 1024 and i < len(size_names) - 1:
        bytes_value /= 1024.0
        i += 1
    
    return f"{bytes_value:.1f} {size_names[i]}"


def print_example_files_by_size(file_sizes, size_to_files, bins=10, log_scale=False, human_readable=True):
    """
    Print example files from different size ranges.
    
    Args:
        file_sizes (list): List of file sizes in bytes
        size_to_files (dict): Mapping of file sizes to file paths
        bins (int): Number of size ranges to create
        log_scale (bool): Whether to use logarithmic scale for size ranges
        human_readable (bool): Whether to format sizes in human-readable format
    """
    if not file_sizes:
        return
    
    sizes = np.array(file_sizes)
    min_size = np.min(sizes)
    max_size = np.max(sizes)
    
    print(f"\nExample Files by Size Range:")
    print("=" * 60)
    
    # Create size ranges based on log_scale parameter
    if min_size == max_size:
        # All files are the same size
        size_ranges = [(min_size, max_size)]
    elif log_scale:
        # Create logarithmic ranges for better distribution
        log_min = np.log10(min_size) if min_size > 0 else 0
        log_max = np.log10(max_size)
        log_ranges = np.logspace(log_min, log_max, bins + 1)
        size_ranges = [(int(log_ranges[i]), int(log_ranges[i + 1])) for i in range(len(log_ranges) - 1)]
    else:
        # Create linear ranges
        linear_ranges = np.linspace(min_size, max_size, bins + 1)
        size_ranges = [(int(linear_ranges[i]), int(linear_ranges[i + 1])) for i in range(len(linear_ranges) - 1)]
    
    for i, (range_min, range_max) in enumerate(size_ranges):
        # Find files in this size range
        files_in_range = []
        for size, files in size_to_files.items():
            if range_min <= size <= range_max:
                files_in_range.extend(files)
        
        if files_in_range:
            # Get a representative file (first one)
            example_file = files_in_range[0]
            file_size = next(size for size, files in size_to_files.items() if example_file in files)
            
            if human_readable:
                range_min_str = format_bytes(range_min)
                range_max_str = format_bytes(range_max)
                file_size_str = format_bytes(file_size)
            else:
                range_min_str = str(range_min)
                range_max_str = str(range_max)
                file_size_str = str(file_size)
            
            print(f"\nRange {i+1}: {range_min_str} - {range_max_str}")
            print(f"  Files in range: {len(files_in_range)}")
            print(f"  Example: {example_file} ({file_size_str})")
            
            # Show a few more examples if there are many files
            if len(files_in_range) > 1:
                additional_examples = files_in_range[1:min(4, len(files_in_range))]
                for example in additional_examples:
                    example_size = next(size for size, files in size_to_files.items() if example in files)
                    if human_readable:
                        example_size_str = format_bytes(example_size)
                    else:
                        example_size_str = str(example_size)
                    print(f"         {example} ({example_size_str})")


def print_histogram_data(sizes, bins, log_scale=False, csv_output=None, human_readable=True):
    """
    Print histogram data (bin counts and ranges).
    
    Args:
        sizes (numpy.array): Array of file sizes
        bins (int): Number of histogram bins
        log_scale (bool): Whether to use logarithmic scale
        csv_output (str): Path to save CSV data (optional)
        human_readable (bool): Whether to format sizes in human-readable format
    """
    print(f"\nHistogram Data:")
    print("=" * 80)
    
    if log_scale:
        # Calculate histogram with log scale
        hist, bin_edges = np.histogram(sizes, bins=bins)
        # Convert bin edges to log scale for display
        log_bin_edges = np.log10(bin_edges)
        
        print(f"{'Bin':<4} {'Range (log10)':<20} {'Range (bytes)':<25} {'Count':<8} {'Percentage':<12}")
        print("-" * 80)
        
        csv_data = []
        for i in range(len(hist)):
            if hist[i] > 0:  # Only show bins with files
                if human_readable:
                    range_start = format_bytes(int(bin_edges[i]))
                    range_end = format_bytes(int(bin_edges[i + 1]))
                else:
                    range_start = str(int(bin_edges[i]))
                    range_end = str(int(bin_edges[i + 1]))
                percentage = (hist[i] / len(sizes)) * 100
                print(f"{i+1:<4} {log_bin_edges[i]:<20.2f} {range_start:<12} - {range_end:<12} {hist[i]:<8} {percentage:<12.2f}%")
                
                # Store data for CSV
                csv_data.append({
                    'bin': i + 1,
                    'range_start_bytes': int(bin_edges[i]),
                    'range_end_bytes': int(bin_edges[i + 1]),
                    'range_start_formatted': range_start,
                    'range_end_formatted': range_end,
                    'count': hist[i],
                    'percentage': percentage,
                    'log_range_start': log_bin_edges[i],
                    'log_range_end': log_bin_edges[i + 1]
                })
    else:
        # Calculate histogram with linear scale
        hist, bin_edges = np.histogram(sizes, bins=bins)
        
        print(f"{'Bin':<4} {'Range (bytes)':<30} {'Count':<8} {'Percentage':<12}")
        print("-" * 60)
        
        csv_data = []
        for i in range(len(hist)):
            if hist[i] > 0:  # Only show bins with files
                if human_readable:
                    range_start = format_bytes(int(bin_edges[i]))
                    range_end = format_bytes(int(bin_edges[i + 1]))
                else:
                    range_start = str(int(bin_edges[i]))
                    range_end = str(int(bin_edges[i + 1]))
                percentage = (hist[i] / len(sizes)) * 100
                print(f"{i+1:<4} {range_start:<15} - {range_end:<15} {hist[i]:<8} {percentage:<12.2f}%")
                
                # Store data for CSV
                csv_data.append({
                    'bin': i + 1,
                    'range_start_bytes': int(bin_edges[i]),
                    'range_end_bytes': int(bin_edges[i + 1]),
                    'range_start_formatted': range_start,
                    'range_end_formatted': range_end,
                    'count': hist[i],
                    'percentage': percentage
                })
    
    print("=" * 80)
    
    # Save to CSV if requested
    if csv_output and csv_data:
        try:
            import csv
            with open(csv_output, 'w', newline='') as csvfile:
                fieldnames = csv_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            print(f"Histogram data saved to: {csv_output}")
        except Exception as e:
            print(f"Warning: Could not save CSV data to {csv_output}: {e}")
    
    return csv_data


def create_histogram_by_type(file_types_data, bins=50, log_scale=False, output_path=None, csv_output=None, human_readable=True):
    """
    Create and display/save histograms for multiple file types on the same plot.
    
    Args:
        file_types_data (dict): Dictionary mapping file types to (file_sizes, size_to_files) tuples
        bins (int): Number of histogram bins
        log_scale (bool): Whether to use logarithmic scale
        output_path (str): Path to save the plot (None for display only)
        csv_output (str): Path to save CSV data (optional)
        human_readable (bool): Whether to format sizes in human-readable format
    """
    if not file_types_data:
        print("No files found to plot.")
        return
    
    # Colors for different file types
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum', 'lightsteelblue', 'wheat', 'lightpink']
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Process each file type
    for i, (file_type, (file_sizes, size_to_files)) in enumerate(file_types_data.items()):
        if not file_sizes:
            continue
            
        sizes = np.array(file_sizes)
        color = colors[i % len(colors)]
        
        # Calculate statistics
        total_size = np.sum(sizes)
        mean_size = np.mean(sizes)
        median_size = np.median(sizes)
        min_size = np.min(sizes)
        max_size = np.max(sizes)
        
        print(f"\n{'='*60}")
        print(f"File Type: {file_type}")
        print(f"{'='*60}")
        print(f"File Size Statistics:")
        print(f"  Total files: {len(sizes):,}")
        if human_readable:
            print(f"  Total size: {format_bytes(total_size)}")
            print(f"  Mean size: {format_bytes(mean_size)}")
            print(f"  Median size: {format_bytes(median_size)}")
            print(f"  Min size: {format_bytes(min_size)}")
            print(f"  Max size: {format_bytes(max_size)}")
        else:
            print(f"  Total size: {total_size:,} bytes")
            print(f"  Mean size: {mean_size:,.0f} bytes")
            print(f"  Median size: {median_size:,.0f} bytes")
            print(f"  Min size: {min_size:,} bytes")
            print(f"  Max size: {max_size:,} bytes")
        
        # Print histogram data for this type
        csv_output_type = f"{csv_output}_{file_type}.csv" if csv_output else None
        print_histogram_data(sizes, bins, log_scale, csv_output_type, human_readable)
        
        # Print example files by size range for this type
        print_example_files_by_size(file_sizes, size_to_files, bins=bins, log_scale=log_scale, human_readable=human_readable)
        
        # Add to plot
        if log_scale:
            plt.hist(sizes, bins=bins, alpha=0.6, color=color, edgecolor='black', 
                    label=f'{file_type} ({len(sizes)} files)', density=True)
        else:
            plt.hist(sizes, bins=bins, alpha=0.6, color=color, edgecolor='black', 
                    label=f'{file_type} ({len(sizes)} files)', density=True)
    
    # Configure plot
    if log_scale:
        plt.xscale('log')
        plt.xlabel('File Size (bytes, log scale)')
    else:
        plt.xlabel('File Size (bytes)')
    
    plt.ylabel('Density')
    plt.title(f'File Size Distribution by Type\n(Total: {sum(len(data[0]) for data in file_types_data.values()):,} files)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def create_histogram(file_sizes, size_to_files=None, bins=50, log_scale=False, output_path=None, csv_output=None, human_readable=True):
    """
    Create and display/save a histogram of file sizes (legacy function for backward compatibility).
    
    Args:
        file_sizes (list): List of file sizes in bytes
        size_to_files (dict): Mapping of file sizes to file paths (optional)
        bins (int): Number of histogram bins
        log_scale (bool): Whether to use logarithmic scale
        output_path (str): Path to save the plot (None for display only)
        csv_output (str): Path to save CSV data (optional)
        human_readable (bool): Whether to format sizes in human-readable format
    """
    if not file_sizes:
        print("No files found to plot.")
        return
    
    # Convert to numpy array for easier manipulation
    sizes = np.array(file_sizes)
    
    # Calculate statistics
    total_size = np.sum(sizes)
    mean_size = np.mean(sizes)
    median_size = np.median(sizes)
    min_size = np.min(sizes)
    max_size = np.max(sizes)
    
    print(f"\nFile Size Statistics:")
    print(f"  Total files: {len(sizes):,}")
    if human_readable:
        print(f"  Total size: {format_bytes(total_size)}")
        print(f"  Mean size: {format_bytes(mean_size)}")
        print(f"  Median size: {format_bytes(median_size)}")
        print(f"  Min size: {format_bytes(min_size)}")
        print(f"  Max size: {format_bytes(max_size)}")
    else:
        print(f"  Total size: {total_size:,} bytes")
        print(f"  Mean size: {mean_size:,.0f} bytes")
        print(f"  Median size: {median_size:,.0f} bytes")
        print(f"  Min size: {min_size:,} bytes")
        print(f"  Max size: {max_size:,} bytes")
    
    # Print histogram data
    print_histogram_data(sizes, bins, log_scale, csv_output, human_readable)
    
    # Print example files by size range
    if size_to_files:
        print_example_files_by_size(file_sizes, size_to_files, bins=bins, log_scale=log_scale, human_readable=human_readable)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    if log_scale:
        # Use log scale for better visualization of wide range of sizes
        plt.hist(sizes, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xscale('log')
        plt.xlabel('File Size (bytes, log scale)')
    else:
        plt.hist(sizes, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('File Size (bytes)')
    
    plt.ylabel('Number of Files')
    if human_readable:
        title_total = format_bytes(total_size)
        stats_text = f'Mean: {format_bytes(mean_size)}\nMedian: {format_bytes(median_size)}\nMin: {format_bytes(min_size)}\nMax: {format_bytes(max_size)}'
    else:
        title_total = f"{total_size:,} bytes"
        stats_text = f'Mean: {mean_size:,.0f} bytes\nMedian: {median_size:,.0f} bytes\nMin: {min_size:,} bytes\nMax: {max_size:,} bytes'
    
    plt.title(f'File Size Distribution\n({len(sizes):,} files, total: {title_total})')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Create a histogram of file sizes in a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('folder_path', nargs='?', help='Path to the folder to scan')
    parser.add_argument('--bins', type=int, default=50, help='Number of histogram bins (default: 50)')
    parser.add_argument('--log-scale', action='store_true', help='Use logarithmic scale for file sizes')
    parser.add_argument('--max-size', type=int, help='Maximum file size to include (in bytes)')
    parser.add_argument('--min-size', type=int, default=0, help='Minimum file size to include (in bytes)')
    parser.add_argument('--output', help='Save plot to file (default: display only)')
    parser.add_argument('--csv-output', help='Save histogram data to CSV file')
    parser.add_argument('--extensions', nargs='+', help='File extensions to include (e.g., .png .jpg .txt)')
    parser.add_argument('--by-type', action='store_true', help='Analyze files by type (XXXXX_<type>_YYYY.png format)')
    parser.add_argument('--no-human-readable', action='store_true', help='Display file sizes in raw bytes instead of human-readable format')
    
    args = parser.parse_args()
    
    # Check if folder path is provided
    if not args.folder_path:
        print("Error: Please provide a folder path to scan.")
        print("Use --help for usage information or --examples to see examples.")
        sys.exit(1)
    
    # Get file sizes
    extensions = args.extensions if args.extensions else None
    human_readable = not args.no_human_readable
    
    if args.by_type:
        # Analyze files by type
        file_types_data = get_file_sizes_by_type(args.folder_path, args.min_size, args.max_size, extensions)
        
        if file_types_data:
            # Create histogram by type
            create_histogram_by_type(file_types_data, args.bins, args.log_scale, args.output, args.csv_output, human_readable)
        else:
            if extensions:
                print(f"No files with extensions {extensions} found matching the criteria.")
            else:
                print("No files found matching the criteria.")
            sys.exit(1)
    else:
        # Legacy single histogram mode
        file_sizes, size_to_files = get_file_sizes(args.folder_path, args.min_size, args.max_size, extensions)
        
        if file_sizes:
            # Create histogram
            create_histogram(file_sizes, size_to_files, args.bins, args.log_scale, args.output, args.csv_output, human_readable)
        else:
            if extensions:
                print(f"No files with extensions {extensions} found matching the criteria.")
            else:
                print("No files found matching the criteria.")
            sys.exit(1)


if __name__ == "__main__":
    main() 