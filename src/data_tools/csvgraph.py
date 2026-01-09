# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "argparse",
#   "numpy",
#   "matplotlib",
#   "scipy"
# ]
# ///

import argparse
import csv
import itertools
import sys
from decimal import Decimal, getcontext
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, TextIO, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.signal import detrend

# Set high precision for decimal calculations
getcontext().prec = 50

# Set global matplotlib settings
plt.rcParams["figure.figsize"] = (4, 4)
plt.rcParams["figure.dpi"] = 80
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.autolayout"] = True


class TimeUnit(Enum):
    SECONDS = (9, "s")
    MILLISECONDS = (6, "ms")
    MICROSECONDS = (3, "us")
    NANOSECONDS = (0, "ns")


class UnitFactor:
    def __init__(self, unit: TimeUnit):
        self.unit = unit
        self.sec = self.convert(TimeUnit.SECONDS)
        self.msec = self.convert(TimeUnit.MILLISECONDS)
        self.usec = self.convert(TimeUnit.MICROSECONDS)
        self.nsec = self.convert(TimeUnit.NANOSECONDS)

    def convert(self, unit: TimeUnit) -> Callable[[Decimal], Decimal]:
        def factor(value: Decimal) -> Decimal:
            return value * (Decimal(10) ** (self.unit.value[0] - unit.value[0]))

        return factor


class TimestampAnalyzer:
    def __init__(self, ts: List[Decimal], unit: TimeUnit = TimeUnit.NANOSECONDS):
        self.ts = ts
        self.ts_diff: List[Decimal] | None = None
        self.ts_detrended: List[Decimal] | None = None
        self.ts_drift: List[Decimal] | None = None
        self.unit = unit
        self.factor = UnitFactor(unit)
        self._plot = "ts"
        self.hidden = []

    def sec(self):
        self.unit = TimeUnit.SECONDS
        return self

    def msec(self):
        self.unit = TimeUnit.MILLISECONDS
        return self

    def usec(self):
        self.unit = TimeUnit.MICROSECONDS
        return self

    def nsec(self):
        self.unit = TimeUnit.NANOSECONDS
        return self

    @property
    def t(self):
        return self.__dict__[self._plot]

    def linear(self):
        self._plot = "ts"
        return self

    def diff(self):
        self.ts_diff = [self.ts[i+1] - self.ts[i] for i in range(len(self.ts) - 1)]
        self._plot = "ts_diff"
        return self

    def mdiff(self, expected: Decimal):
        self.ts_mdiff = [self.ts[i+1] - self.ts[i] for i in range(len(self.ts) - 1)]
        for i in range(len(self.ts_mdiff)):
            self.ts_mdiff[i] -= expected
            while self.ts_mdiff[i] > expected / 2:
                self.ts_mdiff[i] -= expected
            while self.ts_mdiff[i] < -expected / 2:
                self.ts_mdiff[i] += expected
        self._plot = "ts_mdiff"
        return self

    def detrend(self):
        # Convert to numpy array for scipy, then back to Decimal list
        ts_array = np.array([float(t) for t in self.ts])
        detrended_array = detrend(ts_array, type="linear")
        self.ts_detrended = [Decimal(str(val)) for val in detrended_array]
        self._plot = "ts_detrended"
        return self

    def drift(self, expected_slope: Decimal):
        self.ts_drift = []
        for i in range(len(self.ts)):
            linear_trend = expected_slope * Decimal(i) + self.ts[0]
            self.ts_drift.append(self.ts[i] - linear_trend)
        # self.trend = (expected_slope, self.ts[0])
        self._plot = "ts_drift"
        return self

    def plot(self, title: str, color: str = None, label: str = None, size: float = None):
        y = self.__dict__[self._plot]
        # Remove hidden indices
        y_filtered = [y[i] for i in range(len(y)) if i not in self.hidden]
        # Convert units and then to float for plotting
        y_converted = [float(self.factor.convert(self.unit)(val)) for val in y_filtered]
        x = np.arange(len(y_converted))
        plt.scatter(x, y_converted, s=size, color=color, label=label, zorder=len(plt.gca().collections))  # pyright: ignore[reportUnknownMemberType]
        # plt.title(title)  # pyright: ignore[reportUnknownMemberType]
        plt.xlabel("Index")  # pyright: ignore[reportUnknownMemberType]
        # plt.ylabel(f"Timestamp ({self.unit.value[1]})")  # pyright: ignore[reportUnknownMemberType]
        plt.ylabel(f"{title} ({self.unit.value[1]})")  # pyright: ignore[reportUnknownMemberType]
        return self

    def outliers(self, thresh: float) -> List[Tuple[int, Decimal]]:
        y = self.__dict__[self._plot]
        # Convert to float for numpy statistics, but keep original precision
        y_float = [float(val) for val in y]
        mean = Decimal(str(np.mean(y_float)))
        std = Decimal(str(np.std(y_float)))
        outliers: List[Tuple[int, Decimal]] = []
        for i, val in enumerate(y):
            if abs(val - mean) > Decimal(str(thresh)) * std:
                outliers.append((i, val))
        return outliers

    def hide_indices(self, indices: List[int]) -> None:
        self.hidden = indices


def parse_csv(reader, topics: List[str] = None) -> List[Tuple[str, List[Decimal]]]:
    """Parse CSV and return list of (topic_name, timestamps) tuples.

    Args:
        reader: CSV reader object
        topics: Optional list of topic names to filter. If None, returns all topics.

    Returns:
        List of tuples, each containing (topic_name, timestamps_list)
    """
    rows = list(reader)
    if not rows:
        return []

    # First row contains headers
    headers = [h.strip() for h in rows[0] if h]

    # Find which columns to use
    if topics:
        # Find column indices for requested topics
        topic_indices = []
        for topic in topics:
            try:
                idx = headers.index(topic)
                topic_indices.append((topic, idx))
            except ValueError:
                raise ValueError(f"Topic '{topic}' not found in CSV headers: {headers}")
    else:
        # Use all columns except the first (index column)
        topic_indices = [(header, idx + 1) for idx, header in enumerate(headers)]

    # Parse data for each topic
    results = []
    for topic_name, col_idx in topic_indices:
        ts: List[Decimal] = []
        for row in rows[1:]:  # Skip header row
            if not row or len(row) <= col_idx:
                continue
            try:
                ts.append(Decimal(row[col_idx]))
            except (ValueError, IndexError):
                continue
        results.append((topic_name, ts))

    return results


def analyze_ts(
    ts: List[Decimal],
    plot: str,
    unit: TimeUnit,
    to_unit: TimeUnit,
    expected: Decimal,
    rm_outliers: int,
) -> List[Decimal]:
    analyzer = TimestampAnalyzer(ts, unit)
    if rm_outliers > 0:
        outliers = analyzer.diff().outliers(rm_outliers)
        outliers = [i for (i, _) in outliers]
        analyzer.hide_indices(outliers)

    # Helper to convert data to target unit
    def convert_data(data: List[Decimal]) -> List[Decimal]:
        converter = UnitFactor(unit).convert(to_unit)
        return [converter(val) for val in data]

    match plot:
        case "linear":
            return convert_data(analyzer.linear().t)
        case "diff":
            return convert_data(analyzer.diff().t)
        case "mdiff":
            return convert_data(
                analyzer.mdiff(
                    UnitFactor(unit).convert(to_unit)(expected)
                ).t
            )
        case "detrend":
            return convert_data(analyzer.detrend().t)
        case "drift":
            return convert_data(
                analyzer.drift(
                    UnitFactor(unit).convert(to_unit)(expected)
                ).t
            )
        case _:
            raise ValueError(f"Invalid plot type: {plot}")


def plot_ts(
    ts: List[Decimal],
    title: str,
    plot: str,
    from_unit: TimeUnit,
    to_unit: TimeUnit,
    expected: Decimal,
    rm_outliers: int,
    color: str = None,
    label: str = None,
    size: float = None,
    force_title: bool = False
):
    analyzer = TimestampAnalyzer(ts, from_unit)
    if rm_outliers > 0:
        outliers = analyzer.diff().outliers(rm_outliers)
        outliers = [i for (i, _) in outliers]
        analyzer.hide_indices(outliers)

    # Set the output unit based on to_unit parameter
    if to_unit == TimeUnit.SECONDS:
        analyzer = analyzer.sec()
    elif to_unit == TimeUnit.MILLISECONDS:
        analyzer = analyzer.msec()
    elif to_unit == TimeUnit.MICROSECONDS:
        analyzer = analyzer.usec()
    elif to_unit == TimeUnit.NANOSECONDS:
        analyzer = analyzer.nsec()

    plot_titles = {
        "linear": f"{title} Timestamp",
        "diff": f"{title} Difference",
        "mdiff": f"{title} Difference",
        "detrend": f"{title} Detrended",
        "detrended": f"{title} Detrended",
    }

    if force_title:
        plot_title = title
    else:
        plot_title = plot_titles[plot]

    match plot:
        case "linear":
            analyzer.linear().plot(title=plot_title, color=color, label=label, size=size)
        case "diff":
            analyzer.diff().plot(title=plot_title, color=color, label=label, size=size)
        case "mdiff":
            analyzer.mdiff(
                UnitFactor(from_unit).convert(to_unit)(expected)
            ).plot(title=plot_title, color=color, label=label, size=size)
        case "detrend":
            analyzer.detrend().plot(title=plot_title, color=color, label=label, size=size)
        case "drift":
            analyzer.drift(
                UnitFactor(from_unit).convert(to_unit)(expected)
            ).plot(title=plot_title, color=color, label=label, size=size)
        case _:
            raise ValueError(f"Invalid plot type: {plot}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze timestamps")
    parser.add_argument("CSV", help="CSV file(s) to analyze", nargs="*", default=None)
    parser.add_argument(
        "-u",
        "--unit",
        choices=["s", "ms", "us", "ns"],
        default="ns",
        help="Time unit (applies to all files)",
    )
    parser.add_argument(
        "--to-unit",
        choices=["s", "ms", "us", "ns"],
        default="ms",
        help="Output time unit (applies to all files)",
    )
    parser.add_argument(
        "--file-units",
        type=str,
        nargs="+",
        help="Input time unit per file (one per CSV file: s, ms, us, or ns)",
    )
    parser.add_argument(
        "--file-output-units",
        type=str,
        nargs="+",
        help="Output time unit per file (one per CSV file: s, ms, us, or ns)",
    )
    parser.add_argument(
        "-t", "--topics", type=str, nargs="+", help="Topic(s) to analyze (applies to all files)"
    )
    parser.add_argument(
        "--file-topics",
        type=str,
        nargs="+",
        help="Topic(s) per file (one entry per CSV file, use '*' for all topics, comma-separate multiple topics)",
    )
    parser.add_argument("--title", type=str, nargs="+", help="Title of the plot")
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        help="Custom labels for legend (must match number of datasets)",
    )
    parser.add_argument(
        "-p",
        "--plot",
        type=str,
        choices=["linear", "diff", "mdiff", "detrend", "drift"],
    )
    parser.add_argument(
        "-e",
        "--expected",
        type=float,
        help="Expected trend in ms (usable with drift plot)",
        default=200,
    )
    parser.add_argument(
        "-r",
        "--rm_outliers",
        type=float,
        help="Remove outliers above this many std deviations",
        default=0,
    )
    parser.add_argument(
        "-S",
        "--start-index",
        type=int,
        help="Start index for data analysis",
    )
    parser.add_argument(
        "-E",
        "--end-index",
        type=int,
        help="End index for data analysis",
    )
    parser.add_argument(
        "--tocsv",
        action="store_true",
        help="Output to CSV file",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print mean and standard deviation instead of plotting",
    )
    # parser.add_argument("-o", "--output_dir", type=str, help="Output directory")
    parser.add_argument("--stdin", action="store_true", help="Input from stdin")
    parser.add_argument("--stdout", action="store_true", help="Output to stdout")
    parser.add_argument(
        "--stdio", action="store_true", help="Input and output from stdin"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        help="DPI for output plot",
        default=80,
    )
    parser.add_argument(
        "-f",
        "--force-title",
        action="store_true",
        help="Force title for plot",
    )
    return parser.parse_args()


def main():
    units = {
        "s": TimeUnit.SECONDS,
        "ms": TimeUnit.MILLISECONDS,
        "us": TimeUnit.MICROSECONDS,
        "ns": TimeUnit.NANOSECONDS,
    }
    args = parse_args()

    # Default colormap for multiple files
    colors = plt.cm.tab10.colors  # pyright: ignore[reportUnknownMemberType]

    input_files = []
    csv_names = []

    if args.stdin or args.stdio:
        input_files = [sys.stdin]
        csv_names = ["stdin"]
    elif args.CSV:
        input_files = [open(csv_file, 'r') for csv_file in args.CSV]
        csv_names = [Path(csv_file).stem for csv_file in args.CSV]
    else:
        raise ValueError("No input files specified")

    if args.stdout or args.stdio:
        output_file = sys.stdout
    elif not args.stats:
        raise ValueError("Not implemented")
    else:
        output_file = None  # Not needed for stats mode

    if args.dpi:
        plt.rcParams['figure.dpi'] = args.dpi

    # Validate file-topics if provided
    if args.file_topics:
        if len(args.file_topics) != len(input_files):
            raise ValueError(
                f"Number of --file-topics entries ({len(args.file_topics)}) must match "
                f"number of CSV files ({len(input_files)})"
            )
        if args.topics:
            raise ValueError("Cannot use both -t/--topics and --file-topics together")

    # Validate file-units if provided
    if args.file_units:
        if len(args.file_units) != len(input_files):
            raise ValueError(
                f"Number of --file-units entries ({len(args.file_units)}) must match "
                f"number of CSV files ({len(input_files)})"
            )
        # Validate each unit
        valid_units = ["s", "ms", "us", "ns"]
        for unit in args.file_units:
            if unit not in valid_units:
                raise ValueError(f"Invalid unit '{unit}'. Must be one of: {valid_units}")

    # Validate file-output-units if provided
    if args.file_output_units:
        if len(args.file_output_units) != len(input_files):
            raise ValueError(
                f"Number of --file-output-units entries ({len(args.file_output_units)}) must match "
                f"number of CSV files ({len(input_files)})"
            )
        # Validate each unit
        valid_units = ["s", "ms", "us", "ns"]
        for unit in args.file_output_units:
            if unit not in valid_units:
                raise ValueError(f"Invalid unit '{unit}'. Must be one of: {valid_units}")
    
    # Validate mutually exclusive output options
    if args.tocsv and args.stats:
        raise ValueError("Cannot use both --tocsv and --stats together")
    
    try:
        if args.tocsv:
                if len(input_files) > 1:
                    raise ValueError("CSV output only supports single file input")
                if args.topics and len(args.topics) > 1:
                    raise ValueError("CSV output only supports single topic")
                if args.file_topics and ',' in args.file_topics[0]:
                    raise ValueError("CSV output only supports single topic")

                reader = csv.reader(input_files[0])
                # Trim data after header according to start and end index
                if args.start_index is not None:
                    reader = itertools.islice(reader, args.start_index, None)
                if args.end_index is not None:
                    reader = itertools.islice(reader, args.end_index)

                writer = csv.writer(output_file)
                datasets = parse_csv(reader, args.topics)
                if not datasets:
                    raise ValueError("No data found in CSV")

                title, ts = datasets[0]
                t = analyze_ts(
                    ts,
                    args.plot,
                    units[args.unit],
                    units[args.to_unit],
                    Decimal(str(args.expected)),
                    rm_outliers=args.rm_outliers,
                )
                header = f"{title} {args.plot}"
                writer.writerow([header])
                for value in t:
                    writer.writerow([str(value)])
        elif args.stats:
            # Statistics mode - print mean and std dev
            all_datasets = []
            
            for idx, (file_handle, csv_name) in enumerate(zip(input_files, csv_names)):
                reader = csv.reader(file_handle)
                # Trim data after header according to start and end index
                # Read header first to preserve it
                rows = list(reader)
                if args.start_index is not None or args.end_index is not None:
                    start = args.start_index + 1 if args.start_index is not None else 1
                    end = args.end_index + 1 if args.end_index is not None else None
                    rows = [rows[0]] + rows[start:end]
                reader = iter(rows)

                # Determine which topics to extract from this file
                if args.file_topics:
                    # Per-file topic specification
                    file_topic_spec = args.file_topics[idx]
                    if file_topic_spec == '*' or file_topic_spec.lower() == 'all':
                        # Extract all topics
                        file_topics = None
                    else:
                        # Extract specified topics (comma-separated)
                        file_topics = [t.strip() for t in file_topic_spec.split(',')]
                else:
                    # Use global topics (applies to all files)
                    file_topics = args.topics

                # Determine input and output units for this file
                if args.file_units:
                    file_input_unit = units[args.file_units[idx]]
                else:
                    file_input_unit = units[args.unit]
                
                if args.file_output_units:
                    file_output_unit = units[args.file_output_units[idx]]
                else:
                    file_output_unit = units[args.to_unit]

                # Parse CSV with optional topic filtering
                datasets = parse_csv(reader, file_topics)

                for topic_name, ts in datasets:
                    # Analyze the data
                    analyzed_data = analyze_ts(
                        ts,
                        args.plot,
                        file_input_unit,
                        file_output_unit,
                        Decimal(str(args.expected)),
                        rm_outliers=args.rm_outliers,
                    )
                    
                    # Store file info along with dataset and units
                    all_datasets.append((topic_name, analyzed_data, csv_name, len(input_files), file_output_unit))
            
            # Determine how many topics are being plotted across all files
            unique_topics = set(topic_name for topic_name, _, _, _, _ in all_datasets)
            num_topics = len(unique_topics)
            
            # Now create final dataset list with proper labels
            final_datasets = []
            for topic_name, analyzed_data, csv_name, num_files, file_output_unit in all_datasets:
                # Determine the label based on context
                if num_files > 1 and num_topics > 1:
                    # Multiple files AND multiple topics: use "filename: topic"
                    label = f"{csv_name}: {topic_name}"
                elif num_files > 1 and num_topics == 1:
                    # Multiple files, single topic: use filename only
                    label = csv_name
                elif num_files == 1 and num_topics > 1:
                    # Single file, multiple topics: use topic name only
                    label = topic_name
                else:
                    # Single file, single topic: no label needed, use topic name
                    label = topic_name
                
                final_datasets.append((label, analyzed_data, file_output_unit))
            
            # Override labels if custom labels provided
            if args.labels:
                if len(args.labels) != len(final_datasets):
                    raise ValueError(
                        f"Number of custom labels ({len(args.labels)}) must match "
                        f"number of datasets ({len(final_datasets)})"
                    )
                final_datasets = [
                    (custom_label, analyzed_data, file_output_unit)
                    for (_, analyzed_data, file_output_unit), custom_label in zip(final_datasets, args.labels)
                ]
            
            # Print statistics for each dataset
            for label, data, output_unit in final_datasets:
                data_float = [float(d) for d in data]
                mean = np.mean(data_float)
                std = np.std(data_float)
                min_val = np.min(data_float)
                max_val = np.max(data_float)
                print(f"Dataset: {label}")
                print(f"  Mean: {mean} {output_unit.value[1]}")
                print(f"  Std Dev: {std} {output_unit.value[1]}")
                print(f"  Min: {min_val} {output_unit.value[1]}")
                print(f"  Max: {max_val} {output_unit.value[1]}")
                print()
        else:
            # Collect all datasets to plot
            all_datasets = []

            for idx, (file_handle, csv_name) in enumerate(zip(input_files, csv_names)):
                reader = csv.reader(file_handle)
                # Trim data after header according to start and end index
                # Read header first to preserve it
                rows = list(reader)
                if args.start_index is not None or args.end_index is not None:
                    start = args.start_index + 1 if args.start_index is not None else 1
                    end = args.end_index + 1 if args.end_index is not None else None
                    rows = [rows[0]] + rows[start:end]
                reader = iter(rows)

                # Determine which topics to extract from this file
                if args.file_topics:
                    # Per-file topic specification
                    file_topic_spec = args.file_topics[idx]
                    if file_topic_spec == '*' or file_topic_spec.lower() == 'all':
                        # Extract all topics
                        file_topics = None
                    else:
                        # Extract specified topics (comma-separated)
                        file_topics = [t.strip() for t in file_topic_spec.split(',')]
                else:
                    # Use global topics (applies to all files)
                    file_topics = args.topics

                # Determine input and output units for this file
                if args.file_units:
                    file_input_unit = units[args.file_units[idx]]
                else:
                    file_input_unit = units[args.unit]

                if args.file_output_units:
                    file_output_unit = units[args.file_output_units[idx]]
                else:
                    file_output_unit = units[args.to_unit]

                # Parse CSV with optional topic filtering
                datasets = parse_csv(reader, file_topics)

                for topic_name, ts in datasets:
                    # Store file info along with dataset and units
                    all_datasets.append((topic_name, ts, csv_name, len(input_files), file_input_unit, file_output_unit))

            # Determine how many topics are being plotted across all files
            unique_topics = set(topic_name for topic_name, _, _, _, _, _ in all_datasets)
            num_topics = len(unique_topics)

            # Now create final dataset list with proper labels
            final_datasets = []
            for topic_name, ts, csv_name, num_files, file_input_unit, file_output_unit in all_datasets:
                # Determine the label based on context
                if num_files > 1 and num_topics > 1:
                    # Multiple files AND multiple topics: use "filename: topic"
                    label = f"{csv_name}: {topic_name}"
                elif num_files > 1 and num_topics == 1:
                    # Multiple files, single topic: use filename only
                    label = csv_name
                elif num_files == 1 and num_topics > 1:
                    # Single file, multiple topics: use topic name only
                    label = topic_name
                else:
                    # Single file, single topic: no label
                    label = None

                # Determine title
                if args.title:
                    title = " ".join(args.title)
                else:
                    title = topic_name

                final_datasets.append((title, ts, label, file_input_unit, file_output_unit))

            # Override labels if custom labels provided
            if args.labels:
                if len(args.labels) != len(final_datasets):
                    raise ValueError(
                        f"Number of custom labels ({len(args.labels)}) must match "
                        f"number of datasets ({len(final_datasets)})"
                    )
                # Replace labels with custom ones
                final_datasets = [
                    (title, ts, custom_label, file_input_unit, file_output_unit)
                    for (title, ts, _, file_input_unit, file_output_unit), custom_label in zip(final_datasets, args.labels)
                ]

            # Plot all datasets with different colors and sizes
            for idx, (title, ts, label, file_input_unit, file_output_unit) in enumerate(final_datasets):
                # Use different color for each dataset
                color = colors[idx % len(colors)]

                # Calculate size - larger for earlier datasets (lower z-index)
                # Start at 50 and decrease by 15% for each subsequent dataset
                base_size = 50
                size = base_size * (0.85 ** idx) if len(final_datasets) > 1 else None

                plot_ts(
                    ts,
                    title,
                    args.plot,
                    file_input_unit,
                    file_output_unit,
                    Decimal(str(args.expected)),
                    rm_outliers=args.rm_outliers,
                    color=color,
                    label=label,
                    size=size,
                    force_title=args.force_title
                )


            # Add legend if multiple datasets or if custom labels were provided
            if len(final_datasets) > 1 or args.labels:
                plt.legend()  # pyright: ignore[reportUnknownMemberType]

            # Save and close plot
            plt.savefig(output_file)  # pyright: ignore[reportUnknownMemberType]
            plt.close()
    finally:
        # Close input files (only in non-stats mode, as stats mode already processed them)
        if not args.stats:
            for input_file in input_files:
                if input_file != sys.stdin:
                    input_file.close()


if __name__ == "__main__":
    main()
