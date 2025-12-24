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

    def plot(self, filepath: TextIO, title: str):
        y = self.__dict__[self._plot]
        # Remove hidden indices
        y_filtered = [y[i] for i in range(len(y)) if i not in self.hidden]
        # Convert units and then to float for plotting
        y_converted = [float(self.factor.convert(self.unit)(val)) for val in y_filtered]
        x = np.arange(len(y_converted))
        plt.scatter(x, y_converted)  # pyright: ignore[reportUnknownMemberType]
        # if "trend" in self.__dict__:
        #     plt.plot(x, self.trend[0] * x + self.trend[1])  # pyright: ignore[reportUnknownMemberType]
        #     plt.legend([f"Trend (y = {self.trend[0]:.2f} * x + {self.trend[1]:.2f})"])  # pyright: ignore[reportUnknownMemberType]
        plt.title(title)  # pyright: ignore[reportUnknownMemberType]
        plt.xlabel("Index")  # pyright: ignore[reportUnknownMemberType]
        plt.ylabel(f"Timestamp ({self.unit.value[1]})")  # pyright: ignore[reportUnknownMemberType]
        plt.savefig(filepath)  # pyright: ignore[reportUnknownMemberType]
        plt.close()
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


def parse_csv(reader) -> Tuple[str, List[Decimal]]:
    ts: List[Decimal] = []
    title: str = ""
    for row in reader:
        if not row:
            continue
        if title == "":
            title = row[1]
            assert isinstance(title, str)
            continue
        ts.append(Decimal(row[1]))
    return title, ts


def analyze_ts(
    ts: List[Decimal],
    plot: str,
    unit: TimeUnit,
    expected: Decimal,
    rm_outliers: int,
) -> List[Decimal]:
    analyzer = TimestampAnalyzer(ts, unit)
    if rm_outliers > 0:
        outliers = analyzer.diff().outliers(rm_outliers)
        outliers = [i for (i, _) in outliers]
        analyzer.hide_indices(outliers)
    match plot:
        case "linear":
            return analyzer.linear().sec().t
        case "diff":
            return analyzer.diff().msec().t
        case "mdiff":
            return (
                analyzer.mdiff(
                    UnitFactor(TimeUnit.MILLISECONDS).convert(unit)(expected)
                )
                .msec()
                .t
            )
        case "detrend":
            return analyzer.detrend().msec().t
        case "drift":
            return (
                analyzer.drift(
                    UnitFactor(TimeUnit.MILLISECONDS).convert(unit)(expected)
                )
                .msec()
                .t
            )
        case _:
            raise ValueError(f"Invalid plot type: {plot}")


def plot_ts(
    ts: List[Decimal],
    title: str,
    plot: str,
    output_file: TextIO,
    from_unit: TimeUnit,
    to_unit: TimeUnit,
    expected: Decimal,
    # output_dir: str,
    rm_outliers: int,
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

    match plot:
        case "linear":
            title = f"{title} Timestamp"
            analyzer.linear().plot(filepath=output_file, title=title)
        case "diff":
            title = f"{title} Difference"
            analyzer.diff().plot(filepath=output_file, title=title)
        case "mdiff":
            title = f"{title} Difference"
            analyzer.mdiff(
                UnitFactor(from_unit).convert(to_unit)(expected)
            ).plot(filepath=output_file, title=title)
        case "detrend":
            title = f"{title} Detrended"
            analyzer.detrend().plot(filepath=output_file, title=title)
        case "drift":
            title = f"{title} Drift"
            analyzer.drift(
                UnitFactor(from_unit).convert(to_unit)(expected)
            ).plot(filepath=output_file, title=title)
        case _:
            raise ValueError(f"Invalid plot type: {plot}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze timestamps")
    parser.add_argument("CSV", help="CSV file to analyze", nargs="?", default=None)
    parser.add_argument(
        "-u",
        "--unit",
        choices=["s", "ms", "us", "ns"],
        default="ns",
        help="Time unit",
    )
    parser.add_argument(
        "--to-unit",
        choices=["s", "ms", "us", "ns"],
        default="ms",
        help="Time unit",
    )
    parser.add_argument(
        "-t", "--topics", type=str, nargs="+", help="Topic(s) to analyze"
    )
    parser.add_argument("--title", type=str, nargs="+", help="Title of the plot")
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
    # parser.add_argument("-o", "--output_dir", type=str, help="Output directory")
    parser.add_argument("--stdin", action="store_true", help="Input from stdin")
    parser.add_argument("--stdout", action="store_true", help="Output to stdout")
    parser.add_argument(
        "--stdio", action="store_true", help="Input and output from stdin"
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
    if args.stdin or args.stdio:
        input_file = sys.stdin
    if args.stdout or args.stdio:
        output_file = sys.stdout
    else:
        raise ValueError("Not implemented")
    reader = csv.reader(input_file)
    # Trim data after header according to start and end index
    if args.start_index is not None:
        reader = itertools.islice(reader, args.start_index, None)
    if args.end_index is not None:
        reader = itertools.islice(reader, args.end_index)

    if args.tocsv:
        writer = csv.writer(output_file)
        title, ts = parse_csv(reader)
        t = analyze_ts(
            ts,
            args.plot,
            units[args.unit],
            Decimal(str(args.expected)),
            rm_outliers=args.rm_outliers,
        )
        header = f"{title} {args.plot}"
        writer.writerow([header])
        for value in t:
            writer.writerow([str(value)])
    else:
        title, ts = parse_csv(reader)
        if args.title:
            title = " ".join(args.title)
        plot_ts(
            ts,
            title,
            args.plot,
            output_file,
            units[args.unit],
            units[args.to_unit],
            Decimal(str(args.expected)),
            rm_outliers=args.rm_outliers,
        )


if __name__ == "__main__":
    main()
