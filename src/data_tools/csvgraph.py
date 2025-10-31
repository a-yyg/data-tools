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
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, TextIO, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.signal import detrend

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

    def convert(self, unit: TimeUnit) -> Callable[[float], float]:
        def factor(value: float) -> float:
            return value * (10 ** (self.unit.value[0] - unit.value[0]))

        return factor


class TimestampAnalyzer:
    def __init__(self, ts: npt.NDArray[Any], unit: TimeUnit = TimeUnit.NANOSECONDS):
        self.ts = ts
        self.ts_diff = None
        self.ts_detrended = None
        self.ts_drift = None
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

    def linear(self):
        self._plot = "ts"
        return self

    def diff(self):
        self.ts_diff = np.diff(self.ts)
        self._plot = "ts_diff"
        return self

    def detrend(self):
        # Use scipy's detrend function to remove linear trend
        self.ts_detrended = detrend(self.ts, type="linear")
        self._plot = "ts_detrended"
        return self

    def drift(self, expected_slope: float):
        x = np.arange(len(self.ts))
        linear_trend = expected_slope * x + self.ts[0]
        self.ts_drift = self.ts - linear_trend
        # self.trend = (expected_slope, self.ts[0])
        self._plot = "ts_drift"
        return self

    def plot(self, filepath: TextIO, title: str):
        y = self.__dict__[self._plot]
        y = np.delete(y, self.hidden)
        y = np.array(list(map(self.factor.convert(self.unit), y)))
        x = np.arange(len(y))
        plt.scatter(x, y)  # pyright: ignore[reportUnknownMemberType]
        # if "trend" in self.__dict__:
        #     plt.plot(x, self.trend[0] * x + self.trend[1])  # pyright: ignore[reportUnknownMemberType]
        #     plt.legend([f"Trend (y = {self.trend[0]:.2f} * x + {self.trend[1]:.2f})"])  # pyright: ignore[reportUnknownMemberType]
        plt.title(title)  # pyright: ignore[reportUnknownMemberType]
        plt.xlabel("Index")  # pyright: ignore[reportUnknownMemberType]
        plt.ylabel(f"Timestamp ({self.unit.value[1]})")  # pyright: ignore[reportUnknownMemberType]
        plt.savefig(filepath)  # pyright: ignore[reportUnknownMemberType]
        plt.close()
        return self

    def outliers(self, thresh: float) -> List[Tuple[int, float]]:
        y = self.__dict__[self._plot]
        mean = np.mean(y)
        std = np.std(y)
        outliers: List[Tuple[int, float]] = []
        for i, val in enumerate(y):
            if abs(val - mean) > thresh * std:
                outliers.append((i, val))
        return outliers

    def hide_indices(self, indices: List[int]) -> None:
        self.hidden = indices


def parse_csv(reader) -> Tuple[str, List[float]]:
    ts: List[float] = []
    title: str = ""
    for row in reader:
        if not row:
            continue
        if title == "":
            title = row[1]
            assert(isinstance(title, str))
            continue
        ts.append(float(row[1]))
    return title, ts


def analyze_ts(
    ts: List[float],
    title: str,
    plot: str,
    output_file: TextIO,
    unit: TimeUnit,
    expected: float,
    # output_dir: str,
    rm_outliers: int,
):
    analyzer = TimestampAnalyzer(np.array(ts), unit)
    # base_name = csv_file.stem
    # ts_outliers = np.array(ts)[outliers]
    # print(f"Outliers for {base_name}: {ts_outliers}")
    if rm_outliers > 0:
        outliers = analyzer.diff().outliers(rm_outliers)
        outliers = [i for (i, _) in outliers]
        analyzer.hide_indices(outliers)
    match plot:
        case "linear":
            title = f"{title} Timestamp"
            analyzer.linear().sec().plot(filepath=output_file, title=title)
        case "diff":
            title = f"{title} Difference"
            analyzer.diff().msec().plot(filepath=output_file, title=title)
        case "detrend":
            title = f"{title} Detrended"
            analyzer.detrend().msec().plot(filepath=output_file, title=title)
        case "drift":
            title = f"{title} Drift"
            analyzer.drift(UnitFactor(TimeUnit.MILLISECONDS).convert(unit)(expected)).msec().plot(filepath=output_file, title=title)
        case _:
            raise ValueError(f"Invalid plot type: {plot}")
    # analyzer.linear().plot(
    #     Path(output_dir, f"{base_name}_timestamp_plot.png"),
    #     title=f"{title} Timestamp",
    # ).diff().plot(
    #     Path(output_dir, f"{base_name}_timestamp_diff_plot.png"),
    #     title=f"{title} Difference",
    # ).detrend().plot(
    #     Path(output_dir, f"{base_name}_timestamp_detrended_plot.png"),
    #     title=f"{title} Detrended",
    # ).drift(
    #     expected * 1e6
    # ).plot(
    #     Path(output_dir, f"{base_name}_timestamp_drift_plot.png"),
    #     title=f"{title} - ({expected} ms) * x",
    # )


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
        "-t", "--topics", type=str, nargs="+", help="Topic(s) to analyze"
    )
    parser.add_argument("--title", type=str, nargs="+", help="Title of the plot")
    parser.add_argument(
        "-p", "--plot", type=str, choices=["linear", "diff", "detrend", "drift"]
    )
    parser.add_argument(
        "-e",
        "--expected",
        type=float,
        help="Expected trend in ms (usable with drift plot)",
        default=200,
    )
    parser.add_argument(
        "-r", "--rm_outliers", type=float, help="Remove outliers above this many std deviations",
        default=0
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
    title, ts = parse_csv(reader)
    if args.title:
        title = " ".join(args.title)
    analyze_ts(
        ts,
        title,
        args.plot,
        output_file,
        units[args.unit],
        args.expected,
        rm_outliers=args.rm_outliers,
    )


if __name__ == "__main__":
    main()
