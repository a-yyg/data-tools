# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "argparse",
#   "rosbags"
# ]
# ///

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from .utils import Timestamp, get_type, to_stamp


def read_bag(bag_file: Path, topics: [str]) -> Dict[str, List[Timestamp]]:
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([bag_file], default_typestore=typestore) as reader:
        ts: Dict[str, List[Timestamp]] = {}
        for topic in topics:
            ts[topic] = []
            connections = [x for x in reader.connections if x.topic == topic]
            # print(f"Connections: {connections}")
            for connection, timestamp, rawdata in reader.messages(
                connections=connections
            ):
                msg = reader.deserialize(rawdata, connection.msgtype)
                ts[topic].append(to_stamp(msg))
        return ts


def list_bag_topics(bag_file: Path) -> List[str]:
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([bag_file], default_typestore=typestore) as reader:
        topics = [x.topic for x in reader.connections]
        return topics


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze timestamps")
    parser.add_argument("-f", "--file", help="Bag file to analyze")
    parser.add_argument(
        "-t", "--topics", type=str, nargs="+", help="Topic(s) to analyze"
    )
    parser.add_argument(
        "-l", "--list", action="store_true", help="List available topics"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.file:
        print(f"A rosbag file must be provided")
        sys.exit(1)
    if args.list:
        print(f"Available topics: {list_bag_topics(Path(args.file))}")
        sys.exit(0)
    if not args.topics:
        args.topics = list_bag_topics(Path(args.file))
    data = read_bag(
        Path(args.file),
        args.topics,
    )
    spamwriter = csv.writer(sys.stdout)
    topics = list(data.keys())
    n = max(len(data[topic]) for topic in topics)
    spamwriter.writerow(["Index", *topics])
    for i in range(n):
        row = [i]
        for topic in topics:
            timestamps = data[topic]
            if i < len(timestamps):
                row.append(timestamps[i].to_ns())
            else:
                row.append("")
        spamwriter.writerow(row)


if __name__ == "__main__":
    main()
