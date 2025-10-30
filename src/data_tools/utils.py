# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false

from dataclasses import dataclass


@dataclass
class Timestamp:
    sec: int
    nsec: int

    def to_ns(self) -> int:
        return self.sec * 1000000000 + self.nsec


def get_type(obj: object, msgtype_str: str = "__msgtype__") -> str | None:
    if hasattr(obj, msgtype_str):
        return obj.__dict__[msgtype_str]
    return None


def to_stamp(msg: object) -> Timestamp:
    match get_type(msg):
        case "builtin_interfaces/msg/Time":
            assert isinstance(msg.sec, int)
            assert isinstance(msg.nanosec, int)
            return Timestamp(msg.sec, msg.nanosec)
        case "std_msgs/msg/Time":
            assert hasattr(msg, "data"), "std_msgs/msg/Time must have a data field"
            return to_stamp(msg.data)
        case "sensor_msgs/msg/CompressedImage":
            assert hasattr(msg, "header"), (
                "sensor_msgs/msg/CompressedImage must have a header field"
            )
            assert hasattr(msg.header, "stamp"), (
                "sensor_msgs/msg/CompressedImage header must have a stamp field"
            )
            return to_stamp(msg.header.stamp)
        case _:
            raise ValueError(f"Unsupported message: {msg} ({get_type(msg)})")
