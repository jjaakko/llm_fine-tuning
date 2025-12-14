"""Interface for get tracking URI either from local or cloud."""

from typing import Protocol


class TrackerURI(Protocol):
    def get_tracker_uri(self):
        raise NotImplementedError(type(self))
