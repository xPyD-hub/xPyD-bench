"""Output verbosity control (M65).

Provides a ``Verbosity`` enum and a ``VerbosityPrinter`` helper that gates
output based on the active verbosity level.
"""

from __future__ import annotations

import enum
import sys
from typing import IO, Any


class Verbosity(enum.Enum):
    """Output verbosity levels."""

    QUIET = "quiet"
    NORMAL = "normal"
    VERBOSE = "verbose"


class VerbosityPrinter:
    """Gate ``print`` calls by verbosity level.

    Parameters
    ----------
    level:
        Active verbosity level.
    file:
        Output stream (default ``sys.stdout``).
    """

    def __init__(self, level: Verbosity = Verbosity.NORMAL, file: IO[str] | None = None) -> None:
        self.level = level
        self._file = file or sys.stdout

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def quiet(self, *args: Any, **kwargs: Any) -> None:
        """Always printed — even in quiet mode (errors, final result)."""
        kwargs.setdefault("file", self._file)
        print(*args, **kwargs)

    def normal(self, *args: Any, **kwargs: Any) -> None:
        """Printed in *normal* and *verbose* modes (default output)."""
        if self.level in (Verbosity.NORMAL, Verbosity.VERBOSE):
            kwargs.setdefault("file", self._file)
            print(*args, **kwargs)

    def verbose(self, *args: Any, **kwargs: Any) -> None:
        """Printed only in *verbose* mode (extra detail)."""
        if self.level is Verbosity.VERBOSE:
            kwargs.setdefault("file", self._file)
            print(*args, **kwargs)

    # Convenience aliases
    error = quiet
    info = normal
    debug = verbose

    def is_quiet(self) -> bool:
        """Return ``True`` when running in quiet mode."""
        return self.level is Verbosity.QUIET

    def is_verbose(self) -> bool:
        """Return ``True`` when running in verbose mode."""
        return self.level is Verbosity.VERBOSE


def parse_verbosity(value: str | None) -> Verbosity:
    """Parse a string into a ``Verbosity`` enum member.

    Accepts ``"quiet"``, ``"normal"``, or ``"verbose"`` (case-insensitive).
    Returns ``Verbosity.NORMAL`` for ``None``.

    Raises
    ------
    ValueError
        If *value* is not a recognised level.
    """
    if value is None:
        return Verbosity.NORMAL
    try:
        return Verbosity(value.lower())
    except ValueError:
        raise ValueError(
            f"Invalid verbosity level {value!r}. "
            f"Choose from: quiet, normal, verbose."
        ) from None
