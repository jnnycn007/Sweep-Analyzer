"""Utility helpers shared by GUI and analysis code.

This module historically depended on `imgui` for an experimental UI. To keep
`inclusive_range` usable without the `imgui` dependency, `imgui` is imported
only when required.
"""

from __future__ import annotations

from typing import Any, List, Tuple, TypeVar

try:
    import imgui  # type: ignore
except Exception:  # pragma: no cover - only needed in imgui-enabled contexts
    imgui = None  # type: ignore

T = TypeVar("T")


def _require_imgui():
    """Return the imgui module or raise a clear ImportError."""
    if imgui is None:
        raise ImportError("imgui is required for editable UI helpers in util.py")
    return imgui

def do_editable_raw(preamble: str, value: Any, units: str = "", width: int = 100) -> Tuple[bool, str]:
    """Render a raw editable text field in imgui and return (changed, value)."""
    im = _require_imgui()
    im.text(preamble)
    im.same_line()
    im.push_item_width(width)
    im.push_id(preamble)
    changed, new_value = im.input_text(units, str(value))
    im.pop_id()
    im.pop_item_width()

    return (changed, new_value)

def do_editable(preamble: str, value: T, units: str = "", width: int = 100, enable: bool = True) -> T:
    """Render an editable field and coerce back to the input value's type."""
    im = _require_imgui()
    if not enable:
        im.text_disabled(f"{preamble} {value} {units}")
        return value

    type_ = type(value)

    _, new_value = do_editable_raw(preamble, value, units, width)

    if new_value != value:
        try:
            new_value = type_(new_value)
        except Exception:
            return value

        return new_value
    return value

def inclusive_range(start: float, stop: float, step: float) -> List[float]:
    """Return an inclusive range, like `range` but including the stop value."""
    if step == 0:
        return []
    out: List[float] = []
    while start <= stop:
        out.append(start)
        start += step
    return out
