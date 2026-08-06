"""
Debugging utilities for the WSmart+ Route framework.

This module provides tools for monitoring runtime state, including:
- Variable watching (tracing value changes).

Attributes:
    watch (callable): Watch a variable by name. Prints every change with line number.
    watch_all (callable): Watch all local variables in the caller's frame. Prints every change with line number.

Example:
    >>> from logic.src.utils.validation.debug_utils import watch
    >>> watch("my_variable")
    >>> watch_all()
"""

import sys
import traceback
from types import FrameType
from typing import Any, Callable, Optional


def watch(
    var_name: str,
    callback: Optional[Callable[[Any, Any, FrameType], None]] = None,
    *,
    frame_depth: int = 1,
) -> None:
    """
    Watch a variable by name. Prints every change with line number.
    Works with numpy arrays, lists, dicts, primitives – anything.

    Args:
        var_name (str): Name of the variable to watch.
        callback (callable, optional): Custom callback (old, new, frame) -> None.
        frame_depth (int, optional): Stack depth to watch from. Defaults to 1.

    Raises:
        NameError: If variable is not found in locals or globals.

    Returns:
        None
    """
    caller_frame = sys._getframe(frame_depth)
    if callback is None:

        def default_callback(old: Any, new: Any, frame: FrameType) -> None:
            """
            Default callback function that is called for each change in a variable.

            Args:
                old (Any): The old value of the variable.
                new (Any): The new value of the variable.
                frame (FrameType): The frame in which the variable changed.
            """
            stack = traceback.extract_stack(frame)[:-1]
            caller = stack[-1]
            print(f"DEBUG → {var_name}: {old} → {new}")
            print(f"        at {caller.filename}:{caller.lineno} in {caller.name}")

        # 2. Use a final variable that is NOT Optional
        actual_callback: Callable[[Any, Any, FrameType], None] = default_callback
    else:
        actual_callback = callback

    old_value = [None]

    def tracer(frame: FrameType, event: str, arg: Any) -> Callable:
        """
        Function tracer.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: Description of return value.
        """
        if event != "line" or frame is not caller_frame:
            return tracer

        try:
            new_value = frame.f_locals[var_name]
        except KeyError:
            new_value = frame.f_globals.get(var_name)

        if new_value is not old_value[0]:
            # Now Pyrefly knows actual_callback is definitely a Callable
            actual_callback(old_value[0], new_value, frame)
            old_value[0] = new_value
        return tracer

    try:
        initial = caller_frame.f_locals[var_name]
    except KeyError:
        try:
            initial = caller_frame.f_globals[var_name]
        except KeyError as e:
            raise NameError(f"Variable '{var_name}' not found in local or global scope") from e

    old_value[0] = initial
    sys.settrace(tracer)
    print(f"Watching variable: '{var_name}' (Initial: {initial})")


def watch_all(
    callback: Optional[Callable[[str, Any, Any, FrameType], None]] = None,
    *,
    frame_depth: int = 1,
) -> None:
    """
    Watch all local variables in the caller's frame. Prints every change with line number.

    Args:
        callback (callable, optional): Custom callback (var_name, old, new, frame) -> None.
        frame_depth (int, optional): Stack depth to watch from. Defaults to 1.

    Raises:
        NameError: If no variables are found in the caller's frame.

    Returns:
        None
    """
    caller_frame = sys._getframe(frame_depth)
    # 1. Establish the non-optional callback
    if callback is None:

        def default_callback(var_name: str, old: Any, new: Any, frame: FrameType) -> None:
            """
            Default callback function that is called for each change in a variable.

            Args:
                var_name (str): The name of the variable that changed.
                old (Any): The old value of the variable.
                new (Any): The new value of the variable.
                frame (FrameType): The frame in which the variable changed.
            """
            stack = traceback.extract_stack(frame)[:-1]
            caller = stack[-1]
            print(f"DEBUG → {var_name}: {old} → {new}")
            print(f"        at {caller.filename}:{caller.lineno} in {caller.name}")

        actual_callback: Callable[[str, Any, Any, FrameType], None] = default_callback
    else:
        actual_callback = callback

    old_locals = dict(caller_frame.f_locals)

    def tracer(frame: FrameType, event: str, arg: Any) -> Callable:
        """
        Tracer function that is called for each line of code executed.

        Args:
            frame (FrameType): The current frame.
            event (str): The event that triggered the tracer.
            arg (Any): The argument that triggered the tracer.

        Returns:
            Callable: The tracer function.
        """
        if event != "line" or frame is not caller_frame:
            return tracer

        current_locals = frame.f_locals
        for name, new_value in current_locals.items():
            if name not in old_locals:
                # 2. Use the guaranteed non-optional callback
                actual_callback(name, None, new_value, frame)
                old_locals[name] = new_value
            elif old_locals[name] is not new_value:
                actual_callback(name, old_locals[name], new_value, frame)
                old_locals[name] = new_value
        return tracer

    sys.settrace(tracer)
    print("Watching all local variables.")
