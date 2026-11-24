"""Safely diagnose HydraHarp DLL loading without opening hardware or the laser."""

from __future__ import annotations

import ctypes
import platform
import struct
import sys
from pathlib import Path


DLL_PATH = Path(r"D:\Control\HydraHarp\hhlib.dll")
MACHINE_TYPES = {
    0x014C: "x86 (32-bit)",
    0x8664: "x64 (64-bit)",
    0x01C4: "ARM (32-bit)",
    0xAA64: "ARM64",
}


def pe_machine(path: Path) -> str:
    """Read the architecture marker from a Windows PE file."""
    try:
        with path.open("rb") as file:
            if file.read(2) != b"MZ":
                return "not a Windows PE file (missing MZ header)"
            file.seek(0x3C)
            pe_offset_data = file.read(4)
            if len(pe_offset_data) != 4:
                return "invalid PE file (missing PE header offset)"
            pe_offset = struct.unpack("<I", pe_offset_data)[0]
            file.seek(pe_offset)
            if file.read(4) != b"PE\0\0":
                return "invalid PE file (missing PE signature)"
            machine_data = file.read(2)
            if len(machine_data) != 2:
                return "invalid PE file (missing machine type)"
            machine = struct.unpack("<H", machine_data)[0]
            return f"{MACHINE_TYPES.get(machine, 'unknown architecture')} (0x{machine:04X})"
    except OSError as exc:
        return f"could not read file: {exc}"


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({struct.calcsize('P') * 8}-bit)")
    print(f"Python executable: {sys.executable}")
    print(f"Windows: {platform.platform()}")
    print(f"HydraHarp DLL: {DLL_PATH}")

    if not DLL_PATH.is_file():
        print("Result: DLL file not found at this path.")
        return 1

    print(f"DLL architecture: {pe_machine(DLL_PATH)}")
    try:
        ctypes.WinDLL(str(DLL_PATH))
    except (OSError, AttributeError) as exc:
        print(f"Load result: FAILED: {exc}")
        print("No device was opened; this only asked Windows to load the DLL.")
        return 2

    print("Load result: SUCCESS")
    print("No device was opened; this only asked Windows to load the DLL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
