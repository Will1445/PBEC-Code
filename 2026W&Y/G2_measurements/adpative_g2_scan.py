"""Standalone HydraHarp/Toptica Lab 4 power scan.

Example:
    python adpative_g2_scan.py --start 33.00 --end 33.40 --step 0.05 --nd ND4

The script skips power points whose .npz file already exists, asks for an
explicit filter/setup confirmation before touching hardware, and saves each
histogram and plot as it is acquired.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "Data"


def parse_args():
    parser = argparse.ArgumentParser(description="Acquire a Lab 4 HydraHarp power scan.")
    parser.add_argument("--start", type=float, default=33.00, help="First laser power in mW (default: 33.00).")
    parser.add_argument("--end", type=float, default=33.40, help="Last laser power in mW (default: 33.40).")
    parser.add_argument("--step", type=float, default=0.05, help="Power increment in mW (default: 0.05).")
    parser.add_argument("--nd", default="ND4", help="ND filter label used in filenames and folder name.")
    parser.add_argument("--acquisition-time", type=float, default=20.0, help="Acquisition time per point in seconds.")
    parser.add_argument("--settling-time", type=float, default=2.0, help="Wait after setting laser power, in seconds.")
    parser.add_argument("--channel", type=int, default=0, help="HydraHarp input channel (default: 0).")
    parser.add_argument("--laser-com-port", default="COM5", help="Toptica laser serial port (default: COM5).")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Base data directory (default: Data folder beside this script).",
    )
    return parser.parse_args()


def validate_args(args):
    if args.step <= 0:
        raise ValueError("Power step must be positive.")
    if args.end < args.start:
        raise ValueError("End power must be at or above start power.")
    if args.acquisition_time <= 0 or args.acquisition_time > 60:
        raise ValueError("Acquisition time must be greater than 0 and no more than 60 seconds.")
    if args.settling_time < 0:
        raise ValueError("Settling time cannot be negative.")
    if not args.nd.strip() or any(char in args.nd for char in r'<>:"/\|?*'):
        raise ValueError("ND label must be non-empty and cannot contain filename-reserved characters.")


def make_data_folder(base_directory: Path, nd_label: str) -> Path:
    now = datetime.now()
    folder = (
        base_directory
        / now.strftime("%Y")
        / now.strftime("%Y%m")
        / now.strftime("%Y%m%d")
        / "lab4"
        / nd_label
    )
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def measure_histogram(hydra, seconds: float, channel: int):
    if not 0 <= channel < hydra.channels:
        raise ValueError(
            f"HydraHarp channel {channel} is unavailable; this unit has {hydra.channels} input channels."
        )

    hydra.start(seconds)
    try:
        deadline = time.monotonic() + seconds + 10.0
        while not hydra.complete():
            if time.monotonic() > deadline:
                raise TimeoutError("HydraHarp acquisition timed out.")
            time.sleep(0.05)
    finally:
        result = hydra.stop_read()

    counts = np.asarray(result["histograms"][channel], dtype=np.uint32)
    resolution_ps = float(hydra.settings["resolution_ps"])
    delay_ns = np.arange(counts.size, dtype=float) * resolution_ps / 1000.0
    return delay_ns, counts, result


def main():
    args = parse_args()
    validate_args(args)

    powers = np.round(
        args.start + np.arange(int(round((args.end - args.start) / args.step)) + 1) * args.step,
        2,
    )
    if not np.isclose(powers[-1], args.end, atol=1e-8):
        raise ValueError("The selected step does not land on the end power; adjust --end or --step.")

    folder = make_data_folder(args.data_dir, args.nd.strip())
    pending = []
    for power in powers:
        name = f"{power:.2f}mW_{args.nd}"
        npz_path = folder / f"{name}.npz"
        if npz_path.is_file():
            print(f"Skipping existing: {npz_path}")
        else:
            pending.append((float(power), name))

    print("LAB4 POWER SCAN")
    print(f"Power: {args.start:.2f}–{args.end:.2f} mW, step {args.step:.2f} mW")
    print(f"ND filter: {args.nd}; channel: {args.channel}")
    print(f"Acquisition: {args.acquisition_time:g} s per point")
    print(f"New measurements: {len(pending)}")
    print(f"Saving to: {folder}")
    if not pending:
        print("No new measurements to acquire.")
        return 0

    confirmation = input(
        f"Confirm {args.nd} is installed and the detector count rate is safe. "
        "Type YES to connect to the laser and HydraHarp and begin: "
    )
    if confirmation != "YES":
        print("Scan cancelled; no hardware was connected.")
        return 1

    # This file lives under 2026W&Y/G2_measurements, so add the 2026W&Y
    # directory to import the shared HydraHarp adapter with the 64-bit DLL path.
    wy_directory = Path(__file__).resolve().parents[1]
    if str(wy_directory) not in sys.path:
        sys.path.insert(0, str(wy_directory))

    from utils.hydraharp import HydraHarp
    from pylablib.devices import Toptica

    hydra = None
    laser = None
    try:
        hydra = HydraHarp()
        laser = Toptica.TopticaIBeam(args.laser_com_port)
        print("HydraHarp and Toptica laser connected.")

        for index, (power, name) in enumerate(pending, start=1):
            print(f"[{index}/{len(pending)}] Measuring {power:.2f} mW", flush=True)
            laser.set_channel_power(1, power / 1000.0)
            time.sleep(args.settling_time)

            delay_ns, counts, _result = measure_histogram(
                hydra, seconds=args.acquisition_time, channel=args.channel
            )

            npz_path = folder / f"{name}.npz"
            png_path = folder / f"{name}.png"
            if npz_path.exists():
                raise FileExistsError(f"Refusing to overwrite: {npz_path}")
            np.savez(
                npz_path,
                delay_ns=delay_ns,
                counts=counts,
                power_mW=power,
                ND=args.nd,
                acquisition_time_s=args.acquisition_time,
                channel=args.channel,
            )
            print(f"Saved: {npz_path}", flush=True)

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(delay_ns, counts)
            ax.set_xlabel("Instrument delay (ns)")
            ax.set_ylabel("Coincidence counts / bin")
            ax.set_title(f"{power:.2f} mW | {args.nd}")
            fig.tight_layout()
            fig.savefig(png_path, dpi=150)
            plt.show()
            plt.close(fig)
            print(f"Finished {power:.2f} mW", flush=True)

        print(f"LAB4 SCAN COMPLETE. Data saved to: {folder}")
        return 0
    finally:
        if hydra is not None:
            hydra.close()
        if laser is not None:
            close = getattr(laser, "close", None)
            if callable(close):
                close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Scan failed: {exc}", file=sys.stderr)
        raise
