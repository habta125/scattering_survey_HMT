from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_profile_from_txt(filename: str | Path) -> dict[str, Any]:
    """
    Load a pulse profile from a text file where the 4th column
    contains the profile intensity.
    """
    filename = Path(filename)
    data = np.loadtxt(filename, comments="#")

    if data.ndim == 1:
        raise ValueError(f"{filename} does not look like a multi-column profile file")
    if data.shape[1] < 4:
        raise ValueError(f"{filename} must have at least 4 columns")

    profile = np.asarray(data[:, 3], dtype=float)

    return {
        "profile": profile,
        "nbin": len(profile),
        "source_format": "txt",
        "filename": str(filename),
        "freq_mhz": np.nan,
        "bandwidth_mhz": np.nan,
        "mjd_start": np.nan,
        "period_sec": np.nan,
    }


def load_profile_from_ar(
    filename: str | Path,
    dedisperse: bool = True,
    remove_baseline: bool = False,
    tscrunch: bool = True,
    fscrunch: bool = True,
    pscrunch: bool = True,
) -> dict[str, Any]:
    """
    Load a processed PSRCHIVE .ar file and return a 1D profile plus metadata.
    """
    filename = Path(filename)

    try:
        import psrchive
    except ImportError as exc:
        raise ImportError(
            "psrchive is required to read .ar files. "
            "Install PSRCHIVE and make sure Python can import psrchive."
        ) from exc

    arch = psrchive.Archive_load(str(filename))

    if dedisperse:
        try:
            arch.dedisperse()
        except Exception:
            pass

    if remove_baseline:
        try:
            arch.remove_baseline()
        except Exception:
            pass

    if pscrunch:
        try:
            arch.pscrunch()
        except Exception:
            pass

    if fscrunch:
        try:
            arch.fscrunch()
        except Exception:
            pass

    if tscrunch:
        try:
            arch.tscrunch()
        except Exception:
            pass

    data = arch.get_data()

    # Expected shape: (subint, pol, chan, bin)
    if data.ndim != 4:
        raise ValueError(f"Unexpected archive data shape for {filename}: {data.shape}")

    profile = np.asarray(data[0, 0, 0, :], dtype=float)

    try:
        freq_mhz = float(arch.get_centre_frequency())
    except Exception:
        freq_mhz = np.nan

    try:
        bandwidth_mhz = float(arch.get_bandwidth())
    except Exception:
        bandwidth_mhz = np.nan

    try:
        mjd_start = float(arch.start_time().in_days())
    except Exception:
        mjd_start = np.nan

    period_sec = np.nan
    try:
        period_sec = float(arch.get_Integration(0).get_folding_period())
    except Exception:
        pass

    if not np.isfinite(period_sec):
        try:
            nsub = float(arch.get_nsubint())
            if nsub > 0:
                period_sec = float(arch.integration_length()) / nsub
        except Exception:
            period_sec = np.nan

    return {
        "profile": profile,
        "nbin": len(profile),
        "source_format": "ar",
        "filename": str(filename),
        "freq_mhz": freq_mhz,
        "bandwidth_mhz": bandwidth_mhz,
        "mjd_start": mjd_start,
        "period_sec": period_sec,
    }


def load_profile(filename: str | Path, **kwargs) -> dict[str, Any]:
    """
    Automatically detect file type and load the profile.
    Supports:
      - .txt
      - .ar
    """
    filename = Path(filename)
    suffix = filename.suffix.lower()

    if suffix == ".txt":
        return load_profile_from_txt(filename)

    if suffix == ".ar":
        return load_profile_from_ar(filename, **kwargs)

    raise ValueError(f"Unsupported file type: {filename}")


def load_metadata(metadata_csv: str | Path) -> pd.DataFrame:
    """
    Required columns:
      - name
      - filename

    Optional columns:
      - period_sec
      - freq_mhz
    """
    df = pd.read_csv(metadata_csv)

    required = {"name", "filename"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv missing required columns: {sorted(missing)}")

    if "period_sec" not in df.columns:
        df["period_sec"] = np.nan

    if "freq_mhz" not in df.columns:
        df["freq_mhz"] = np.nan

    return df
