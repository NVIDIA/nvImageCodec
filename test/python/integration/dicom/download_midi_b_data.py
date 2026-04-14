

import os
import functools
import time
from pathlib import Path

from tcia_utils import nbia


# MIDI-B PHI Detection test / validation data sets.
collection_names = [
    "MIDI-B-Curated-Test",
    "MIDI-B-Curated-Validation",
    "MIDI-B-Synthetic-Test",
    "MIDI-B-Synthetic-Validation"
]

# Transient error substrings that warrant a retry
DOWNLOAD_RETRY_ERRORS = (
    "Response ended prematurely",
    "Connection reset",
    "Connection refused",
    "timed out",
    "Timeout",
    "Connection aborted",
    "Remote end closed",
)
MAX_DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_RETRY_DELAY_SEC = 5

@functools.lru_cache(maxsize=1)
def get_all_series():
    """
    Return list of all series from all MIDI-B collections, caching the result.
    
    Returns:
        List of series dictionaries, each containing SeriesInstanceUID and other metadata
    """
    all_series = []
    for collection_name in collection_names:
        series = nbia.getSeries(collection=collection_name)
        if series is None:
            print(f"{collection_name}: No series found")
            continue
        all_series.extend(series)
    return all_series

def get_series_by_uid(series_instance_uid):
    """
    Find a specific series by SeriesInstanceUID.
    
    Args:
        series_instance_uid: The SeriesInstanceUID to find
        
    Returns:
        Series dictionary or None if not found
    """
    all_series = get_all_series()
    for series in all_series:
        if series.get('SeriesInstanceUID') == series_instance_uid:
            return series
    return None

def _zip_path_for_series(out_dir: str, series_uid: str):
    """Return path to the zip for this series if it exists, else None."""
    out_path = Path(out_dir)
    if not out_path.exists():
        return None
    matches = list(out_path.rglob(f"*{series_uid}*.zip"))
    return str(matches[0]) if matches else None


def download_series(series):
    """
    Download a series as a zip file.
    
    Args:
        series: Either a series dictionary or a SeriesInstanceUID string
        
    Returns:
        Path to downloaded zip file, or None if download failed
    """
    # If series is a string, look it up
    if isinstance(series, str):
        series_uid = series
        series = get_series_by_uid(series_uid)
        if series is None:
            print(f"Series {series_uid} not found in any collection")
            return None
    else:
        series_uid = series.get("SeriesInstanceUID")
        if not series_uid:
            print("Series has no SeriesInstanceUID")
            return None

    out_dir = os.path.join(os.path.dirname(__file__), ".midi_b_data")
    os.makedirs(out_dir, exist_ok=True)

    last_error = None
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            nbia.downloadSeries([series], as_zip=True, path=out_dir)
            zip_path = _zip_path_for_series(out_dir, series_uid)
            if zip_path:
                return zip_path
            last_error = RuntimeError("Download completed but no zip file found")
            if attempt < MAX_DOWNLOAD_ATTEMPTS:
                print(
                    f"Attempt {attempt}/{MAX_DOWNLOAD_ATTEMPTS} failed for series {series_uid} "
                    f"(download completed but no zip file found); "
                    f"retrying in {DOWNLOAD_RETRY_DELAY_SEC}s..."
                )
                time.sleep(DOWNLOAD_RETRY_DELAY_SEC)
            else:
                print(f"Error downloading series {series_uid}: {last_error}")
                return None
        except Exception as e:
            last_error = e
            msg = str(e)
            if attempt < MAX_DOWNLOAD_ATTEMPTS and any(
                err in msg for err in DOWNLOAD_RETRY_ERRORS
            ):
                print(
                    f"Attempt {attempt}/{MAX_DOWNLOAD_ATTEMPTS} failed ({msg}); "
                    f"retrying in {DOWNLOAD_RETRY_DELAY_SEC}s..."
                )
                time.sleep(DOWNLOAD_RETRY_DELAY_SEC)
            else:
                print(f"Error downloading series {series_uid}: {e}")
                return None
    if last_error:
        print(f"Error downloading series {series_uid}: {last_error}")
    return None

def download_all():
    """Download all available series from MIDI-B collections."""
    for series in get_all_series():
        download_series(series)

if __name__ == "__main__":
    download_all()