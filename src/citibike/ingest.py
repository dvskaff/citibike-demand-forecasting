"""Download NYC Citi Bike trip files and condense them into daily ride counts.

The public bucket (https://s3.amazonaws.com/tripdata) holds one zip per *year* for older data and one zip
per *month* for recent data. Zips can contain CSVs, folders of CSVs, or further zips, so the reader below
walks them recursively. Each CSV is aggregated with DuckDB straight to daily counts, so the
multi-gigabyte raw files never need to sit in memory, and every source file's daily counts are cached
as a tiny Parquet file so later runs only download what is new.
"""

from __future__ import annotations

import io
import logging
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import duckdb
import polars as pl
import requests

log = logging.getLogger(__name__)

YEARLY_RE = re.compile(r"^(\d{4})-citibike-tripdata\.zip$")
MONTHLY_RE = re.compile(r"^(\d{4})(\d{2})-citibike-tripdata(?:\.csv)?\.zip$")
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

DAILY_SCHEMA = {
    "date": pl.Date,
    "member_casual": pl.Utf8,
    "rideable_type": pl.Utf8,
    "trips": pl.Int64,
}


@dataclass(frozen=True)
class Source:
    key: str  # file name in the bucket
    year: int
    month: int | None  # None for a whole-year zip
    size: int = 0

    @property
    def cache_name(self) -> str:
        return self.key.replace(".zip", "").replace(".csv", "") + ".parquet"


# --------------------------------------------------------------------------- listing & selection
def list_bucket(bucket_url: str, session: requests.Session | None = None) -> list[tuple[str, int]]:
    """Return (key, size) for every object in the public S3 bucket (handles pagination)."""
    session = session or requests.Session()
    keys: list[tuple[str, int]] = []
    marker = ""
    while True:
        resp = session.get(bucket_url + "/", params={"marker": marker} if marker else None, timeout=60)
        resp.raise_for_status()
        root = ElementTree.fromstring(resp.content)
        for item in root.iter(f"{S3_NS}Contents"):
            key = item.findtext(f"{S3_NS}Key")
            size = int(item.findtext(f"{S3_NS}Size") or 0)
            keys.append((key, size))
        if root.findtext(f"{S3_NS}IsTruncated") == "true" and keys:
            marker = keys[-1][0]
        else:
            return keys


def parse_sources(keys: list[tuple[str, int]]) -> list[Source]:
    """Keep only NYC system files (Jersey City files start with 'JC-')."""
    out = []
    for key, size in keys:
        if m := MONTHLY_RE.match(key):
            out.append(Source(key, int(m.group(1)), int(m.group(2)), size))
        elif m := YEARLY_RE.match(key):
            out.append(Source(key, int(m.group(1)), None, size))
    return out


def select_sources(sources: list[Source], start_year: int, start_month: int) -> list[Source]:
    """Pick a non-overlapping set of files covering start..latest.

    If a whole-year zip exists it is used for that year and that year's monthly zips are skipped, so no
    ride is counted twice while the operator is in the middle of consolidating months into a year file.
    """
    yearly = {s.year: s for s in sources if s.month is None}
    chosen: list[Source] = []
    for s in sources:
        if s.year < start_year:
            continue
        if s.month is None:
            chosen.append(s)
        elif s.year not in yearly and (s.year, s.month) >= (start_year, start_month):
            chosen.append(s)
    return sorted(chosen, key=lambda s: (s.year, s.month or 0))


# --------------------------------------------------------------------------- aggregation
def _find_column(columns: list[str], *candidates: str) -> str | None:
    lower = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def aggregate_csv(path: Path, con: duckdb.DuckDBPyConnection | None = None) -> pl.DataFrame:
    """Aggregate one trip CSV to rides per day x rider type x bike type."""
    con = con or duckdb.connect()
    src = str(path).replace("'", "''")
    cols = [
        r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_csv('{src}', all_varchar=true)").fetchall()
    ]
    start_col = _find_column(cols, "started_at", "starttime", "start time")
    if start_col is None:
        raise ValueError(f"No start-time column in {path.name}: {cols}")
    member_col = _find_column(cols, "member_casual", "usertype", "user type")
    bike_col = _find_column(cols, "rideable_type")

    member_expr = (
        f"""CASE WHEN lower("{member_col}") IN ('member', 'subscriber') THEN 'member'
                 WHEN lower("{member_col}") IN ('casual', 'customer') THEN 'casual'
                 ELSE 'unknown' END"""
        if member_col
        else "'unknown'"
    )
    bike_expr = f"""coalesce(nullif(lower("{bike_col}"), ''), 'unknown')""" if bike_col else "'unknown'"
    query = f"""
        SELECT try_cast(substr(trim("{start_col}"), 1, 10) AS DATE) AS date,
               {member_expr} AS member_casual,
               {bike_expr} AS rideable_type,
               count(*)::BIGINT AS trips
        FROM read_csv('{src}', all_varchar=true, ignore_errors=true)
        GROUP BY ALL
        HAVING date IS NOT NULL
    """
    return pl.from_arrow(con.execute(query).to_arrow_table()).cast(DAILY_SCHEMA)


def _iter_csvs(zf: zipfile.ZipFile, workdir: Path):
    """Yield paths to every CSV inside a zip, descending into nested zips."""
    for info in zf.infolist():
        name = info.filename
        if info.is_dir() or "__MACOSX" in name or Path(name).name.startswith("."):
            continue
        lower = name.lower()
        if lower.endswith(".zip"):
            with zf.open(info) as fh, zipfile.ZipFile(io.BytesIO(fh.read())) as inner:
                yield from _iter_csvs(inner, workdir)
        elif lower.endswith(".csv"):
            target = workdir / Path(name).name
            with zf.open(info) as fh, open(target, "wb") as out:
                shutil.copyfileobj(fh, out, length=16 * 1024 * 1024)
            yield target
            target.unlink(missing_ok=True)


def aggregate_zip(zip_path: Path) -> pl.DataFrame:
    """Aggregate every CSV in a (possibly nested) zip to daily counts."""
    con = duckdb.connect()
    parts = []
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(zip_path) as zf:
        for csv_path in _iter_csvs(zf, Path(tmp)):
            log.info("  aggregating %s", csv_path.name)
            parts.append(aggregate_csv(csv_path, con))
    if not parts:
        return pl.DataFrame(schema=DAILY_SCHEMA)
    return (
        pl.concat(parts)
        .group_by("date", "member_casual", "rideable_type")
        .agg(pl.col("trips").sum())
        .sort("date")
    )


def download(url: str, dest: Path, session: requests.Session | None = None) -> Path:
    session = session or requests.Session()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as out:
            for chunk in resp.iter_content(chunk_size=16 * 1024 * 1024):
                out.write(chunk)
    return dest


def ingest_trips(bucket_url: str, history_start: str, raw_dir: Path, interim_dir: Path) -> pl.DataFrame:
    """Download (or reuse cached) daily counts for every source file since `history_start`."""
    start_year, start_month = int(history_start[:4]), int(history_start[5:7])
    session = requests.Session()
    sources = select_sources(parse_sources(list_bucket(bucket_url, session)), start_year, start_month)
    if not sources:
        raise RuntimeError("No Citi Bike trip files found in the bucket listing.")
    log.info("Using %d source files: %s ... %s", len(sources), sources[0].key, sources[-1].key)

    interim_dir.mkdir(parents=True, exist_ok=True)
    latest_key = sources[-1].key
    frames = []
    for src in sources:
        cached = interim_dir / src.cache_name
        size_tag = interim_dir / (src.cache_name + f".{src.size}")
        # Re-process if the file changed size upstream, or if it is the newest file (it may be re-issued).
        if cached.exists() and size_tag.exists() and src.key != latest_key:
            frames.append(pl.read_parquet(cached))
            continue
        log.info("Downloading %s (%.0f MB)", src.key, src.size / 1e6)
        zip_path = download(f"{bucket_url}/{src.key}", raw_dir / src.key, session)
        daily = aggregate_zip(zip_path)
        zip_path.unlink(missing_ok=True)
        daily.write_parquet(cached)
        for old in interim_dir.glob(src.cache_name + ".*"):
            old.unlink()
        size_tag.touch()
        frames.append(daily)

    return combine_daily(frames, history_start)


def combine_daily(frames: list[pl.DataFrame], history_start: str) -> pl.DataFrame:
    """Sum the per-file counts and keep complete days from `history_start` on."""
    start = pl.lit(history_start).str.to_date()
    daily = (
        pl.concat(frames)
        .group_by("date", "member_casual", "rideable_type")
        .agg(pl.col("trips").sum())
        .filter(pl.col("date") >= start)
        .sort("date")
    )
    # A month file can contain a few rides that started just before midnight on the next month's first
    # day; drop any trailing day whose volume is under 20% of the prior week (an incomplete day).
    totals = daily.group_by("date").agg(pl.col("trips").sum()).sort("date")
    if totals.height > 8:
        last = totals.row(-1, named=True)
        prior = totals.slice(-8, 7)["trips"].mean()
        if last["trips"] < 0.2 * prior:
            daily = daily.filter(pl.col("date") < last["date"])
    return daily
