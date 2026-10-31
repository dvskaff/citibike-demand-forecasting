import io
import zipfile
from datetime import date

import polars as pl

from citibike.ingest import Source, aggregate_zip, combine_daily, parse_sources, select_sources

NEW_SCHEMA = (
    "ride_id,rideable_type,started_at,ended_at,start_station_name,member_casual\n"
    "a,electric_bike,2025-01-01 08:00:00.123,2025-01-01 08:10:00,X,member\n"
    "b,classic_bike,2025-01-01 09:00:00,2025-01-01 09:30:00,Y,casual\n"
    "c,electric_bike,2025-01-02 07:00:00,2025-01-02 07:05:00,Z,member\n"
)
OLD_SCHEMA = (
    '"tripduration","starttime","stoptime","usertype"\n'
    '600,"2025-01-02 10:00:00","2025-01-02 10:10:00","Subscriber"\n'
    '900,"2025-01-03 11:00:00","2025-01-03 11:15:00","Customer"\n'
)


def _nested_zip(path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("202501-citibike-tripdata_2.csv", OLD_SCHEMA)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("2025-citibike-tripdata/202501-citibike-tripdata_1.csv", NEW_SCHEMA)
        z.writestr("2025-citibike-tripdata/202501-part2.zip", inner.getvalue())
        z.writestr("__MACOSX/._junk.csv", "garbage")
    return path


def test_aggregate_nested_zip_handles_both_schemas(tmp_path):
    daily = aggregate_zip(_nested_zip(tmp_path / "2025-citibike-tripdata.zip"))
    totals = dict(daily.group_by("date").agg(pl.col("trips").sum()).iter_rows())
    assert totals == {date(2025, 1, 1): 2, date(2025, 1, 2): 2, date(2025, 1, 3): 1}
    by_member = dict(daily.group_by("member_casual").agg(pl.col("trips").sum()).iter_rows())
    assert by_member == {"member": 3, "casual": 2}  # Subscriber->member, Customer->casual


def test_parse_and_select_sources_prefers_yearly_files():
    keys = [
        ("2024-citibike-tripdata.zip", 1),
        ("202401-citibike-tripdata.zip", 1),
        ("202501-citibike-tripdata.zip", 1),
        ("202502-citibike-tripdata.csv.zip", 1),
        ("JC-202501-citibike-tripdata.csv.zip", 1),
        ("index.html", 1),
    ]
    sources = parse_sources(keys)
    assert all(not s.key.startswith("JC") for s in sources)
    chosen = select_sources(sources, 2024, 1)
    assert [s.key for s in chosen] == [
        "2024-citibike-tripdata.zip",
        "202501-citibike-tripdata.zip",
        "202502-citibike-tripdata.csv.zip",
    ]
    assert select_sources(sources, 2025, 2)[-1] == Source("202502-citibike-tripdata.csv.zip", 2025, 2, 1)


def test_combine_daily_sums_files_and_drops_incomplete_last_day():
    days = pl.date_range(date(2025, 1, 1), date(2025, 1, 10), "1d", eager=True)
    full = pl.DataFrame({"date": days, "member_casual": "member", "rideable_type": "x", "trips": 1000})
    stub = pl.DataFrame(
        {"date": [date(2025, 1, 11)], "member_casual": ["member"], "rideable_type": ["x"], "trips": [5]}
    )
    out = combine_daily([full.cast({"trips": pl.Int64}), stub.cast({"trips": pl.Int64})], "2025-01-02")
    assert out["date"].min() == date(2025, 1, 2)
    assert out["date"].max() == date(2025, 1, 10)


def test_ingest_trips_against_a_fake_bucket(tmp_path, monkeypatch):
    """Full download path: S3 XML listing -> zip download -> DuckDB aggregation -> cache."""
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from citibike.ingest import ingest_trips

    files = {"202501-citibike-tripdata.zip": _nested_zip(tmp_path / "m.zip").read_bytes()}
    listing = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>false</IsTruncated>'
        + "".join(f"<Contents><Key>{k}</Key><Size>{len(v)}</Size></Contents>" for k, v in files.items())
        + "<Contents><Key>JC-202501-citibike-tripdata.csv.zip</Key><Size>1</Size></Contents>"
        "</ListBucketResult>"
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0].removeprefix("/tripdata").lstrip("/")
            body = listing if path == "" else files.get(path)
            self.send_response(200 if body else 404)
            self.end_headers()
            self.wfile.write(body or b"")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/tripdata"
        daily = ingest_trips(url, "2025-01-01", tmp_path / "raw", tmp_path / "interim")
    finally:
        server.shutdown()
    assert daily["trips"].sum() == 5
    assert (tmp_path / "interim" / "202501-citibike-tripdata.parquet").exists()
    assert not list((tmp_path / "raw").glob("*.zip"))  # raw zips are deleted after aggregation
