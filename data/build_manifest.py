"""ATLAS_MANIFEST.csv: the public CELLxGENE files behind every number, with versioned download URLs.

    python data/build_manifest.py            re-read url, size, cell count and title of every row from the CELLxGENE curation API
    python data/build_manifest.py --check    HEAD every download_url and compare its size with remote_size (exit 1 on a mismatch)

Rows keep their key, name, set and md5. md5 is filled only where we hashed our own copy; CELLxGENE publishes no checksum.
The URL names a dataset version, so it keeps working after the dataset is updated; a refresh would point to the new version.
"""
import csv
import json
import os
import sys
import urllib.request

M = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ATLAS_MANIFEST.csv")
API = "https://api.cellxgene.cziscience.com/curation/v1/collections/{}/datasets/{}"


def rows():
    with open(M, newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def refresh():
    cols, R = rows()
    for r in R:
        if not (r["dataset_id"] and r["collection_id"]):
            continue
        with urllib.request.urlopen(API.format(r["collection_id"], r["dataset_id"]), timeout=60) as u:
            d = json.load(u)
        a = [x for x in d["assets"] if x["filetype"] == "H5AD"][0]
        assert a["url"].endswith(".h5ad"), a
        r.update(download_url=a["url"], remote_size=a["filesize"], n_cells=d["cell_count"], title=d["title"][:120].replace(",", ";"))
    with open(M, "w", newline="") as f:
        w = csv.DictWriter(f, cols, lineterminator="\n")
        w.writeheader()
        w.writerows(R)


def check():
    bad = 0
    for r in rows()[1]:
        if not r["download_url"]:
            continue
        req = urllib.request.Request(r["download_url"], method="HEAD")
        try:
            n = int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])
        except Exception as e:
            n = repr(e)
        ok = str(n) == r["remote_size"]
        bad += not ok
        print(("ok  " if ok else "BAD ") + r["key"], n if not ok else "")
    return bad


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(1 if check() else 0)
    refresh()
