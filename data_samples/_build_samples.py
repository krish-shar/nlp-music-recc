"""Compile small, representative slices of the project's data assets.

Outputs (all under data_samples/):
  raw_csv_sample.csv         -- 2 songs x 5 artists, lyrics truncated to 600 chars
  raw_json_sample.json       -- one full Genius-API song record (Taylor Swift)
  pairs_train_sample.csv     -- 25 training pairs sampled across artists
  pairs_val_sample.csv       -- 15 validation pairs sampled across artists
"""
import json
import os
import random
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/venushabuwaneka/Downloads/nlphw")
OUT = ROOT / "data_samples"
OUT.mkdir(exist_ok=True)

random.seed(0)

# ---------- raw CSV sample ----------
csv_dir = ROOT / "Small_Pop_Song_Dataset" / "csv"
artists = ["TaylorSwift", "BillieEilish", "Drake", "EdSheeran", "ArianaGrande"]
rows = []
for a in artists:
    df = pd.read_csv(csv_dir / f"{a}.csv")
    df = df.dropna(subset=["Lyric"]).head(2)
    for _, r in df.iterrows():
        lyric = str(r["Lyric"])
        if len(lyric) > 600:
            lyric = lyric[:600].rstrip() + " ..."
        rows.append({
            "Artist": r["Artist"],
            "Title": r["Title"],
            "Album": r.get("Album"),
            "Year": r.get("Year"),
            "Date": r.get("Date"),
            "Lyric": lyric,
        })
pd.DataFrame(rows).to_csv(OUT / "raw_csv_sample.csv", index=False)
print(f"wrote raw_csv_sample.csv ({len(rows)} rows)")

# ---------- raw JSON sample ----------
with open(ROOT / "Small_Pop_Song_Dataset" / "json files" / "Lyrics_TaylorSwift.json") as f:
    blob = json.load(f)
song = blob["songs"][0]
keep_top = ["alternate_names", "name", "id", "url", "facebook_name", "instagram_name"]
artist_meta = {k: blob.get(k) for k in keep_top if k in blob}
sample = {
    "artist_metadata": artist_meta,
    "first_song": song,
}
with open(OUT / "raw_json_sample.json", "w") as f:
    json.dump(sample, f, indent=2, ensure_ascii=False)
print("wrote raw_json_sample.json")

# ---------- pairs samples ----------
def sample_pairs(path, k):
    df = pd.read_parquet(path)
    grouped = df.groupby("Artist", group_keys=False)
    per_artist = max(1, k // max(1, df["Artist"].nunique()))
    picks = grouped.apply(lambda g: g.sample(min(per_artist, len(g)), random_state=0))
    if len(picks) > k:
        picks = picks.sample(k, random_state=0).sort_values(["Artist", "song_id"])
    return picks

train = sample_pairs(ROOT / "lyric_pairs" / "pairs_train.parquet", 25)
val = sample_pairs(ROOT / "lyric_pairs" / "pairs_val.parquet", 15)

train.to_csv(OUT / "pairs_train_sample.csv", index=False)
val.to_csv(OUT / "pairs_val_sample.csv", index=False)
print(f"wrote pairs_train_sample.csv ({len(train)} rows)")
print(f"wrote pairs_val_sample.csv ({len(val)} rows)")

# ---------- summary ----------
print("\nfiles in data_samples/:")
for p in sorted(OUT.iterdir()):
    if p.is_file():
        print(f"  {p.name:30s} {p.stat().st_size:>10,} bytes")
