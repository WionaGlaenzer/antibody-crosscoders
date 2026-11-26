"""Cluster-side UniProt annotation fetcher (Euler has internet w/ eth_proxy).
Resume-capable. Writes results/uniprot_annotations.tsv."""

import urllib.request, urllib.parse, json, time, os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

D = BASE
accs = json.load(open(D + "/results/accs.json"))
fields = "accession,ft_compbias,ft_domain,ft_binding,ft_act_site,ft_transmem,ft_signal,ft_coiled,ft_disulfid,ft_region,ft_helix,ft_strand,ft_turn"
out_path = D + "/results/uniprot_annotations.tsv"
done = set()
if os.path.exists(out_path):
    for l in open(out_path):
        a = l.split("\t", 1)[0]
        if a and a != "Entry":
            done.add(a)
todo = [a for a in accs if a not in done]
print(f"resume: {len(done)} done, {len(todo)} todo", flush=True)


def fetch_chunk(chunk, retries=4):
    q = " OR ".join(f"accession:{a}" for a in chunk)
    url = "https://rest.uniprot.org/uniprotkb/stream?" + urllib.parse.urlencode(
        {"query": q, "fields": fields, "format": "tsv"}
    )
    for r in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            return urllib.request.urlopen(req, timeout=180).read().decode()
        except Exception as e:
            if r == retries - 1:
                return None
            time.sleep(3 * (r + 1))
    return None


CHUNK = 100
t0 = time.time()
fails = []
hdr = os.path.exists(out_path) and os.path.getsize(out_path) > 0
with open(out_path, "a") as fout:
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i : i + CHUNK]
        txt = fetch_chunk(chunk)
        if txt is None:
            fails.extend(chunk)
            continue
        lines = txt.split("\n")
        if not hdr:
            fout.write(lines[0] + "\n")
            hdr = True
        for l in lines[1:]:
            if l.strip():
                fout.write(l + "\n")
        fout.flush()
        if i % (CHUNK * 20) == 0:
            print(f"[{time.time()-t0:.0f}s] {i}/{len(todo)} fails={len(fails)}", flush=True)
        time.sleep(0.15)
# retry fails once
if fails:
    print("retrying", len(fails), "fails", flush=True)
    with open(out_path, "a") as fout:
        for i in range(0, len(fails), CHUNK):
            txt = fetch_chunk(fails[i : i + CHUNK])
            if txt:
                for l in txt.split("\n")[1:]:
                    if l.strip():
                        fout.write(l + "\n")
                fout.flush()
            time.sleep(0.3)
print("DONE. total rows:", sum(1 for _ in open(out_path)) - 1, flush=True)
