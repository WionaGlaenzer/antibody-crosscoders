#!/usr/bin/env python
"""Step 1: parse Swiss-Prot FASTA, filter Ig/antibody, length+quality filter, dedup.
Writes filtered.fasta (for mmseqs) and step1_stats.json.
"""

import gzip, json, re, sys, hashlib

FASTA_GZ = sys.argv[1]  # uniprot_sprot.fasta.gz
OUT_FASTA = sys.argv[2]  # filtered.fasta
OUT_STATS = sys.argv[3]  # step1_stats.json

MINLEN, MAXLEN = 20, 512
STD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# --- Ig / antibody / TCR detection on the protein NAME (FASTA description) ---
# Swiss-Prot FASTA headers carry only: sp|ACC|ENTRY <ProteinName> OS=.. OX=.. GN=.. PE=.. SV=..
# So we filter on the protein name and on the gene symbol (GN=).
NAME_PATTERNS = [
    r"immunoglobulin",
    r"\bantibod",  # antibody / antibodies
    r"\bIg\s?(heavy|kappa|lambda|gamma|mu|alpha|delta|epsilon)\b",
    r"\bIg[GAMDE]\b",  # IgG IgA IgM IgD IgE
    r"T[-\s]?cell receptor",
    r"\bTCR\b",
    r"\bT[-\s]?cell surface glycoprotein CD3",  # TCR-associated (conservative)
    r"\bFc receptor",  # keep conservative: Fc receptors are Ig-related
    r"\bhypervariable",
    r"\bV\(D\)J\b",
]
NAME_RE = re.compile("|".join(NAME_PATTERNS), re.IGNORECASE)

# Gene-symbol prefixes for Ig loci and TCR loci. Specific enough to avoid IGF etc.
GENE_PREFIXES = ("IGH", "IGK", "IGL")  # Ig heavy/kappa/lambda
TCR_GENE_RE = re.compile(r"^TR[ABGD][VDJC]")  # TRBV, TRAC, TRDJ ...


def parse_header(h):
    # h like: sp|Q6GZX4|001R_FRG3G Putative transcription factor 001R OS=... GN=...
    parts = h.split(None, 1)
    tag = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    acc = tag.split("|")[1] if "|" in tag else tag
    # protein name = rest up to first ' XX=' token (OS=/OX=/GN=/PE=/SV=)
    m = re.search(r"\s(?:OS|OX|GN|PE|SV)=", " " + rest)
    name = rest[: m.start()].strip() if m else rest.strip()
    gm = re.search(r"\bGN=(\S+)", rest)
    gene = gm.group(1) if gm else ""
    return acc, name, gene


def is_ig(name, gene):
    if NAME_RE.search(name):
        return "name"
    g = gene.upper()
    if g.startswith(GENE_PREFIXES) or TCR_GENE_RE.match(g):
        return "gene"
    return None


def iter_fasta(fh):
    hdr, seq = None, []
    for line in fh:
        line = line.rstrip("\n")
        if line.startswith(">"):
            if hdr is not None:
                yield hdr, "".join(seq)
            hdr, seq = line[1:], []
        else:
            seq.append(line)
    if hdr is not None:
        yield hdr, "".join(seq)


stats = dict(
    n_downloaded=0,
    n_removed_ig=0,
    ig_by_name=0,
    ig_by_gene=0,
    n_after_ig=0,
    n_removed_length=0,
    n_removed_nonstd=0,
    n_after_quality=0,
    n_dupes_removed=0,
    n_after_dedup=0,
    lengths=[],
)

seen = set()
lengths_kept = []
with gzip.open(FASTA_GZ, "rt") as fh, open(OUT_FASTA, "w") as out:
    for hdr, seq in iter_fasta(fh):
        stats["n_downloaded"] += 1
        acc, name, gene = parse_header(hdr)
        ig = is_ig(name, gene)
        if ig:
            stats["n_removed_ig"] += 1
            stats["ig_by_name" if ig == "name" else "ig_by_gene"] += 1
            continue
        stats["n_after_ig"] += 1
        L = len(seq)
        if L < MINLEN or L > MAXLEN:
            stats["n_removed_length"] += 1
            continue
        if not set(seq) <= STD_AA:  # any non-standard residue (X/B/Z/U/O/*) -> drop
            stats["n_removed_nonstd"] += 1
            continue
        stats["n_after_quality"] += 1
        h = hashlib.md5(seq.encode()).hexdigest()
        if h in seen:
            stats["n_dupes_removed"] += 1
            continue
        seen.add(h)
        lengths_kept.append(L)
        out.write(f">{acc}\n{seq}\n")

stats["n_after_dedup"] = len(seen)
stats["lengths"] = lengths_kept
with open(OUT_STATS, "w") as f:
    json.dump(stats, f)
print("STEP1 done:", {k: v for k, v in stats.items() if k != "lengths"})
