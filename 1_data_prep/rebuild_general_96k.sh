#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16000
#SBATCH --cpus-per-task=8
set -e
D=/cluster/project/reddy/wglaenzer/crosscoders
W=/cluster/scratch/wglaenzer/general_96k_work
mkdir -p $W
module load stack/2024-06 gcc/12.2.0 openmpi/4.1.6 mmseqs2/14-7e284 eth_proxy 2>/dev/null
source $D/../environments/crosscoder/bin/activate

# 1. download Swiss-Prot (needs eth_proxy for internet)
cd $W
if [ ! -f uniprot_sprot.fasta.gz ]; then
  echo "[dl] downloading Swiss-Prot"
  wget -q "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz" -O uniprot_sprot.fasta.gz
fi
echo "[dl] done $(ls -la uniprot_sprot.fasta.gz | awk '{print $5}') bytes"

# 2. step1: filter Ig/antibody + quality + dedup -> filtered.fasta
python $D/code/analysis/step1_filter.py uniprot_sprot.fasta.gz filtered.fasta step1_stats_96k.json
echo "[step1] done"

# 3. mmseqs cluster at 0.5 id (same as original)
mmseqs easy-cluster filtered.fasta clu96 tmp96 --min-seq-id 0.5 -c 0.8 --cov-mode 0
NREP=$(grep -c ">" clu96_rep_seq.fasta)
echo "[cluster] $NREP representatives"

# 4. step2: build parquet with ALL reps (target huge so no subsample) -> 96k set
python $D/code/analysis/step2_build.py clu96_rep_seq.fasta step1_stats_96k.json \
  $D/data/general_protein_96k.parquet $D/results/general_96k_manifest.json \
  $D/results/general_96k_README.md \
  "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz" \
  "2026_02" 1000000 0.5 0.8
echo "[step2] DONE"
python -c "import pandas as pd; d=pd.read_parquet('$D/data/general_protein_96k.parquet'); print('general 96k parquet rows:', len(d))"
