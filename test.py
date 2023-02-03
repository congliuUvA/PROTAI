from pathlib import Path
import ray
pdb_raw_file = Path('/hddstore/cliu3/pdb_raw_files')

idx = 0
for file in pdb_raw_file.rglob("*.gz"):
    idx += 1

print(idx)
