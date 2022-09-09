import pandas as pd
import numpy as np
from pathlib import Path

## Set paths
cwd = Path.cwd()
vault_str = '/home/nef/Documents/nefdocs/vault1'
vault_pth = Path(vault_str)

## Parse Obsidian Vault notes
## Catch notes paths
notes_paths = [
    x for x in vault_pth.rglob('*') if x.suffix == '.md'
]

## read each note
NOTES = pd.DataFrame(notes_paths, columns=['path'])
NOTES['contents'] = np.nan
NOTES['contents'] = NOTES['path'].apply(
    lambda x: Path(x).read_text()
)

## data stemming
