import pandas as pd
import numpy as np
from pathlib import Path

from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

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

## Build notes vector representattion table
##
## remove not meaningful patterns
import re
text = NOTES.loc[10]['contents'] 
text
nspans = [x.span() for x in re.finditer('\\n', text)]
span1 = [x[0] for x in nspans]
span2 = [x[1] for x in nspans]
is_ = [x[0]-1 for x in nspans]
ivals = [text[x] for x in is_]

xy = [x for x in zip(span1, span2, is_, ivals)]
X_MAP = pd.DataFrame(xy, columns=['s1', 's2', 'i', 'ival'])

# TODO: need to find a way to efficiently insert characters 
# in a string using numpy (represent str as a vector and
# then replace items via vect operation)
# then need to wrap it up to a func since i need tto do it again 
# on other pattern (\t)
# aftter that get rid of all \n and \t patterns in the text
# then split text by ' ' and get rid of not meaningful tokens
# then calc TFIDF for each token in a vector and build vector table

def insert_pat(string, index, pat):
    # Use this to insert ' ' under given address so you can
    # split text into tokens for further pre-processing
    return string[:index] + pat + string[index:]

#testfunc
print(insert_pat("355879ACB6", 5, ' FUCK '))


#############################
## data stemming
#create an object of class PorterStemmer
porter = PorterStemmer()