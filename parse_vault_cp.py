import pandas as pd
import numpy as np
from pathlib import Path
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# from nltk.stem import PorterStemmer
# from nltk.stem import LancasterStemmer

## Set paths
cwd = Path.cwd()
#vault_str = '/home/nef/Documents/nefdocs/vault1'
vault_str = r'C:\\Users\\Rafal\\Documents\\vault1'
vault_pth = Path(vault_str)

## Parse Obsidian Vault notes
## Catch notes paths
notes_paths = [
    x for x in vault_pth.rglob('*') if x.suffix == '.md'
]

## read each note
NOTES = pd.DataFrame(notes_paths, columns=['path'])
NOTES['contents'] = np.nan
breakpoint()
NOTES['contents'] = NOTES['path'].apply(
    lambda x: Path(x).read_text()
)

# TODO: get rid of numbers, html code, url links etc. before fit transform vectorizer for TFIDF modeling
patterns = [
    "<div style='background-color: #62535D60; padding:5px; border-radius: 5px;'>\n\t<p></p>\n</div>\n\n>", 
    "div",
    "Shells",
    "Tags"
]
regs = [
    '(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])', #match url
    '(\!\[(.*)\])', #match pictures encoded as ![picturename]
    '(\$(.*)\$)', #match any LaTex so you can flatten it
    '(\d\d:\d\d)'#match 00:00 time format to remove it
]

## Build notes vector representattion table
text = NOTES.loc[10]['contents'] 
vectorizer3 = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
X3 = vectorizer3.fit_transform([text])
#vectorizer3.get_feature_names_out()



# Useless functions:
def pattern_indexes(pattern, document):
    """Function scans for pattern addresses
    POINTLESS xd
    """
    nspans = [x.span() for x in re.finditer('\\n', text)]
    span1 = [x[0] for x in nspans]
    span2 = [x[1] for x in nspans]
    is_ = [x[0]-1 for x in nspans]
    ivals = [text[x] for x in is_]

    xy = [x for x in zip(span1, span2, is_, ivals)]
    X_MAP = pd.DataFrame(xy, columns=['s1', 's2', 'i', 'ival'])

    return X_MAP

def insert_pat(string, index, pat):
    # Use this to insert ' ' under given address so you can
    # split text into tokens for further pre-processing
    return string[:index] + pat + string[index:]
