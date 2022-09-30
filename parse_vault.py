import pandas as pd
import numpy as np
from pathlib import Path
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# from nltk.stem import PorterStemmer
# from nltk.stem import LancasterStemmer

## Set paths
cwd = Path.cwd()
# vault_str = '/home/nef/Documents/nefdocs/vault1'
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
NOTES['contents'] = NOTES['path'].apply(
    lambda x: Path(x).read_text(encoding='utf-8')
)

NOTES_bak = NOTES.copy()

## PREPROCESSING ##

pats = [
    # r"<div style='background-color: #62535D60; padding:5px; border-radius: 5px;'>\n\t<p></p>\n</div>\n\n>", 
    r"shells",
    r"tags",
    r"date",
    r"links",
    r"source",
    r"todo",
    r"\n",
    r"\t",
    r'__',
    r'_',
    r'(<div.*div>)',
    r'(#\S+)', #remove tags,
    r'(\!\[(.*)\])', #remove anything encapsulated in ![ ]
    # r'(\$(.*)\$)', #match any LaTex so you can flatten it
    r'(\d\d:\d\d)',#match 00:00 time format to remove it 
    r'(\d\d\d\d\d\d)', #match date pattern 1
    r'(\d\d-\d\d-\d\d)', #match dat attern 2
    r'(\d\d)',
    r'({.*})' #match special templat expressions
]

regurl = [r'(\S+\.(com|net|org|edu|gov|pl|eu|de)(\/\S+)?)'] #match url

# Text preprocessing

NOTES['contents'] = (
        NOTES['contents']
        .str.lower()
)

# Remove known not meaningful patterns
def sub_patterns(patterns):
    for pattern in patterns:
        NOTES['contents'] = (
            NOTES['contents'].apply(
                lambda x: re.sub(pattern, ' ', fr'{x}')
            )
        )
sub_patterns(pats)

# Escape special chars so we can match urls
schars = [
    ':',
    '/',
]
escape = lambda s, escapechar, specialchars: "".join(
    escapechar + c if c in specialchars or c == escapechar else c for c in s
    )
for chr_ in schars:
    NOTES['contents'] = (
        NOTES['contents'].apply(
            lambda x: escape(x, '\\', chr_)
        )
    )

## remove urls now
sub_patterns(regurl)


## test
cont_test = pd.merge(
    NOTES['contents'], 
    NOTES_bak['contents'], 
    left_index=True, 
    right_index=True, 
    how='left'
)

########################################################

## Build notes vector representattion table
text = NOTES['contents'] 
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.99,
    min_df=0.01,
    norm='l2',
    smooth_idf=True,
    analyzer='word', 
    ngram_range=(1, 4))
X = vectorizer.fit_transform(text)
features = vectorizer.get_feature_names_out()


## Construct DF
FNAMES = NOTES['path'].apply(lambda x: Path(x).stem)
MDF = pd.DataFrame(
    X.toarray(),
    columns=features,
    index=FNAMES
)





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

#testfunc
print(insert_pat("355879ACB6", 5, ' FUCK '))