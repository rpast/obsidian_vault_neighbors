""" Welcome to Vault Neighbors script!
v.1.0

The script is intended to be run from CLI with predefined set of arguments.
Run `python scriptname.py -h` to get help on possible options.

The purpose of this script is to aid the creative process of note-taking in 
Obsidian.md environment. By default it returns 10 nearest neighbors of a 
given note. This serves as a recommendation of connections that can be formed
between the notes. Thanks to that novel edges can be created.

"""

# TODO:
# Accuracy measurement (current graph edges vs recommended neighbors)
# 1.    Write function that returns all outlinks for a given note
# 2.    Write function that compares outlinks with nns and spits out the score
# show quantified distance between notes
# peek note content


import re
import argparse

from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 50)
# pd.set_option('display.width', 1000)


# Global parametrisation
PARSER = argparse.ArgumentParser()

enc = 'utf-8'
de = False
do = False
ipth = './nnignore.txt'
ppth = './nnpatterns.txt'
vpth = './nnpath.txt'
dist = 'cosine'
nn = 10
verb = False
t = None
pk = False

PARSER.add_argument(
    '-e', '--encoding',
    help=f'Set encoding for vault notes. Default {enc}',
    type=str,
    default=enc
)
PARSER.add_argument(
    '-de', '--drop_empty',
    help='Drop empty notes instead of filling them in with titles',
    action='store_true',
    default=de
)
PARSER.add_argument(
    '-do', '--drop_outlinks',
    help='Dont take outlinks into the nearest neighbors analysis',
    action='store_true',
    default=do
)
PARSER.add_argument(
    '-ip', '--ignore_path',
    help=f'Set path to nnignore.txt file. Default {ipth}',
    type=str,
    default=ipth
)
PARSER.add_argument(
    '-mt', '--metric',
    help='Set distance metric for NN algorithm. Default \'cosine\'. Available metrics: jaccard, cosine.',
    type=str,
    default=dist
)
PARSER.add_argument(
    '-nn', '--neighbors_number',
    help='Set nearest neighbors number the script should return. Default 10.',
    type=int,
    default=nn
)
PARSER.add_argument(
    '-pp', '--pattern_path',
    help=f'Set path to nnpatterns.txt. Default {ppth}',
    type=str,
    default=ppth
)
PARSER.add_argument(
    '-pk', '--peek',
    help='Show note content after it got processed by the script.',
    action='store_true',
    default=pk
)
PARSER.add_argument(
    '-t', '--title',
    help='Pass full note title for which you want to get nearest neighbors',
    type=str,
    default=t
)
PARSER.add_argument(
    '-v', '--verbose',
    help='Make script output more verbose',
    action='store_true',
    default=verb
)
PARSER.add_argument(
    '-vp', '--vault_path',
    help=f'Set path to nnpath.txt. Default {vpth}',
    type=str,
    default=vpth
)

ARGS = PARSER.parse_args()

# Global params
NOTE_TITLE =ARGS.title

ENCODING = ARGS.encoding
DROP_EMPTY = ARGS.drop_empty
DROP_OLINKS = ARGS.drop_outlinks
IGNORE_PTH = ARGS.ignore_path
PATTERN_PTH = ARGS.pattern_path
V_PATH = ARGS.vault_path
DISTANCE = ARGS.metric
N_NUMBER = ARGS.neighbors_number
VERBOSE = ARGS.verbose
PEEK = ARGS.peek

PARM_CONTAINER = [ENCODING, DROP_EMPTY, DROP_OLINKS, IGNORE_PTH, PATTERN_PTH,
                  V_PATH, DISTANCE, N_NUMBER, VERBOSE]
if VERBOSE:
    print('Global parameters:')
    print(PARM_CONTAINER)
    print('\n')


# READ DATA ##
def read_vault(pth, verbose=False):
    """Reads .md files under given path

    """
    if verbose:
        print(f'Parsing vault at {pth}')

    notes_paths = [
        x for x in pth.rglob('*') if x.suffix == '.md'
    ]

    # read each note
    notes = pd.DataFrame(
        notes_paths,
        columns=['path']
    )
    notes['contents'] = np.nan
    notes['contents'] = notes['path'].apply(
        lambda x: Path(x).read_text(encoding='utf-8')
    )

    return notes


# PREPROCESSING ##
def read_txt(pth, enc, v):
    """Reads txt file if available

    """

    if v:
        print(f'Reading text file under {pth}')

    f_ = Path(pth)

    if f_.exists():
        return f_.read_text(encoding=enc)

    return None


def prep_patterns(patterns, v):
    """Transforms patterns txt file into a Python list

    """

    if v:
        print('Cleaning txt patterns')

    if patterns is not None:
        reg = '\\n'
        patterns_sub = re.sub(reg, ' ', patterns)
        pats = patterns_sub.split(' ')
        pats = [x for x in pats if x != '']

        return pats

    return None


def filter_ignore(df_, patterns, v):
    """Returns indexes of notes to ignore as per nnignore.txt

    """

    if v:
        print(f'Filtering out notes under ignored locations: {patterns}')

    subdf_ = df_.copy()

    indexes = []
    subdf_['path_str'] = np.nan
    subdf_['path_str'] = subdf_['path'].apply(
        lambda x: str(x)
    )
    if isinstance(patterns, list):
        for pat in patterns:
            ignored = (subdf_
                       [subdf_['path_str'].str.contains(fr'{pat}', na=False)])
            indexes.append(ignored.index.values)
        indexes = [j for i in indexes for j in i]
    elif isinstance(patterns, str):
        ignored = (
            subdf_
            [subdf_['path_str'].str.contains(fr'{patterns}', na=False)]
        )
        indexes.append(ignored.index.values)

    return indexes


if VERBOSE:
    print('### Notes recommendation script run. ###')


# Set paths
# Read vault path file. If no path defined, then ask user to provide it.
VP = read_txt(
    V_PATH,
    ENCODING,
    VERBOSE
)
if VP is None:
    print('nnpath.txt not found\n')
    VAULT_STR = input('Provide absolute path to the vault: ')
    VAULT_PTH = Path(VAULT_STR)
else:
    VP = prep_patterns(
        VP,
        VERBOSE
    )
    if len(VP) != 1:
        print('nnpath.txt is content is ambiguous\n')
        VAULT_STR = input('Provide absolute path to the vault: ')
        VAULT_PTH = Path(VAULT_STR)
    else:
        VAULT_PTH = Path(VP[0])


# Read .md files located under given vault location
NOTES_DF = read_vault(
    VAULT_PTH,
    verbose=VERBOSE
)

# Grab patterns for note ignore logic
IGNORES = read_txt(
    IGNORE_PTH,
    ENCODING,
    VERBOSE
)

# Grab indexses to drop from dataframe
ILIST = filter_ignore(
    NOTES_DF,
    IGNORES,
    VERBOSE
)

# Get rid of notes that contain path patterns from nnignore file
if len(ILIST) != 0:
    NOTES_DF = NOTES_DF[~NOTES_DF.index.isin(ILIST[0])]


# Handle empty notes
def handle_empty(df_, drop_empty, v):
    """Deals with empty notes in the dataset

    """

    if v:
        print(f'Handling empty notes. Drop empty = {drop_empty}')

    # Make sure all empty notes have no trailing spaces
    df_['contents'] = df_['contents'].str.strip()

    # Set filter
    empty_f = (df_['contents'] == '')

    if drop_empty:
        df_ = df_[~empty_f]
    else:
        # Use stem as contents
        df_.loc[empty_f, 'contents'] = (
            df_
            ['path'].apply(
                lambda x: x.stem
            )
        )

    return df_


NOTES_DF = handle_empty(
    NOTES_DF,
    DROP_EMPTY,
    VERBOSE
)
# Make sure the index is increasing monotonicaly
NOTES_DF = NOTES_DF.reset_index(drop=True)


def get_titles(df_, v):
    """Create a title feature

    """

    if v:
        print('Preparing notes titles')

    df_['title'] = df_['path'].apply(
            lambda x: x.stem.lower()
        )

    return df_


# Text preprocessing
NOTES_DF['contents'] = (
    NOTES_DF
    ['contents']
    .str.strip()
)
NOTES_DF['contents'] = (
        NOTES_DF['contents']
        .str.lower()
)
NOTES_DF = get_titles(
    NOTES_DF,
    VERBOSE
)

# Tackle stop words/patterns
# Create a copy of a dataframe to process further with regex
NOTES_REG_DF = NOTES_DF.copy()

# Implementation of pattern.txt file
PATTERNS = read_txt(
    PATTERN_PTH,
    ENCODING,
    VERBOSE
)
pats = prep_patterns(
    PATTERNS,
    VERBOSE
)

# If user specifies so, drop outgoing links from the note so they are not
# taken into nearest neighbors algorithm
def drop_olinks(patterns, olink_flag):
    """ Enrich exception patterns with outgoing link pattern if flag = True

    """
    if olink_flag:
        olink_pat = r'\[\[.*\]\]'

        if patterns is not None:
            patterns.append(olink_pat)
            return patterns

    return patterns


pats = drop_olinks(
    pats,
    DROP_OLINKS
)


# Regex pattern matching and cleaning
# Remove known not meaningful patterns
def sub_patterns(patterns, field, df_, v):
    """ Substitute passed patterns with ' '

    """

    if v:
        print('Removing known not meaningful patterns from notes')

    if patterns is not None:
        for pattern in patterns:
            df_[field] = (
                df_[field].apply(
                    lambda x: re.sub(pattern, ' ', f'{x}')
                )
            )
        return df_

    return df_


NOTES_PROC = sub_patterns(
    pats,
    'contents',
    NOTES_REG_DF,
    VERBOSE
)


# Build table for manual testing of text preprocessing - before/after comparison
cont_test = pd.merge(
    NOTES_PROC['contents'],
    NOTES_DF['contents'],
    left_index=True,
    right_index=True,
    how='left'
)
if VERBOSE:
    print('Built cont_test dataframe for manual testing in console env')


############################################
# Model                                    #
# Build notes vector representattion table #

if VERBOSE:
    print('Building vector representation')

text = NOTES_PROC['contents']

# Model specific parameters
STOP_WORDS = 'english'
MAX_DF = 0.99
MIN_DF = 1
NORM = 'l2'
NGRAM_RANGE = (1, 3)

if VERBOSE:
    print(f'Modeling vars: STOP_WORDS={STOP_WORDS}, MAX_DF={MAX_DF}, MIN_DF={MIN_DF}, NORM={NORM}, NGRAM_RANGE={NGRAM_RANGE}')


def tfidf_vec(swords, mxdf, midf, nrm, nrange, v):
    """ TfidfVectorizer instatiate

    """

    if v:
        print('Using TFIDF vectorizer')

    vectorizer = TfidfVectorizer(
        stop_words=swords,
        max_df=mxdf,
        min_df=midf,
        norm=nrm,
        smooth_idf=True,
        analyzer='word',
        ngram_range=nrange
    )

    return vectorizer


def count_vec(midf, nrange, v):
    """ CountVectorizer instatiate

    """

    if v:
        print('Using Count vectorizer')

    vectorizer = CountVectorizer(
        min_df=midf,
        ngram_range=nrange,
        stop_words='english',
        binary=True
     )

    return vectorizer


# Vectorizer depends on the distance method user wants to use
if VERBOSE:
    print(f'Apllied metric = {DISTANCE}')

if DISTANCE == 'cosine':
    vectorizer = tfidf_vec(
        STOP_WORDS,
        MAX_DF,
        MIN_DF,
        NORM,
        NGRAM_RANGE,
        VERBOSE
    )
elif DISTANCE == 'jaccard':
    vectorizer = count_vec(
        MIN_DF,
        NGRAM_RANGE,
        VERBOSE
    )
else:
    raise TypeError("Distance not recognized. Use: cosine, jaccard")


# Vectorize documents and processing
X = vectorizer.fit_transform(text)
X_d = X.todense()
X_d = np.asarray(X_d)
if DISTANCE == 'jaccard':
    X_d = np.array(X_d, dtype=bool)


# We will use features list later to catch what tokens were taken
# into the calculation
FEATURES = vectorizer.get_feature_names_out()


# Construct helper DF
MDF = pd.DataFrame(
    X.toarray(),
    columns=FEATURES,
    index=NOTES_PROC['title']
)
if VERBOSE:
    print('Constructed MDF helper dataframe')


# Find nearest neighbors
def calc_neighbors(metric, nnum, data, v):
    """ wrap NearestNeighbors with given distance metric

    """

    if v:
        print(f'Calculating neighbors. NN={N_NUMBER}')

    nbrs = NearestNeighbors(
        n_neighbors=nnum+1,
        algorithm='brute',
        metric=metric
    ).fit(data)

    return nbrs


NBRS = calc_neighbors(
    DISTANCE,
    N_NUMBER,
    X_d,
    VERBOSE
)

# Grab quantified distances for given indices
distances, indices = NBRS.kneighbors(X_d)


# Interface for NN query
def find_index(title, corpus):
    """Helper function to extract index from master table using given note title

    :param title: Obsidian note title
    :type title: str
    :param corpus: Source table with processed notes
    :type corpus: pd.DataFrame
    :return: index number corresponding to a given title or None
    :rtype: int
    """
    tit = title.lower()

    note_query = corpus[corpus['title'] == tit]

    if note_query.empty:
        return None

    return note_query.index[0]


def show_nn(note_title, corpus, dropna=False):
    """Returns nearest neighbors dataframe for a given note. The table consists
    of neighbors titles, tokens used by the model and their corresponding values.

    :param note_title: title of the note for which user wants to extract nns
    :type note_title: str
    :param corpus: Source table with processed notes
    :type corpus: pd.DataFrame
    :param dropna: use this if you want to see only non-zero value tokens for
    at least one neighbor
    :type dropna: bool
    :returns: pd.DataFrame of nearest neighbors for given note title. First
    column represents a note for which nearest neighbors are calculated in the
    next columns.
    """

    i = find_index(
        note_title,
        corpus
    )

    df_ = pd.DataFrame(indices)

    # If note not found in corpus:
    if i is None:
        print("WARNING: Master note not found in the corpus.")
        print('Check the title or if the note is empty.')
        return None

    nns = df_[df_[0] == i].values

    if len(nns) == 0:
        print('WARNING: Neighbor list for this note is empty.')
        print('The note may be empty or too short or nnpatterns.txt wiped out its contents')
        return None

    sub_df = MDF.iloc[nns[0]].T

    if dropna:
        sub_df_s = (
            sub_df
            .replace(0, np.nan)
            .dropna(how='all')
        )
        return sub_df_s

    return sub_df


if NOTE_TITLE is not None:
    recom_df = show_nn(
        NOTE_TITLE,
        NOTES_PROC
    )

    if recom_df is not None:
        for _ in recom_df.columns:
            print(_)

    # Define contents print when user wants to look it up
    if PEEK:
        idx = find_index(
            NOTE_TITLE,
            NOTES_PROC
        )

        print('\nNote contents after processing >>>')
        print(NOTES_PROC.loc[idx]['contents'])
        print('\n')


if VERBOSE:
    print('\nTERMINATED')
