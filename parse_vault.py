# TODO:
# 1. Write function that returns all outlinks for a given note
# 2. Write function that compares outlinks with nns and spits out the score
# 3. let user choose what folder names in the vault to exclude (txt file)
# 4. let user choose how to treat empty notes (delete or fill)
# 5. let user provide customized regex file for pattern removal
# 6. implement it as script with arguments to be passed


# SETUP
import re

import pandas as pd
import numpy as np

from pathlib import Path
from sklearn import neighbors

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
# pd.set_option('display.width', 1000)

# Global params
ENCODING = 'utf-8'
DROP_EMPTY = False
IGNORE_PTH = './nnignore.txt'

# Set paths
cwd = Path.cwd()
vault_str = '/home/nef/Documents/nefdocs/vault1'
# vault_str = r'C:\\Users\\Rafal\\Documents\\vault1'
vault_pth = Path(vault_str)


# READ DATA ##
def read_vault(pth, verbose=False):
    """Reads .md files under given path

    TODO: allow user to exclude folders
    """
    if verbose:
        print(f'Parsing vault at {pth}')

    notes_paths = [
        x for x in vault_pth.rglob('*') if x.suffix == '.md'
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


# Read .md files located under given vault location
NOTES_DF = read_vault(
    vault_pth
)


# PREPROCESSING ##
def read_txt(pth):
    """Reads txt file if available
    """
    ignore_f = Path(pth)
    if ignore_f.exists():
        return ignore_f.read_text(encoding=ENCODING)
    else:
        return None


def filter_ignore(df_, patterns):
    """Returns indexes of notes to ignore as per nnignore.txt
    """

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


# Grab patterns for note ignore logic
IGNORES = read_txt(
    IGNORE_PTH
)

# Grab indexses to drop from dataframe
ILIST = filter_ignore(
    NOTES_DF,
    IGNORES
)

# Get rid of notes that contain path patterns from nnignore file
NOTES_DF = NOTES_DF[~NOTES_DF.index.isin(ILIST[0])]


# Handle empty notes
def handle_empty(df_, drop_empty):
    """Deals with empty notes in the dataset
    """

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
    DROP_EMPTY
)


def get_titles(df_):
    """Create a title feature
    """
    df_['title'] = df_['path'].apply(
            lambda x: x.stem.lower()
        )
    return df_


# Create backup dataframe for manual testing
NOTES_DF = get_titles(
    NOTES_DF
)

# Tackle stop words/patterns
# Create a copy of a dataframe to process further with regex
NOTES_REG_DF = NOTES_DF.copy()
# TODO: allow user to switch outlinks on/off

# Escape special chars so we can match urls
# schars = [
#     ':',
#     '/',
# ]

# escape = lambda s, escapechar, specialchars: "".join(
#     escapechar + c if c in specialchars or c == escapechar else c for c in s
#     )


# NOTES_REG_DF['contents'] = (
#     NOTES_REG_DF['contents'].apply(
#         lambda x: escape(x, '\\', schars)
#     )
# )

# Implementation of pattern.txt file
PATTERNS = read_txt('./nnpatterns.txt')

# Prune patterns and transform them to a list
# (?<=\#)(.*?)(?=\\n) #get rid of comments
# \# get rid of remaining hash symbols
# split data on \n pattern 
# inspect items in a resulting list

pats = [
    r'(\S+\.(com|net|org|edu|gov|pl|eu|de)(\/\S+)?)',  # match regular url
    r'(https\:\/\/docs.+\)',  # match google docs url in (...)
    r"2nd",
    r"1st",
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
    # r'\[\[.*\]\]', #drop obsidian forward links
    r'(<div.*div>)',
    r'(#\S+)',  # remove tags,
    r'(\!\[(.*)\])',  # remove anything encapsulated in ![ ]
    r'(\d\d:\d\d)',  # match 00:00 time format to remove it 
    r'(\d\d\d\d\d\d)',  # match date pattern 1
    r'(\d\d-\d\d-\d\d)',  # match dat pattern 2
    r'(\d\d)',  # any two digits
    r'({.*})'  # match special template expressions
]


# Text preprocessing
NOTES_REG_DF['contents'] = (
    NOTES_REG_DF
    ['contents']
    .str.strip()
)
NOTES_REG_DF['contents'] = (
        NOTES_REG_DF['contents']
        .str.lower()
)


# Step 1
# Remove known not meaningful patterns
def sub_patterns(patterns, field, df_):
    """ Substitute passed patterns with ' '
    """
    for pattern in patterns:
        df_[field] = (
            df_[field].apply(
                lambda x: re.sub(pattern, ' ', f'{x}')
            )
        )
    return df_


NOTES_PROC = sub_patterns(pats, 'contents', NOTES_REG_DF)

# remove urls now
# sub_patterns(regurl, 'contents', NOTES_PROC)

# Build table for manual testing of text preprocessing before
# we model the data
cont_test = pd.merge(
    NOTES_PROC['contents'],
    NOTES_DF['contents'],
    left_index=True,
    right_index=True,
    how='left'
)

########################################################
# Model the documents with TFIDF approach
# Build notes vector representattion table
print('building vector representation')

text = NOTES_PROC['contents'] 

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.99,
    min_df=1,
    norm='l2',
    smooth_idf=True,
    analyzer='word',
    ngram_range=(1, 3)
)

# vectorizer = CountVectorizer(
#     min_df=1,
#     ngram_range=(1, 3),
#     stop_words='english',
#     binary=True
# )

X = vectorizer.fit_transform(text)

# We will use features list later on to catch what tokens were taken 
# into the TFIDF calculation
features = vectorizer.get_feature_names_out()


# Construct DF
MDF = pd.DataFrame(
    X.toarray(),
    columns=features,
    index=NOTES_PROC['title']
)

# Find nearest neighbors
n_number = 10

print('Find n nearest neighbors')
X_d = X.todense()
nbrs = NearestNeighbors(
    n_neighbors=n_number+1,
    algorithm='brute',
    metric='cosine'
    # metric='jaccard'
).fit(X_d)

distances, indices = nbrs.kneighbors(X_d)

# Interface for NN query


def find_index(title):
    """Helper function to extract index from master table using given note title

    :param title: Obsidian note title
    :type title: str
    :return: index number corresponding to a given title
    :rtype: int
    """
    t = title.lower()
    return NOTES_PROC[NOTES_PROC['title']==t].index[0]

def show_nn(note_title, dropna=False):       
    """Returns nearest neighbors dataframe for a given note. The table consists of neighbors titles, tokens used by the model and their corresponding TFIDF values. 
    
    :param note_title: title of the note for which user wants to extract nns
    :type note_title: str
    :param dropna: use this if you want to see only non-zero value tokens for at least one neighbor
    :type dropna: bool
    :returns: pd.DataFrame of nearest neighbors for given note title. First column represents a note for which nearest neighbors are calculated in the next columns.
    """
    
    i = find_index(
        note_title
    )

    df_ = pd.DataFrame(indices)
    nns = df_[df_[0] == i].values
    sub_df = MDF.iloc[nns[0]].T

    if dropna:
        sub_df_s = (
            sub_df
            .replace(0, np.nan)
            .dropna(how='all')
        )
        return sub_df_s
    else:
        return sub_df


#TODO: come up with accuracy measurement (current graph edges vs recommended neighbors)
#TODO: what is wrong with note 332? Now it is 90
# answer: there are tokens that are to common or rare to be taken in by 
# vectorizer. This creates 'holes' in index list output from NN algorithm.
#TODO: interface
# Use name feature to query for index number
# find nearest neighbors in model output by index number
# use NOTES_UPD again to return nearest neighbors names


#############################
#### CLUSTERING PART TBC ####
#############################

# ## Find optimal number of clusters
# print('finding optimal clusters')
# def find_optimal_clusters(data, max_k):
#     iters = range(2, max_k+1, 2)
    
#     sse = []
#     for k in iters:
#         sse.append(MiniBatchKMeans(
#             n_clusters=k, 
#             random_state=20
#         ).fit(data).inertia_)

#         print('Fit {} clusters'.format(k))
        
#     f, ax = plt.subplots(1, 1)
#     ax.plot(iters, sse, marker='o')
#     ax.set_xlabel('Cluster Centers')
#     ax.set_xticks(iters)
#     ax.set_xticklabels(iters)
#     ax.set_ylabel('SSE')
#     ax.set_title('SSE by Cluster Center Plot')
#     # f.show()
    
# find_optimal_clusters(X, 20)


# ## Clustering
# cl_n = input('How many clusters do you want to use?: ')
# clusters = MiniBatchKMeans(
#     n_clusters=int(cl_n), 
#     random_state=20
# ).fit_predict(X)

# ## Dim reduction and plotting
# print('Formin TSNE plot')
# def plot_tsne_pca(data, labels):
#     max_label = max(labels)
#     max_items = np.random.choice(range(data.shape[0]), size=50, replace=False)
    
#     pca = PCA(n_components=2).fit_transform(data[max_items,:].todense())
#     tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data[max_items,:].todense()))
    
    
#     idx = np.random.choice(range(pca.shape[0]), size=5, replace=False)
#     label_subset = labels[max_items]
#     label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
    
#     f, ax = plt.subplots(1, 2, figsize=(14, 6))
    
#     ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
#     ax[0].set_title('PCA Cluster Plot')
    
#     ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
#     ax[1].set_title('TSNE Cluster Plot')

#     f.show()
    
# plot_tsne_pca(X, clusters)


# # Getting top keywords
# print('Getting top keywords')
# def get_top_keywords(data, clusters, labels, n_terms):
#     df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    
#     for i,r in df.iterrows():
#         print('\nCluster {}'.format(i))
#         print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
            
# get_top_keywords(X, clusters, vectorizer.get_feature_names(), 10)


print('TERMINATED')
