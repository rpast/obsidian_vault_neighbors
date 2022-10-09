import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import re
from sklearn import neighbors

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# from nltk.stem import PorterStemmer
# from nltk.stem import LancasterStemmer

# pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
# pd.set_option('display.width', 1000)


## Set paths
cwd = Path.cwd()
vault_str = '/home/nef/Documents/nefdocs/vault1'
# vault_str = r'C:\\Users\\Rafal\\Documents\\vault1'
vault_pth = Path(vault_str)

print('parsing vault')
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

## PREPROCESSING ##
print('data preprocessing')

# Transform empty notes
NOTES['empty'] = np.nan

# Set filtering logic
empty_f = (NOTES['contents']=='')
nonsh_f = (NOTES['empty'] == True) & \
(NOTES['path'].apply(
    lambda x: re.search(r"Shells", str(x)) == None
    )
)
NOTES['title'] = NOTES['path'].apply(
        lambda x: x.stem.lower()
    )

NOTES['contents'] = NOTES['contents'].str.strip()

# Create helper feature
NOTES.loc[empty_f, 'empty'] = True

# Filter out empty notes not from Shells folder
NOTES_UPDT = NOTES[~nonsh_f].copy()

# For notes coming from Shell folder -  use their stem as contents
NOTES_UPDT.loc[empty_f, 'contents'] = (
    NOTES_UPDT
    ['path'].apply(
        lambda x: x.stem
    )
)

# Create backup dataframe for manual testing
NOTES_bak = NOTES.copy()

# Tackle stop words/patterns
pats = [
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
    r'(\d\d-\d\d-\d\d)', #match dat pattern 2
    r'(\d\d)',
    r'({.*})' #match special template expressions
]
regurl = [r'(\S+\.(com|net|org|edu|gov|pl|eu|de)(\/\S+)?)'] #match url

# Text preprocessing
NOTES_UPDT['contents'] = (
    NOTES_UPDT
    ['contents']
    .str.strip()
)
NOTES_UPDT['contents'] = (
        NOTES_UPDT['contents']
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
                lambda x: re.sub(pattern, ' ', fr'{x}')
            )
        )
    return df_

NOTES_PROC = sub_patterns(pats, 'contents', NOTES_UPDT)

# Step 2
# Escape special chars so we can match urls
schars = [
    ':',
    '/',
]

escape = lambda s, escapechar, specialchars: "".join(
    escapechar + c if c in specialchars or c == escapechar else c for c in s
    )


NOTES_PROC['contents'] = (
    NOTES_PROC['contents'].apply(
        lambda x: escape(x, '\\', schars)
    )
)

## remove urls now
sub_patterns(regurl, 'contents', NOTES_PROC)

# Build table for manual testing of text preprocessing before
# we model the data
cont_test = pd.merge(
    NOTES_PROC['contents'], 
    NOTES_bak['contents'], 
    left_index=True, 
    right_index=True, 
    how='left'
)

########################################################
## Model the documents with TFIDF approach
## Build notes vector representattion table
print('building vector representation')

text = NOTES_PROC['contents'] 

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.99,
    min_df=0.01,
    norm='l2',
    smooth_idf=True,
    analyzer='word', 
    ngram_range=(1, 4))

X = vectorizer.fit_transform(text)

# We will use features list later on to catch what tokens were taken 
# into the TFIDF calculation
features = vectorizer.get_feature_names_out()


## Construct DF
MDF = pd.DataFrame(
    X.toarray(),
    columns=features,
    index=NOTES_PROC['title']
)

## Find nearest neighbors
print('Find n nearest neighbors')
X_d = X.todense()
nbrs = NearestNeighbors(
    n_neighbors=5, 
    algorithm='brute',
    metric='cosine'
).fit(X_d)

distances, indices = nbrs.kneighbors(X_d)

## Interface for NN query
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