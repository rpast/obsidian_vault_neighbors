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

## Set paths
cwd = Path.cwd()
# vault_str = '/home/nef/Documents/nefdocs/vault1'
vault_str = r'C:\\Users\\Rafal\\Documents\\vault1'
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

# Transform empty notes
NOTES['contents'] = NOTES['contents'].str.strip()
empty_f = (NOTES['contents']=='')
NOTES['empty'] = np.nan
NOTES.loc[empty_f, 'empty'] = True
# For notes coming from shells use their stem as contents
nonsh_f = (NOTES['empty'] == True) & \
(NOTES['path'].apply(
    lambda x: re.search(r"Shells", str(x)) == None
    )
)
# filter out empty notes not from Shells folder
NOTES_UPDT = NOTES[~nonsh_f].copy()


# Remove the rest
NOTES_UPDT.loc[empty_f, 'contents'] = (
    NOTES_UPDT
    ['path'].apply(
        lambda x: x.stem
    )
)

# Create backup dataframe for testing
NOTES_bak = NOTES.copy()


print('data preprocessing')
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

NOTES_UPDT['contents'] = (
        NOTES_UPDT['contents']
        .str.lower()
)

# Remove known not meaningful patterns
def sub_patterns(patterns):
    for pattern in patterns:
        NOTES_UPDT['contents'] = (
            NOTES_UPDT['contents'].apply(
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
    NOTES_UPDT['contents'] = (
        NOTES_UPDT['contents'].apply(
            lambda x: escape(x, '\\', chr_)
        )
    )

## remove urls now
sub_patterns(regurl)


## test
cont_test = pd.merge(
    NOTES_UPDT['contents'], 
    NOTES_bak['contents'], 
    left_index=True, 
    right_index=True, 
    how='left'
)

########################################################
print('building vector representation')
## Build notes vector representattion table
text = NOTES_UPDT['contents'] 
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
FNAMES = NOTES_UPDT['path'].apply(lambda x: Path(x).stem)
MDF = pd.DataFrame(
    X.toarray(),
    columns=features,
    index=FNAMES
)


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


## NN algorithm
print('find n nearest neighbors')
X_d = X.todense()
nbrs = NearestNeighbors(
    n_neighbors=5, 
    algorithm='brute',
    metric='cosine'
).fit(X_d)

distances, indices = nbrs.kneighbors(X_d)


## Interface for NN query
def show_nn(index, dropna=False):
    df_ = pd.DataFrame(indices)
    nns = df_[df_[0] == index].values
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
#TODO: what is wrong with note 332?
#TODO: sth wrong w regurl 

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