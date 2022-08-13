# Obsidian Vault Clustering
Obsidian.md is useful knowledge management software that allows users to connect 
create notes in a form of markdown files, and connect them with forward and 
back-links. This produces a graph that can be traversed by user in search of 
meaningful connections. The semantic quality of edges in the graph is dictated
only by linkage method utilised by the user. This may result with suboptimal
results. 

## Objective
The objective of this exercise is to mine meaningful semantic connections between
the notes using NLP techniques so user will receive a list of nearest neighbours 
for a given note. That list will be treated as a suggestion of links that could
be formed by the user.

## Dev steps
- [ ] Parse corpus of .md documents
- [ ] Data pre-processing (normialization, stemming, removal of not meaningful
tokens)
- [ ] Data vectorization
- [ ] Pattern mining with unsupervised learning
- [ ] Operationalization of clusters (nearest neighbours preview for each note) 
