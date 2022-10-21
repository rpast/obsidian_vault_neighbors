# Obsidian Vault Neighbors
### Note recommendation script
 
Obsidian.md is an open source knowledge management software. As its creators describe
it: "Obsidian is a powerful and extensible knowledge base that works on top of your 
local folder of plain text files."
The Obsidian allows for forward and back-links as a way of connection between notes.
This method produces a graph. It can be traversed by users in search of further 
meaningful connections. The quality of semantic edges in the graph is dictated by 
the judgement of the user.


If you want to learn more about the Obsidian tool, check: [Obsidian.md](obsidian.md)


## Objective
The objective of this script is to mine meaningful semantic connections between
the notes in a given Obsidian vault using NLP techniques so user receives a list 
of nearest neighbours for a given note. Nearest neighbors is a collection of notes
that are closest to a given note in terms of the frequency of the language used.
User can treat the list as a recommendation for outer links or inspiration. 

## Dev steps
- [x] Parse corpus of .md documents
- [x] Data pre-processing (normialization, stemming, removal of not meaningful
tokens)
- [x] Data vectorization
- [x] Pattern mining with unsupervised learning
- [x] Operationalization of clusters (nearest neighbours preview for each note) 
