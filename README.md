Singular (C++)
=============================
Singular is an implementation of the template for spectral word embedding
methods presented in Model-based Word Embeddings from Decompositions of Count
Matrices (Stratos et al., 2015). It shows (among other things) that canonical
correlation analysis (CCA) is a particular type of decomposition that recovers
the parameters of certain hidden Markov models (HMMs).

Highlights
-----------
1. Singular can be used to obtain high-quality word embeddings. The recommended
default setting is to perform CCA with the square-root transformation of count
data: this tends to work well out of the box for a variety of tasks such as word
similarity and analogy. They can also serve as powerful real-valued features in
a supervised language task, e.g., named-entity recognition (NER).

2. Singular subsumes various ways of applying SVD on co-occurrence counts of
word-context pairs #(w, c). The counts are typically transformed first (log,
square-root, etc.) - see the paper for details.

 **No scaling**: SVD on #(w, c).

 **PPMI**: SVD on max(log{#(w, c) / #(w) / #(c)}, 0).

 **Regression**: SVD on #(w, c) / #(w).

 **CCA**: SVD on #(w, c) / #(w)^{1/2} / #(c)^{1/2}.

3. Singular also performs *agglomerative clustering* of word embeddings (as
part of the pipeline). Therefore, it can be used to reproduce the Brown
clustering algorithm of Stratos et al. (2014): just do CCA without data
transformation (with an appropriate context definition, see below).

Pipeline
--------
Singular consists of three steps:

1. Compute co-occurrence counts from a corpus (a text file or a directory of
text files). This is achieved by sliding a "window" across the corpus. The size
of the window (`--window`) determines the size of the context. If it's 2, the
context is simply a word to the right. If it's 5, the context is two words to
the left and two words to the right. You can also choose to distinguish sentence
boundaries if the corpus has a sentence per line (`--sentences`). Finally,
contexts can be either bag-of-words (`--context bag`) or position-sensitive
(`--context list`).

2. Perform low-rank (`--dim`) SVD on a certain choice of transformation
(`--transform`) and scaling (`--scale`) of the counts.

3. Perform agglomerative clustering over the resulting word vectors.

The output of step 1 is saved so that once the statistics are collected, the
actual corpus is no longer needed. You can force recomputing the statistics
from scratch with a flag (`-f`).

Usage
-----
Type `make` to compile (clang++). The code has been used on Mac and Linux. Type
`./singular` or `./singular -h` to see all options. Possible example usages are:

* Do CCA with the square-root transformation on a large corpus that has a
sentence per line. Use the bag-of-words context definition:

`./singular --corpus [corpus] --output [output] --rare 100 --sentences --window 11 --context bag --dim 500 --transform sqrt --scale cca`

* Do CCA with no transformation on a relatively small corpus. Do not distinguish
sentence boundaries. Use the list-of-words context definition (this is the
setting in Stratos et al. (2014)):

`./singular --corpus [corpus] --output [output] --rare 6 --window 5 --context list --dim 30 --transform raw --scale cca`

In similar manners, you can try different combinations of transformation and
scaling. The resulting word vectors are stored as `output/wordvectors_*` and
the corresponding cluster bit strings are stored as `output/agglomerative_*`
(where `*` is a signature marking the configuration).