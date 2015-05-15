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

Scripts
-------
You might find the scripts under the folder `scripts/` useful. These are Python
scripts for useful tasks such as evaluating similarity scores
(`evaluate_similarity.py`), answering analogy questions (`evaluate_analogy.py`),
and interactively displaying nearest neighbors (`show_similar.py`). Some are
version 3.0 and others are 2.7 so be careful.

### Some nearest neighbor examples with my own embeddings

`python scripts/show_similar.py ../scratch/singular_wiki/wordvectors_rare100_spl_window11_bag_hash0_dim500_sqrt_cca_pseudo0_ce0P75_se0`

`Read 188302 embeddings of dimension 500`

`Type a word (or just quit the program):` amazing

        0.8094		incredible
		0.7642		astonishing
		0.6411		fantastic
		0.5984		wonderful
		0.5894		awesome
		0.5860		astounding
		0.5844		unbelievable
		0.5817		uncanny
		0.5717		terrific
		0.5646		remarkable
		0.5443		extraordinary
		0.5415		impressive
		0.5166		superb
		0.5163		marvellous
		0.5047		unparalleled
		0.5023		exceptional
		0.4938		marvelous
		0.4933		phenomenal
		0.4781		stupendous
		0.4667		incomparable
		0.4569		spectacular
		0.4563		breathtaking
		0.4488		startling
		0.4472		excellent
		0.4448		stunning
		0.4445		superhuman
		0.4435		marvels
		0.4369		awe-inspiring
		0.4336		amazed
		0.4291		super-human


`Type a word (or just quit the program):` intolerable

		0.6881		unbearable
		0.5984		appalling
		0.5955		deplorable
		0.5763		unjust
		0.5378		unacceptable
		0.5241		unjustifiable
		0.5196		repugnant
		0.5156		abhorrent
		0.5135		inexcusable
		0.5072		atrocious
		0.5036		miserable
		0.4990		reprehensible
		0.4979		disgraceful
		0.4953		odious
		0.4903		shameful
		0.4851		unspeakable
		0.4821		inhuman
		0.4782		unreasonable
		0.4776		counter-productive
		0.4754		inhumane
		0.4716		unjustified
		0.4675		oppressive
		0.4662		abject
		0.4639		unimaginable
		0.4549		inescapable
		0.4531		endure
		0.4525		unwarranted
		0.4520		aggravating
		0.4509		unpleasant
		0.4497		undignified


`Type a word (or just quit the program):` rochester

		0.6702		binghamton
		0.6556		albany
		0.6463		hartford
		0.6452		utica
		0.6233		syracuse
		0.6201		elmira
		0.5795		bridgeport
		0.5762		newark
		0.5488		watertown
		0.5300		haven
		0.5231		ogdensburg
		0.5168		burlington
		0.5128		danbury
		0.5115		worcester
		0.5109		erie
		0.5088		buffalo
		0.5073		detroit
		0.5051		ithaca
		0.5021		waterbury
		0.5006		ny
		0.4999		york
		0.4997		trenton
		0.4945		boston
		0.4927		poughkeepsie
		0.4876		brooklyn
		0.4855		springfield
		0.4846		niagara
		0.4824		chicago
		0.4796		akron
		0.4739		newburgh
