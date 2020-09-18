# Word embedding models

Neural network word embeddings versus traditional count-based distributional models. Copied from [here](https://github.com/Hironsan/awesome-embedding-models)

## Table of Contents
* **[Papers](#papers)**
* **[Researchers](#researchers)**
* **[Courses and Lectures](#courses-and-lectures)**
* **[Datasets](#datasets)**
* **[Implementations and Tools](#implementations-and-tools)**

### Existing training sets of word2vec

* [Original](https://code.google.com/archive/p/word2vec/)
* [Multilingual](https://github.com/Kyubyong/wordvectors)
* [Another multilingual](https://sites.google.com/site/rmyeid/projects/polyglot)
* [Fasttext by mikolov](https://github.com/icoxfog417/fastTextJapaneseTutorial)
* [Levy dependency based word embeddings - good for syntactic simialrity](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/)
* [meta-embeddings](http://cistern.cis.lmu.de/meta-emb/)
* [Standford Glove](http://nlp.stanford.edu/projects/glove/)
* [Lexvec](https://github.com/alexandres/lexvec)
* [word2vec](https://en.wikipedia.org/wiki/Word2vec)
* [Computation of Normalized Edit Distance and Applications](https://ai2-s2-pdfs.s3.amazonaws.com/156c/f06f920a98152668dd17a43fde9c68fc0d9b.pdf)

## Papers
[All papers](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=)

### Word Embeddings

#### Standard - Word2vec, GloVe, FastText

* [Efficient Estimation of Word Representations in Vector Space (2013), T. Mikolov et al.](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=efficient-estimation-of-word-representations.pdf)
* [Distributed Representations of Words and Phrases and their Compositionality (2013), T. Mikolov et al.](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=distributed-representations-of-words.pdf)
* [word2vec Parameter Learning Explained (2014), Xin Rong](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=1411.2738.pdf)
* [word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method (2014), Yoav Goldberg, Omer Levy](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=1402.3722.pdf)
* [GloVe: Global Vectors for Word Representation (2014), J. Pennington et al.](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=glove.pdf)
* [Improving Word Representations via Global Context and Multiple Word Prototypes (2012), EH Huang et al.](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=HuangACL12.pdf)
* [Enriching Word Vectors with Subword Information (2016), P. Bojanowski et al.](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=1607.04606v1.pdf)
* [Bag of Tricks for Efficient Text Classification (2016), A. Joulin et al.](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=1607.01759.pdf)

#### Embedding Enhancement

* [Retrofitting Word Vectors to Semantic Lexicons (2014), M. Faruqui et al.](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F/1411.4166.pdf)
* [Better Word Representations with Recursive Neural Networks for Morphology (2013), T.Luong et al.](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=conll13_morpho.pdf)
* [Dependency-Based Word Embeddings (2014), Omer Levy, Yoav Goldberg](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=dependency-based-word-embeddings-acl-2014.pdf)
* [Not All Neural Embeddings are Born Equal (2014), F. Hill et al.](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=1410.0718.pdf)
* [Two/Too Simple Adaptations of Word2Vec for Syntax Problems (2015), W. Ling](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=naacl2015.pdf)

#### Comparing count-based vs predict-based method

* [Linguistic Regularities in Sparse and Explicit Word Representations (2014), Omer Levy, Yoav Goldberg](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=conll2014analogies.pdf)
* [Don’t count, predict! A systematic comparison of context-counting vs. context-predicting semantic vectors (2014), M. Baroni](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=P14-1023.pdf)
* [Improving Distributional Similarity with Lessons Learned from Word Embeddings (2015), Omer Levy](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=Q15-1016.pdf)

**Evaluation, Analysis**

* [Evaluation methods for unsupervised word embeddings (2015), T. Schnabel](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=D15-1036.pdf)
* [Intrinsic Evaluation of Word Vectors Fails to Predict Extrinsic Performance (2016), B. Chiu](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=W16-2501.pdf)
* [Problems With Evaluation of Word Embeddings Using Word Similarity Tasks (2016), M. Faruqui](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=1605.02276.pdf)
* [Improving Reliability of Word Similarity Evaluation by Redesigning Annotation Task and Performance Measure (2016), Oded Avraham, Yoav Goldberg](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=1611.03641.pdf)
* [Evaluating Word Embeddings Using a Representative Suite of Practical Tasks (2016), N. Nayak](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=2016-acl-veceval.pdf)

### Phrase, Sentence and Document Embeddings

**Sentence**

* [Skip-Thought Vectors](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=1506.06726.pdf)

**Document**

* [Distributed Representations of Sentences and Documents](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=1405.4053)

### Sense Embeddings

* [SENSEMBED: Learning Sense Embeddings for Word and Relational Similarity](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=ACL_2015_Iacobaccietal.pdf)
* [Multi-Prototype Vector-Space Models of Word Meaning](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=reisinger.naacl-2010.pdf)

### Neural Language Models

* [Recurrent neural network based language model](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=mikolov_interspeech2010_IS100722.pdf)
* [A Neural Probabilistic Language Model](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=bengio03a.pdf)
* [Linguistic Regularities in Continuous Space Word Representations](https://www.dropbox.com/sh/rfs43j74dcqv76v/AACxd4gWlIfa38caWjXPXs1pa?dl=0%2F&preview=N13-1090.pdf)

## Researchers

* [Tomas Mikolov](https://scholar.google.co.jp/citations?user=oBu8kMMAAAAJ&hl=en)
* [Yoshua Bengio](https://scholar.google.co.jp/citations?user=kukA0LcAAAAJ&hl=en)
* [Yoav Goldberg](https://scholar.google.co.jp/citations?user=0rskDKgAAAAJ&hl=en)
* [Omer Levy](https://scholar.google.co.jp/citations?user=PZVd2h8AAAAJ&hl=en)
* [Kai Chen](https://scholar.google.co.jp/citations?user=TKvd_Z4AAAAJ&hl=en)

## Courses and Lectures

* [CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/index.html)
* [Udacity Deep Learning](https://www.udacity.com/course/deep-learning--ud730)

## Datasets
### Training

* [Wikipedia](https://dumps.wikimedia.org/enwiki/)
* [WestburyLab.wikicorp.201004](http://www.socher.org/index.php/Main/ImprovingWordRepresentationsViaGlobalContextAndMultipleWordPrototypes)

### Evaluation

* [SemEval-2012 Task 2](https://www.cs.york.ac.uk/semeval-2012/task2.html)
* [WordSimilarity-353](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/)
* [Stanford's Contextual Word Similarities (SCWS)](http://www.socher.org/index.php/Main/ImprovingWordRepresentationsViaGlobalContextAndMultipleWordPrototypes)
* [Stanford Rare Word (RW) Similarity Dataset](http://stanford.edu/~lmthang/morphoNLM/)

### Pre-Trained Word Vectors
Convenient downloader for pre-trained word vectors:
* [chakin](https://github.com/chakki-works/chakin)


Links for pre-trained word vectors:
* [Word2vec pretrained vector(English Only)](https://code.google.com/archive/p/word2vec/)
* [Word2vec pretrained vectors for 30+ languages](https://github.com/Kyubyong/wordvectors)
* [FastText pretrained vectors for 90 languages](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)
* [FastText pretrained vector for Japanese with NEologd](https://drive.google.com/open?id=0ByFQ96A4DgSPUm9wVWRLdm5qbmc)
* [word vectors trained by GloVe](http://nlp.stanford.edu/projects/glove/)
* [Dependency-Based Word Embeddings](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/)
* [Meta-Embeddings](http://cistern.cis.lmu.de/meta-emb/)
* [Lex-Vec](https://github.com/alexandres/lexvec)
* [Huang et al. (2012)'s embeddings (HSMN+csmRNN)](http://stanford.edu/~lmthang/morphoNLM/)
* [Collobert et al. (2011)'s embeddings (CW+csmRNN)](http://stanford.edu/~lmthang/morphoNLM/)
* [BPEmb: subword embeddings for 275 languages](https://github.com/bheinzerling/bpemb)

## Implementations and Tools
### Word2vec

* [Original](https://code.google.com/archive/p/word2vec/)
* [gensim](https://radimrehurek.com/gensim/models/word2vec.html)
* [TensorFlow](https://www.tensorflow.org/versions/r0.12/tutorials/word2vec/index.html)

### GloVe

* [Original](https://github.com/stanfordnlp/GloVe)
