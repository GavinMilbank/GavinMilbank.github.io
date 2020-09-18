Based on a [CMU machine leaning course by William Cohen](http://www.cs.cmu.edu/~wcohen/10-601/)

## A. PROBABILITY
### 1. Probability
* Zeno's paradox addressed with positional notation,
* Probability and axioms as the One True Path to dealing with uncertainty
* Conditional probability definition
* Bayes rule as One True Way to combine beliefs and prior knowledge
* [Di Finetti and pseudo-data](https://en.wikipedia.org/wiki/De_Finetti%27s_theorem) [more notes](http://www.stats.ox.ac.uk/~steffen/teaching/grad/definetti.pdf)
* [Covariance and Central Limit Theorem](http://ttic.uchicago.edu/~dmcallester/ttic101-07/lectures/Gaussians/Gaussians.pdf)
* [PCA, SVD, LSI and Kernel PCA](http://ttic.uchicago.edu/~dmcallester/ttic101-07/lectures/PCA/PCA.pdf)
* [Convexity and Jensen's](http://ttic.uchicago.edu/~dmcallester/ttic101-07/lectures/jensen/jensen.pdf)
* (Information theory)[http://ttic.uchicago.edu/~dmcallester/ttic101-07/lectures/entropy/entropy.pdf]

### 2. Applications of Bayes rule
* MLE, MAP, Smoothing and Bayes rule
* Density estimation and classification
* Naive Bayes for binomials
* Naive Bayes for multinomials
* Naive Bayes for Gaussians
* Naives Bayes as MAP with smoothing

# B. STATISTICAL DECISION THEORY

## Multivariate regression
- keep variables where H_{0} of correlation=0 has low p-value (is unlikely).
## PAC-learning and overfitting
* Bias-variance tradeoff for linear regression
* Multiple tests and union bound; Valiant's result and PAC-learning
## Mistake-bounded analysis
* MB analysis of perceptrons
* perceptrons and SVMs
* MB compared to PAC results
## Experimental practice for classifiers
* Confidence intervals
* k-fold CV and lv-1-out
* paired tests and McNemar's test
* multiple tests, Bonferroni, ANOVA
## More theory
* Shattering, VC-dimension, empirical loss
* Occam's razor results
* Empirical loss minimization
* Regret-based analysis

# B LINEAR MODELS
## 3. Logistic regression
* Naive Bayes is a linear classifier
* Warmup: MLE estimate for a binomial
* how LogReg maximizes Log CL
* how LogReg's gradent matches expectations
* LogReg and regularization
## 4. Linear regression
* Linear regression via gradient descent
* Linear regression via normal equations
* Linear regression and regularization (ridge regression)
* Linear regression and L1 regularization (lasso)
* Logistic regression and L1

# C. NONLINEAR LEARNERS
## 9. Nonlinearity and ML
* ANN definition, network of logistic units
* Backpropagation derivation
* Expressiveness of logistic networks
* Optimization tricks for BP
## 10. Deep NN
* Autoencoders
* RBFs (restricted Boltzmann machines)
* Training deep networks
* NN architecture, Convolutional NNs, ...
## 11. Kernels and SVMs
* kernel perceptrons
* SVMs and margin optimization
* Lagrangians and constrained optimization
## 12. KNN and Decision trees
* Decision trees
    * TDIDT and pruning
* Regression trees
* Bagged decision trees
* BPETs vs logistic regression: Provost's results

# D. UNSUPERVISED AND SEMI-SUPERVISED LEARNING
## 13.  K-means like clustering methods
* Clustering: mixtures of Gaussians, mixtures of multinomials
* TFIDF and k-means
* SSL in EM setting
* Convergence of EM
## 14-15. Dimension reduction
* [](pdf/14. dimension-reduction.pdf)
* Eigenvalue calculations
* Linear regression via normal equations => PCA
* SVD
* Matrix factorization via SGD
## 16. Graph-based and EM-based SSL
* SSL k-means, mixtures, etc
* MAD and Jacoby iteration

# E. GRAPHICAL MODELS
## 17. HMMs
* Definition
* Forward-backward
* Baum-Welsh
* IE with HMMS
* HMMS for protein-modeling
## 18. Directed networks and BP
* Definition
* d-separation
* application: social contagion
* BP
* LBP, Gibbs
## 19. Topic models
* LDA
* Network models
## 20. Markov networks
* compiling directed to undirected networks
* conditional linear-chain random fields
* pseudo-likelihood

# F. ADVANCED TOPICS
## 21. Ensembles
* Stacking/Boosting
* Random forests
## 22. Scalability
* streaming learning methods
* parameter servers
* randomized methods
## 23. MDP
* [Markov decision process](https://en.wikipedia.org/wiki/Markov_decision_process)
## 24. Distant training
* [Distant supervision](http://deepdive.stanford.edu/distant_supervision)
