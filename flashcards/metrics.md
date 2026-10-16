# Flashcards: ML Evaluation Metrics

## Confusion Matrix and Basic Rates

- What are the four cells of the confusion matrix in binary classification? == True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
- What is a false positive (Type I error)? == The model predicts "positive" but the true class is negative (a false alarm)
- What is a false negative (Type II error)? == The model predicts "negative" but the true class is positive (a miss)
- Formula for accuracy? == (TP + TN) / (TP + TN + FP + FN), the share of all examples classified correctly
- What is the accuracy paradox? == With strong class imbalance (e.g. 99% negatives) a model that always predicts "negative" reaches 99% accuracy without learning anything
- Formula for precision (positive predictive value)? == TP / (TP + FP), the share of positive predictions that are actually positive
- Formula for recall (sensitivity, true positive rate)? == TP / (TP + FN), the share of actual positives that are found
- Formula for specificity (true negative rate)? == TN / (TN + FP), the share of actual negatives correctly predicted as negative
- Formula for the false positive rate (FPR)? == FP / (FP + TN) = 1 − specificity
- Formula for the false negative rate (FNR)? == FN / (FN + TP) = 1 − recall
- Which of precision, recall and specificity depends on prevalence (share of positives)? == Precision; recall and specificity are each computed within one true class and do not depend on prevalence
- When is precision more important than recall? == When false positives are expensive, e.g. spam filters (an important email lands in spam) or automatic account bans
- When is recall more important than precision? == When false negatives are expensive, e.g. cancer screening, fraud detection, safety filters
- How does raising the decision threshold typically change precision and recall? == Precision tends to rise and recall falls (fewer but more confident positive predictions)

## Composite Classification Metrics

- Formula for the F1 score? == 2 · precision · recall / (precision + recall) = 2TP / (2TP + FP + FN)
- Why does F1 use the harmonic rather than the arithmetic mean? == The harmonic mean is dominated by the smaller value, so a model with recall 1 and precision 0.01 does not get a good score
- Which confusion-matrix cell does the F1 score ignore? == True negatives (TN)
- Formula for the F-beta score? == (1 + β²) · P · R / (β² · P + R)
- What do β > 1 and β < 1 mean in the F-beta score? == β > 1 weights recall more (e.g. F2), β < 1 weights precision more (e.g. F0.5)
- Formula for balanced accuracy (binary)? == (TPR + TNR) / 2; for multiclass, the mean of per-class recall
- Formula for the Matthews correlation coefficient (MCC)? == (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
- Range and interpretation of the MCC? == −1 (total disagreement) to +1 (perfect), 0 equals chance; uses all four cells and is robust to class imbalance
- Formula for Cohen's kappa? == κ = (p_o − p_e) / (1 − p_e), with observed agreement p_o and agreement expected by chance p_e
- What is Cohen's kappa typically used for? == Agreement between two annotators (inter-annotator agreement) or model vs. label, corrected for chance agreement
- What is Youden's J? == J = sensitivity + specificity − 1 = TPR − FPR; the threshold with maximal J is a common operating point on the ROC curve

## Multiclass Averaging

- What is macro averaging? == Compute the metric per class and take the unweighted mean; every class counts equally, so rare classes have a large influence
- What is micro averaging? == Sum TP, FP and FN over all classes and compute the metric from the totals; frequent classes dominate
- What is weighted averaging? == Per-class metric weighted by support (number of true examples of each class)
- What does micro-F1 equal in single-label multiclass classification? == Accuracy (micro-precision = micro-recall = micro-F1 = accuracy)
- What is top-k accuracy? == The share of examples whose true class is among the k classes with the highest scores

## ROC, PR Curves and Thresholds

- What does the ROC curve show? == TPR (y-axis) against FPR (x-axis) over all possible thresholds
- Probabilistic interpretation of ROC-AUC? == The probability that a random positive example gets a higher score than a random negative example
- What ROC-AUC does a random classifier have, and a perfect one? == 0.5 and 1.0
- What does a ROC-AUC clearly below 0.5 mean? == The ranking is systematically inverted; flipping the scores would give a good model
- Why can ROC-AUC look too optimistic under strong class imbalance? == FPR has the many negatives in its denominator, so even many false positives barely move it while precision collapses
- What does the precision-recall curve show? == Precision (y-axis) against recall (x-axis) over all thresholds
- Where is the baseline of the PR curve (random classifier)? == At the prevalence, i.e. the share of positive examples (not at 0.5)
- When should you report PR-AUC instead of ROC-AUC? == When the positive class is rare and the quality of positive predictions matters (e.g. fraud, retrieval, anomalies)
- How is average precision (AP) computed? == AP = Σ_n (R_n − R_(n−1)) · P_n, a step-wise area under the PR curve
- What is TPR at a fixed FPR (e.g. TPR@1% FPR)? == Recall at the threshold where the FPR reaches a fixed maximum; more practical than AUC when there is a false-alarm budget
- Why must you not tune the threshold on the test set? == It is data leakage and biases the reported metric upwards; choose the threshold on validation data and only measure on the test set
- Why are ROC-AUC and PR-AUC threshold-independent but not sufficient on their own for a product? == They score the ranking over all thresholds, but production runs exactly one threshold whose precision and recall must be reported separately

## Probabilistic Predictions and Calibration

- Formula for log loss (binary cross-entropy)? == −(1/N) Σ [y · log(p) + (1 − y) · log(1 − p)]
- What does log loss punish especially hard? == Confident wrong predictions (p near 0 or 1 on the wrong side); the loss goes to infinity
- Formula for the Brier score? == (1/N) Σ (p − y)², the mean squared error of the predicted probability; 0 is perfect
- What is a proper scoring rule? == A scoring rule whose expected value is optimised by predicting the true probabilities (e.g. log loss, Brier score)
- What does "well calibrated" mean? == Of all predictions with probability 0.8, about 80% are actually positive
- How is the expected calibration error (ECE) computed? == Bin predictions by confidence, compute |accuracy − mean confidence| per bin, and average weighted by bin size
- What does a reliability diagram show? == Observed accuracy against predicted confidence per bin; perfect calibration lies on the diagonal
- What is temperature scaling? == Post-hoc calibration that divides the logits by a scalar T learned on validation data; changes confidence but not the argmax prediction

## Regression Metrics

- Formula for mean absolute error (MAE)? == (1/N) Σ |y − ŷ|
- Formula for mean squared error (MSE)? == (1/N) Σ (y − ŷ)²
- Formula for root mean squared error (RMSE) and its advantage over MSE? == √MSE; it has the same unit as the target
- MAE vs. RMSE with outliers? == RMSE squares errors and punishes large errors disproportionately; MAE is more robust to outliers
- Which constant prediction minimises MAE, and which minimises MSE? == MAE: the median; MSE: the mean
- Formula for the coefficient of determination R²? == 1 − SS_res / SS_tot = 1 − Σ(y − ŷ)² / Σ(y − ȳ)²
- Can R² be negative? == Yes, on test data, when the model is worse than always predicting the mean
- What is adjusted R² for? == It penalises extra features: 1 − (1 − R²)(n − 1)/(n − p − 1); it only rises if a feature explains more than chance
- Formula for MAPE? == (100%/N) Σ |y − ŷ| / |y|
- Two weaknesses of MAPE? == Undefined or exploding when y is near 0; asymmetric, because over-forecasts are penalised without bound while under-forecasts (for positive y) are capped at 100%
- Formula for sMAPE? == (100%/N) Σ 2|y − ŷ| / (|y| + |ŷ|), a symmetric variant of MAPE ranging from 0 to 200%
- What is MASE? == The model's MAE divided by the in-sample MAE of a naive (seasonal) forecast; < 1 means better than naive, scale-free and defined with zeros
- Formula for RMSLE and when to use it? == √((1/N) Σ (log(1 + ŷ) − log(1 + y))²); measures relative errors for targets spanning orders of magnitude and penalises under-estimation more than over-estimation
- What is the Huber loss? == Quadratic for errors below δ and linear above; combines the smoothness of MSE with the outlier robustness of MAE
- What does the pinball (quantile) loss measure? == The quality of a quantile forecast: under-prediction is weighted by τ, over-prediction by (1 − τ); for τ = 0.5 it is proportional to MAE

## Ranking and Information Retrieval

- Formula for precision@k? == Number of relevant results in the top k / k
- Formula for recall@k? == Number of relevant results in the top k / total number of relevant documents
- What is hit rate@k? == The share of queries with at least one relevant document in the top k
- Formula for mean reciprocal rank (MRR)? == The mean over queries of 1 / rank of the first relevant result
- What is mean average precision (MAP) in retrieval? == The mean of average precision over all queries; AP sums precision@k at every position k holding a relevant result and divides by the number of all relevant documents
- Formula for DCG@k? == Σ_(i=1..k) rel_i / log₂(i + 1); relevant results higher up count more
- What is NDCG@k? == DCG@k / IDCG@k, normalised by the ideal ranking; ranges from 0 to 1 and supports graded relevance
- When NDCG instead of MRR? == NDCG for graded relevance and several relevant results; MRR when only the first relevant result matters (e.g. question answering, navigation)

## NLP and Generative Models

- How does BLEU work? == Geometric mean of modified n-gram precision (n = 1 to 4) between candidate and references, multiplied by a brevity penalty
- What is the brevity penalty in BLEU for? == It punishes candidates that are too short, which could otherwise reach high precision with a few safe words
- How does ROUGE work and what is it used for? == Recall-oriented n-gram overlap with references (ROUGE-1, ROUGE-2), mainly for summarisation
- What does ROUGE-L measure? == Overlap via the longest common subsequence of candidate and reference
- What is METEOR? == An MT metric with unigram matching including stemming and synonyms, a recall-weighted F-mean and a fragmentation penalty for word order
- What is chrF? == An F-score over character n-grams; more robust than BLEU for morphologically rich languages such as German
- How does BERTScore work? == Tokens of candidate and reference are greedily matched by cosine similarity of contextual embeddings, giving precision, recall and F1
- Main weakness of n-gram metrics such as BLEU and ROUGE? == They measure surface overlap, so correct paraphrases are penalised and meaning-distorting answers with many shared words are rewarded
- Formula for perplexity? == exp(−(1/N) Σ log p(x_i | x_<i)), the exponential of the mean negative log-likelihood per token; lower is better
- Why are perplexity values of different language models often not comparable? == They depend on the tokenizer and vocabulary; they are only comparable with the same tokenisation and test set
- What is exact match (EM) in question answering? == The share of answers that exactly match the reference after normalisation (lowercasing, punctuation, articles)
- What is token F1 in question answering (SQuAD)? == F1 over the overlap of the token sets of answer and reference; gives partial credit instead of all or nothing
- Formula for word error rate (WER)? == (S + D + I) / N with substitutions, deletions, insertions and N reference words; can exceed 1
- What does pass@k measure in code generation? == The probability that at least one of k generated solutions passes all unit tests
- Unbiased estimator for pass@k with n samples of which c are correct? == 1 − C(n − c, k) / C(n, k)
- Typical biases of LLM-as-a-judge? == Position bias (order), verbosity bias (longer answers preferred), self-preference (own model family preferred)
- How do you reduce position bias in pairwise LLM-judge comparisons? == Judge each pair in both orders and only count consistent verdicts, or average them
- What does faithfulness (groundedness) measure in RAG? == The share of claims in the answer that are supported by the retrieved context; measures hallucination relative to the context
- What do context precision and context recall measure in RAG? == Context precision: whether retrieved chunks are relevant and relevant ones are ranked high; context recall: whether the context contains everything needed for the reference answer

## Computer Vision

- Formula for intersection over union (IoU)? == Area of overlap / area of union of predicted and true box or mask
- When does a detection count as a true positive? == When the class is right and its IoU with a not-yet-matched ground-truth box is above the threshold (e.g. 0.5)
- What is mAP in object detection? == Average precision per class averaged over all classes; COCO also averages over IoU thresholds 0.50 to 0.95 in steps of 0.05
- Formula for the Dice coefficient and its relation to F1 and IoU? == 2|A ∩ B| / (|A| + |B|); equals the pixel-level F1 score, and Dice = 2 · IoU / (1 + IoU)

## Clustering

- Formula for the silhouette coefficient of a point? == s = (b − a) / max(a, b), with a = mean distance within its own cluster and b = mean distance to the nearest other cluster; ranges from −1 to 1
- What is the adjusted Rand index (ARI)? == Pairwise agreement of two clusterings corrected for chance; 1 is perfect, about 0 is random, requires ground-truth labels
- What is normalized mutual information (NMI)? == Mutual information between clustering and labels normalised to 0 to 1; requires ground-truth labels
- Davies-Bouldin vs. Calinski-Harabasz index: which direction is better? == Davies-Bouldin: lower is better; Calinski-Harabasz: higher is better; neither needs labels
- What is the elbow method? == Plot inertia (sum of squared distances to centroids) against the number of clusters k and pick the k at the bend

## Evaluation Methodology and Statistics

- What are the training, validation and test sets each for? == Training: learn parameters; validation: choose hyperparameters and thresholds; test: a single unbiased estimate of final performance
- What is data leakage in evaluation? == Information from the test set reaches training or model decisions (e.g. duplicates, normalising on all data, tuning on the test set), making metrics too optimistic
- What is k-fold cross-validation? == Split the data into k folds, train k times on k − 1 folds and evaluate on the remaining fold, then average the results
- When use stratified cross-validation? == With imbalanced classes, so every fold has the same class distribution as the full dataset
- How do you evaluate time-series models correctly? == Time-ordered splits (rolling or expanding window): always train on the past and test on the future, no random shuffling
- What is group k-fold and when do you need it? == All examples of a group (e.g. patient, user, translations of the same text) go into the same fold; prevents leakage through correlated examples
- Formula for the Wald confidence interval of a proportion? == p̂ ± z · √(p̂(1 − p̂)/n)
- Why use a Wilson instead of a Wald interval for metrics such as accuracy, recall or FPR? == Wald fails for small n or p̂ near 0 or 1 (the interval collapses to width 0 or leaves [0, 1]); Wilson stays within [0, 1] and has better coverage
- What does the rule of three say? == With 0 observed events in n trials, the upper 95% bound of the event rate is about 3/n
- How does a bootstrap confidence interval for a metric work? == Resample the test set with replacement many times, recompute the metric each time, and take percentiles (e.g. 2.5% and 97.5%) as the interval
- Which test compares two classifiers on the same test set? == McNemar's test; it only uses the examples where exactly one of the two models is right
- What does Goodhart's law say about metrics? == "When a measure becomes a target, it ceases to be a good measure"; optimising a proxy metric decouples it from the real goal
- Why report several seeds or runs in benchmarks? == Variance from initialisation, sampling and data order can exceed the measured difference; mean and spread show whether a gain is real
