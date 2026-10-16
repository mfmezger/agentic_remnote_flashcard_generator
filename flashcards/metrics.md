# Flashcards: ML-Evaluationsmetriken

## Konfusionsmatrix und Grundgrößen

- Welche vier Zellen hat die Konfusionsmatrix bei binärer Klassifikation? == True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
- Was ist in False Positive (Fehler 1. Art)? == Das Model sagt "positive", die wahre Klasse ist negative (Fehlalarm)
- Was ist in False Negative (Fehler 2. Art)? == Das Model sagt "negative", die wahre Klasse ist positive (verpasster Treffer)
- Formel der Accuracy? == (TP + TN) / (TP + TN + FP + FN), Anteil aller korrekt klassifizierten Beispiele
- Was ist das Accuracy-Paradox? == Bei starkem Klassenungleichgewicht (z. B. 99 % negative) erreicht in Model, das immer "negative" sagt, 99 % Accuracy, ohne etwas gelernt zu haben
- Formel der Precision (Positive Predictive Value)? == TP / (TP + FP), Anteil der positiven Vorhersagen, die wirklich positive sind
- Formel des Recall (Sensitivity, True Positive Rate)? == TP / (TP + FN), Anteil der echten Positiven, die gefunden werden
- Formel der Specificity (True Negative Rate)? == TN / (TN + FP), Anteil der echten Negativen, die korrekt also negative erkannt werden
- Formel der False Positive Rate (FPR)? == FP / (FP + TN) = 1 − Specificity
- Formel der False Negative Rate (FNR)? == FN / (FN + TP) = 1 − Recall
- Welche der Größen Precision, Recall, Specificity hängt von der Prävalenz (Anteil Positiver) ab? == Precision; Recall und Specificity werden jeweils nur innerhalb einer wahren Klasse berechnet und sind prävalenzunabhängig
- Wann ist Precision wichtiger also Recall? == Wenn False Positives teuer sind, z. B. Spam-Filter (wichtige Mail landet im Spam) oder automatische Sperrungen
- Wann ist Recall wichtiger also Precision? == Wenn False Negatives teuer sind, z. B. Krebs-Screening, Betrugserkennung, Safety-Filter
- Wie verändert in höherer Entscheidungs-Threshold typischerweise Precision und Recall? == Precision steigt tendenziell, Recall sinkt (weniger, aber sicherere positive Vorhersagen)

## Zusammengesetzte Klassifikationsmetriken

- Formel des F1-Scores? == 2 · Precision · Recall / (Precision + Recall) = 2TP / (2TP + FP + FN)
- Warum nutzt der F1-Score das harmonische statt des arithmetischen Mittels? == Das harmonische Mittel wird vom kleineren Wert dominiert; in Model mit Recall 1 und Precision 0,01 bekommt so keinen guten Score
- Welche Zelle der Konfusionsmatrix ignoriert der F1-Score? == True Negatives (TN)
- Formel des F-beta-Scores? == (1 + β²) · P · R / (β² · P + R)
- Was bedeutet β > 1 bzw. β < 1 beim F-beta-Score? == β > 1 gewichtet Recall stärker (z. B. F2), β < 1 gewichtet Precision stärker (z. B. F0.5)
- Formel der Balanced Accuracy (binär)? == (TPR + TNR) / 2; bei Multiklasse der Mittelwert des Recalls pro Klasse
- Formel des Matthews Correlation Coefficient (MCC)? == (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
- Wertebereich und Interpretation des MCC? == −1 (komplett falsch) bis +1 (perfekt), 0 entspricht Zufall; nutzt alle vier Zellen und ist robust bei Klassenungleichgewicht
- Formel von Cohen's Kappa? == κ = (p_o − p_e) / (1 − p_e), mit beobachteter Übereinstimmung p_o und zufällig erwarteter Übereinstimmung p_e
- Wofür wird Cohen's Kappa typischerweise verwendet? == Übereinstimmung zwischen zwei Annotatoren (Inter-Annotator-Agreement) oder Model vs. Label, korrigiert um Zufallsübereinstimmung
- Was ist Youden's J? == J = Sensitivity + Specificity − 1 = TPR − FPR; der Threshold mit maximalem J ist in gängiger Arbeitspunkt auf der ROC-Kurve

## Multiklassen-Mittelung

- Was ist Macro-Averaging? == Metrik pro Klasse berechnen und ungewichtet mitteln; jede Klasse zählt gleich, seltene Klassen haben großen Einfluss
- Was ist Micro-Averaging? == TP, FP und FN über alle Klassen aufsummieren und daraus die Metrik berechnen; häufige Klassen dominieren
- Was ist Weighted-Averaging? == Metrik pro Klasse, gewichtet mit dem Support (Anzahl wahrer Beispiele der Klasse)
- Welchem Wert entspricht Micro-F1 bei Single-Label-Multiklassen-Klassifikation? == Der Accuracy (Micro-Precision = Micro-Recall = Micro-F1 = Accuracy)
- Was ist Top-k-Accuracy? == Anteil der Beispiele, bei denen die wahre Klasse under den k Klassen mit den höchsten Scores ist

## ROC, PR-Kurve und Schwellwerte

- Was zeigt die ROC-Kurve? == TPR (y-Achse) gegen FPR (x-Achse) für alle möglichen Thresholds
- Probabilistische Interpretation der ROC-AUC? == Wahrscheinlichkeit, dass in zufälliges positives Beispiel einen höheren Score bekommt also in zufälliges negatives
- Welche ROC-AUC hat in Zufallsklassifikator, welche in perfekter? == 0,5 bzw. 1,0
- Was bedeutet eine ROC-AUC deutlich under 0,5? == Das Ranking ist systematic invertiert; Scores umdrehen würde in gutes Model ergeben
- Warum kann ROC-AUC bei starkem Klassenungleichgewicht zu optimistic wirken? == Die FPR hat die vielen Negativen im Nenner; selbst viele False Positives bewegen sie kaum, obwohl die Precision einbricht
- Was zeigt die Precision-Recall-Kurve? == Precision (y-Achse) gegen Recall (x-Achse) über alle Thresholds
- Wo liegt die Baseline der PR-Kurve (Zufallsklassifikator)? == Auf Höhe der Prävalenz, also des Anteils positiver Beispiele (nicht bei 0,5)
- Wann sollte man PR-AUC statt ROC-AUC berichten? == Bei seltener positiver Klasse, wenn die Qualität der positiven Vorhersagen zählt (z. B. Betrug, Retrieval, Anomalien)
- Wie berechnet sich Average Precision (AP)? == AP = Σ_n (R_n − R_(n−1)) · P_n, eine stufenweise Fläche under der PR-Kurve
- Was ist TPR at fixed FPR (z. B. TPR@1 % FPR)? == Recall an dem Threshold, bei dem die FPR einen festen Höchstwert erreicht; praxisnäher also AUC, wenn in Fehlalarm-Budget vorgegeben ist
- Warum darf man den Threshold nicht auf dem Testset tunen? == Das ist Data Leakage; die berichtete Metrik wird optimistic verzerrt. Threshold auf Validierungsdaten wählen, auf dem Testset nur messen
- Warum sind ROC-AUC und PR-AUC threshold-unabhängig, aber für in Product allein nicht ausreichend? == Sie bewerten das Ranking über alle Thresholds; in Production läuft aber genau in Threshold, dessen Precision/Recall man separate berichten muss

## Probabilistische Vorhersagen und Kalibrierung

- Formel des Log Loss (Binary Cross-Entropy)? == −(1/N) Σ [y · log(p) + (1 − y) · log(1 − p)]
- Was bestraft Log Loss besonders stark? == Selbstsichere falsche Vorhersagen (p nahe 0 oder 1 auf der falschen Seite); der Verlust geht gegen unendlich
- Formel des Brier Scores? == (1/N) Σ (p − y)², mittlerer quadratischer Fehler der vorhergesagten Wahrscheinlichkeit; 0 ist perfekt
- Was ist eine Proper Scoring Rule? == Eine Bewertungsregel, deren Erwartungswert optimiert wird, wenn die wahren Wahrscheinlichkeiten vorhergesagt werden (z. B. Log Loss, Brier Score)
- Was bedeutet "gut kalibriert"? == Von allen Vorhersagen mit Wahrscheinlichkeit 0,8 sind auch etwa 80 % tatsächlich positive
- Wie berechnet sich der Expected Calibration Error (ECE)? == Vorhersagen nach Confidence in Bins einteilen, pro Bin |Accuracy − mittlere Confidence| berechnen, nach Bin-Größe gewichtet mitteln
- Was zeigt in Reliability Diagram? == Beobachtete Accuracy gegen vorhergesagte Confidence pro Bin; perfekte Kalibrierung liegt auf der Diagonalen
- Was ist Temperature Scaling? == Post-hoc-Kalibrierung: Logits durch einen auf Validierungsdaten gelernten Scalar T teilen; ändert die Confidence, nicht die argmax-Vorhersage

## Regressionsmetriken

- Formel des Mean Absolute Error (MAE)? == (1/N) Σ |y − ŷ|
- Formel des Mean Squared Error (MSE)? == (1/N) Σ (y − ŷ)²
- Formel des Root Mean Squared Error (RMSE) und sein Vorteil gegenüber MSE? == √MSE; hat dieselbe Einheit wie die Zielgröße
- Unterschied MAE vs. RMSE bei Ausreißern? == RMSE quadriert Fehler und bestraft große Fehler überproportional; MAE ist robuster gegen Ausreißer
- Welche konstante Vorhersage minimiert MAE, welche MSE? == MAE: der Median; MSE: der Mittelwert
- Formel des Bestimmtheitsmaßes R²? == 1 − SS_res / SS_tot = 1 − Σ(y − ŷ)² / Σ(y − ȳ)²
- Kann R² negative werden? == Ja, auf Testdaten, wenn das Model schlechter ist also die konstante Vorhersage des Mittelwerts
- Wozu dient das adjustierte R²? == Es bestraft zusätzliche Features: 1 − (1 − R²)(n − 1)/(n − p − 1); steigt nur, wenn in Feature mehr erklärt also Zufall
- Formel des MAP? == (100 %/N) Σ |y − ŷ| / |y|
- Zwei Schwächen des MAP? == Undefiniert bzw. explodierend bei y nahe 0; asymmetrisch, weil Überschätzungen unbegrenzt bestraft werden, Unterschätzungen (bei positivem y) höchstens mit 100 %
- Formel des sMAP? == (100 %/N) Σ 2|y − ŷ| / (|y| + |ŷ|), symmetrische Variante des MAP mit Wertebereich 0 bis 200 %
- Was ist der MASE? == MAE des Modells geteilt durch den MAE eines naiven (saisonalen) Forecasts; < 1 heißt besser also naiv, skalenunabhängig und definiert bei Nullen
- Formel des RMSLE und wann nutzt man ihn? == √((1/N) Σ (log(1 + ŷ) − log(1 + y))²); misst relative Fehler bei Zielgrößen über mehrere Größenordnungen und bestraft Unterschätzung stärker also Überschätzung
- Was ist der Huber Loss? == Quadratisch für Fehler kleiner δ, linear darüber; kombiniert die Glattheit des MSE mit der Ausreißer-Robustheit des MAE
- Was misst der Pinball Loss (Quantile Loss)? == Die Güte einer Quantilsvorhersage: Unterschätzung wird mit τ, Überschätzung mit (1 − τ) gewichtet; für τ = 0,5 proportional zum MAE

## Ranking und Information Retrieval

- Formel von Precision@k? == Anzahl relevanter Treffer under den Top k / k
- Formel von Recall@k? == Anzahl relevanter Treffer under den Top k / Gesamtzahl relevanter Dokumente
- Was ist Hit Rate@k? == Anteil der Queries, bei denen mindestens in relevantes Dokument under den Top k ist
- Formel des Mean Reciprocal Rank (MRR)? == Mittelwert über Queries von 1 / Rang des ersten relevanten Treffers
- Was ist Mean Average Precision (MAP) im Retrieval? == Mittelwert der Average Precision über alle Queries; AP summiert Precision@k an jeder Position k mit relevantem Treffer und teilt durch die Zahl aller relevanten Dokumente
- Formel des DCG@k? == Σ_(i=1..k) rel_i / log₂(i + 1); relevante Treffer weiter oben zählen mehr
- Was ist NDCG@k? == DCG@k / IDCG@k, normiert auf das ideale Ranking; Wertebereich 0 bis 1, unterstützt abgestufte Relevanz
- Wann NDCG statt MRR? == NDCG bei abgestufter Relevanz und mehreren relevanten Treffern; MRR, wenn nur der erste relevante Treffer zählt (z. B. Question Answering, Navigation)

## NLP und generative Modelle

- Wie funktioniert BLEU? == Geometrisches Mittel der modifizierten n-Gramm-Precision (n = 1 bis 4) zwischen Kandidat und Referenzen, multipliziert mit einer Brevity Penalty
- Wozu dient die Brevity Penalty in BLEU? == Sie bestraft zu kurze Kandidaten, die sonst durch wenige sichere Wörter eine hohe Precision erreichen würden
- Wie funktioniert ROUGE und wofür wird es eingesetzt? == Recall-orientierte n-Gramm-Überlappung mit Referenzen (ROUGE-1, ROUGE-2), vor allem für Zusammenfassungen
- Was misst ROUGE-L? == Überlappung über die längste gemeinsame Teilfolge (Longest Common Subsequence) von Kandidat und Reference
- Was ist METEOR? == MT-Metrik mit Unigramm-Matching inkl. Stemming und Synonymen, recall-gewichtetem F-Mittel und Fragmentierungsstrafe für die Wortreihenfolge
- Was ist chrF? == F-Score auf Zeichen-n-Grammen; robuster also BLEU bei morphologisch reichen Sprachen wie Deutsch
- Wie funktioniert BERTScore? == Tokens von Kandidat und Reference werden über Cosine Similarity kontextueller Embeddings greedy gematcht; daraus Precision, Recall und F1
- Hauptschwäche von n-Gramm-Metriken wie BLEU und ROUGE? == Sie messen Oberflächenüberlappung; korrekte Paraphrasen werden bestraft und sinnentstellende Antworten mit vielen gleichen Wörtern belohnt
- Formel der Perplexity? == exp(−(1/N) Σ log p(x_i | x_<i)), Exponential der mittleren negativen Log-Likelihood pro Token; niedriger ist besser
- Warum sind Perplexity-Werte verschiedener Sprachmodelle oft nicht vergleichbar? == Sie hängen vom Tokenizer bzw. Vokabular ab; nur bei gleicher Tokenisierung und gleichem Testset vergleichbar
- Was ist Exact Match (EM) im Question Answering? == Anteil der Antworten, die nach Normalisierung (Kleinschreibung, Satzzeichen, Artikel) exakt der Reference entsprechen
- Was ist Token-F1 im Question Answering (SQuAD)? == F1 über die Überlappung der Token-Mengen von Antwort und Reference; gibt Teilpunkte statt alles oder nichts
- Formel der Word Error Rate (WER)? == (S + D + I) / N mit Substitutionen, Deletionen, Insertionen und N Wörtern der Reference; kann über 1 liegen
- Was misst pass@k in der Code-Generierung? == Wahrscheinlichkeit, dass mindestens eine von k generierten Lösungen alle Unit-Tests besteht
- Unverzerrter Schätzer für pass@k bei n Samples mit c korrekten? == 1 − C(n − c, k) / C(n, k)
- Typische Biases von LLM-as-a-Judge? == Position Bias (Reihenfolge), Verbosity Bias (längere Antworten bevorzugt), Self-Preference (eigene Modellfamilie bevorzugt)
- Wie reduziert man Position Bias bei paarweisen LLM-Judge-Vergleichen? == Jedes Paar in beiden Reihenfolgen bewerten und nur konsistente Urteile werten bzw. mitteln
- Was misst Faithfulness (Groundedness) bei RAG? == Anteil der Aussagen in der Antwort, die durch den abgerufenen Kontext gestützt sind; misst Halluzinationen gegenüber dem Kontext
- Was messen Context Precision und Context Recall bei RAG? == Context Precision: ob die abgerufenen Chunks relevant und relevante weit oben gerankt sind; Context Recall: ob der Kontext alle für die Referenzantwort nötigen Informationen enthält

## Computer Vision

- Formel der Intersection over Union (IoU)? == Fläche der Überlappung / Fläche der Vereinigung von vorhergesagter und wahrer Box bzw. Maske
- Wann zählt eine Detektion also True Positive? == Wenn die Klasse stimmt und die IoU mit einer noch nicht zugeordneten Ground-Truth-Box über dem Schwellwert liegt (z. B. 0,5)
- Was ist mAP bei Object Detection? == Average Precision pro Klasse, gemittelt über alle Klassen; bei COCO zusätzlich gemittelt über IoU-Schwellen 0,50 bis 0,95 in 0,05-Schritten
- Formel des Dice-Koeffizienten und Zusammenhang mit F1 und IoU? == 2|A ∩ B| / (|A| + |B|); entspricht dem F1-Score auf Pixelebene, Dice = 2 · IoU / (1 + IoU)

## Clustering

- Formel des Silhouette-Koeffizienten eines Punktes? == s = (b − a) / max(a, b), mit a = mittlere Distanz im eigenen Cluster und b = mittlere Distanz zum nächstgelegenen fremden Cluster; Wertebereich −1 bis 1
- Was ist der Adjusted Rand Index (ARI)? == Paarweise Übereinstimmung zweier Clusterings, korrigiert um Zufall; 1 perfekt, etwa 0 zufällig, braucht Ground-Truth-Labels
- Was ist Normalized Mutual Information (NMI)? == Mutual Information zwischen Clustering und Labels, normiert auf 0 bis 1; braucht Ground-Truth-Labels
- Davies-Bouldin- vs. Calinski-Harabasz-Index: Was ist jeweils besser? == Davies-Bouldin: niedriger ist besser; Calinski-Harabasz: höher ist besser; beide brauchen keine Labels
- Was ist die Elbow-Method? == Inertia (Summe der quadrierten Distanzen zu den Zentroiden) gegen die Clusterzahl k plotten und den Knick also k wählen

## Evaluationsmethodik und Statistik

- Wofür sind Training-, Validierungs- und Testset jeweils da? == Training: Parameter lernen; Validierung: Hyperparameter und Thresholds wählen; Test: einmalige, unverzerrte Schätzung der Endleistung
- Was ist Data Leakage in der Evaluation? == Information aus dem Testset gelangt ins Training oder in Modellentscheidungen (z. B. Duplikate, Normalisierung auf allen Daten, Tuning am Testset); Metriken werden zu optimistic
- Was ist k-fache Cross-Validation? == Daten in k Folds teilen, k-mal auf k − 1 Folds trainieren und auf dem übrigen Fold evaluieren, Ergebnisse mitteln
- Wann stratifizierte Cross-Validation? == Bei unbalancierten Klassen, damit jeder Fold dieselbe Klassenverteilung wie der Gesamtdatensatz hat
- Wie evaluiert man Zeitreihenmodelle korrekt? == Zeitlich geordnete Splits (Rolling bzw. Expanding Window): immer auf der Vergangenheit trainieren und auf der Zukunft testen, kein zufälliges Mischen
- Was ist Group-k-Fold und wann braucht man ihn? == Alle Beispiele einer Gruppe (z. B. Patient, Nutzer, Übersetzungen desselben Textes) landen im selben Fold; verhindert Leakage über korrelierte Beispiele
- Formel des Wald-Konfidenzintervalls für eine Proportion? == p̂ ± z · √(p̂(1 − p̂)/n)
- Warum Wilson- statt Wald-Interval für Metriken wie Accuracy, Recall oder FPR? == Wald versagt bei kleinem n oder p̂ nahe 0 bzw. 1 (Interval kollabiert auf Breite 0 oder verlässt [0, 1]); Wilson bleibt in [0, 1] und hat bessere Abdeckung
- Was besagt die Rule of Three? == Bei 0 beobachteten Ereignissen in n Versuchen liegt die obere 95-%-Grenze der Ereignisrate bei etwa 3/n
- Wie funktioniert in Bootstrap-Konfidenzintervall für eine Metrik? == Testset viele Male mit Zurücklegen resamplen, Metrik jeweils neu berechnen, Perzentile (z. B. 2,5 % und 97,5 %) also Interval nehmen
- Welcher Test vergleicht zwei Klassifikatoren auf demselben Testset? == McNemar-Test; nutzt nur die Beispiele, bei denen genau eines der beiden Modelle richtig liegt
- Was besagt Goodhart's Law im Kontext von Metriken? == "Wenn in Maß zum Ziel wird, hört es auf, in gutes Maß zu sein"; Optimierung auf eine Proxy-Metrik entkoppelt sie vom eigentlichen Ziel
- Warum sollte man bei Benchmarks mehrere Seeds bzw. Läufe berichten? == Varianz durch Initialisierung, Sampling und Datenreihenfolge kann größer sein also der gemessene Unterschied; Mittelwert und Streuung zeigen, ob in Gewinn echt ist
