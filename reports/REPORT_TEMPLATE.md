# Customer Support Ticket Auto-Router & Prioritizer

## 1. Problem statement
Explain why support tickets need automatic department routing and urgency prioritization.

## 2. Dataset
- Dataset name/source:
- License/usage notes:
- Number of records:
- Department labels and counts:
- Urgency labels and counts:
- Missing/duplicate/noisy data handling:

## 3. NLP pipeline architecture
Describe: input ticket → text cleaning → TF-IDF / sentence embeddings → department classifier + urgency classifier → dashboard output.

## 4. Experimental setup
- Train/validation/test split:
- Random seed:
- Baseline: TF-IDF + Logistic Regression
- Improved model: sentence-transformer embeddings + Logistic Regression
- Hardware/software:

## 5. Results
Report accuracy, macro-F1, per-class precision/recall/F1, and confusion matrices for both tasks and both approaches. Do not invent or estimate scores.

## 6. Error analysis
Include at least 3–5 actual test-set mistakes. For each: ticket text (anonymized), true label, predicted label, likely reason, and possible improvement.

## 7. Limitations and future scope
Discuss class imbalance, ambiguous tickets, domain-specific language, confidence thresholds, human review, and possible fine-tuning of a transformer.
