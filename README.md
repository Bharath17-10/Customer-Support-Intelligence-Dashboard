# Customer Support Intelligence Dashboard

Streamlit customer-support ticket management backed by the project's trained TF-IDF and Logistic Regression classifiers.

## What it does

- Validates and preprocesses the labelled `data/tickets.csv` training dataset.
- Reports missing, blank, and duplicate rows plus department and urgency distributions.
- Removes exact duplicate training examples before creating reproducible train/validation/test splits.
- Trains separate department and urgency models and saves them to the existing `models/` paths.
- Classifies customer submissions with those models, then saves them in SQLite as Pending.
- Persists status updates and displays database-backed ticket metrics and analytics.
- Saves internal response notes on customer tickets; notes are not emailed to customers.
- Keeps historical dataset examples out of the live ticket queue unless an admin explicitly imports them as `Demo / Dataset Imported`.
- Keeps batch CSV predictions as a preview by default; users can inspect a selected query and explicitly save one or all readable results as demo records for access in Ticket management without adding them to live customer metrics.

## Project structure

```text
NLP36_Customer_Support_Ticket_AutoRouter/
├── app.py
├── requirements.txt
├── data/
│   ├── tickets.csv
│   └── support_tickets.db       # created automatically
├── models/
│   ├── department_tfidf.joblib
│   └── urgency_tfidf.joblib
├── reports/
│   ├── dataset_analysis.json
│   ├── tfidf_evaluation.json
│   └── error_examples.json
├── src/
│   ├── dashboard.py
│   ├── database.py
│   ├── preprocess.py
│   ├── ticket_manager.py
│   ├── train.py
│   └── train_embeddings.py
└── tests/
    └── test_ticket_system.py
```

## Setup and train (Windows PowerShell)

Run from the project directory:

```powershell
if (-not (Test-Path .\.venv\Scripts\python.exe)) { python -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = (Resolve-Path .\src).Path
.\.venv\Scripts\python.exe .\src\train.py --data .\data\tickets.csv --seed 42
```

The trainer uses the columns `ticket_text`, `department`, and `urgency`. It writes dataset quality and class-distribution details to `reports\dataset_analysis.json`, and validation/test metrics to `reports\tfidf_evaluation.json`. Training examples are never inserted into the customer-ticket database by training.

Scikit-learn is pinned to 1.8.0 to match the serialized TF-IDF models.

## Run and test

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\streamlit.exe run .\app.py --server.address 127.0.0.1
```

Customer tickets and status updates are stored in `data\support_tickets.db`. Demo imports are clearly marked and excluded from live customer metrics. The database starts without sample customer tickets. On **Batch processing**, choose a row in **Select a query to inspect** to see its complete text and predictions. Use **Save selected query as demo** for one row or **Save classified tickets as demos** for all readable rows; find saved entries in **Ticket management** by selecting **All including demo records**. Re-importing the same CSV skips rows already saved.

Open a customer ticket in **Ticket management** to view its details, update its status, and add response notes. Response notes are stored in the local database and are not sent as email.

## Optional admin dataset import

Dataset import is disabled unless an admin password is set in the current PowerShell session. Set a private value locally; do not commit it:

```powershell
$securePassword = Read-Host "Set the support admin password" -AsSecureString
$env:SUPPORT_ADMIN_PASSWORD = [System.Net.NetworkCredential]::new("", $securePassword).Password
.\.venv\Scripts\streamlit.exe run .\app.py --server.address 127.0.0.1
```

Open **Admin · Dataset import**, unlock the page, select a labelled CSV, confirm the demo-record notice, and explicitly submit the import. Duplicate rows are skipped idempotently. Imported records are tagged `Demo / Dataset Imported`; they do not train models or count as live customer tickets. The password gate is intended for local development; production access should be protected behind authenticated HTTPS.

## Optional sentence-transformer experiment

```powershell
.\.venv\Scripts\python.exe -m pip install sentence-transformers
$env:PYTHONPATH = (Resolve-Path .\src).Path
.\.venv\Scripts\python.exe .\src\train_embeddings.py --data .\data\tickets.csv --seed 42
```

The encoder downloads on first use and may require additional RAM and disk space. The Streamlit app continues to use the TF-IDF models.

Predictions are decision support. Support employees control ticket status and resolution; model predictions do not resolve tickets.
