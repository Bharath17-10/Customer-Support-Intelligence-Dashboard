import re

import pandas as pd

REQUIRED_COLUMNS = ["ticket_text", "department", "urgency"]


def clean_text(text):
    """Light normalization; preserves words that may signal urgency."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"[^a-z0-9\s!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_readable_text(text):
    return bool(re.search(r"[a-z0-9]", clean_text(text)))


def load_dataset_with_report(path):
    try:
        source = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"Could not read dataset CSV: {error}") from error

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in source]
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    frame = source[REQUIRED_COLUMNS].copy()
    report = {
        "source": str(getattr(path, "name", path)),
        "input_records": int(len(frame)),
        "missing_value_records": int(frame.isna().any(axis=1).sum()),
    }
    frame = frame.dropna(subset=REQUIRED_COLUMNS).copy()
    frame["ticket_text"] = frame["ticket_text"].map(clean_text)
    for column in ("department", "urgency"):
        frame[column] = frame[column].astype(str).str.strip()

    invalid_rows = (
        frame["ticket_text"].eq("")
        | frame["department"].eq("")
        | frame["urgency"].eq("")
    )
    report["empty_value_records"] = int(invalid_rows.sum())
    frame = frame.loc[~invalid_rows].copy()

    report["duplicate_records"] = int(frame.duplicated(REQUIRED_COLUMNS).sum())
    frame = frame.drop_duplicates(REQUIRED_COLUMNS, keep="first").reset_index(drop=True)
    report["usable_records"] = int(len(frame))
    report["department_distribution"] = {
        str(label): int(count)
        for label, count in frame["department"].value_counts().sort_index().items()
    }
    report["urgency_distribution"] = {
        str(label): int(count)
        for label, count in frame["urgency"].value_counts().sort_index().items()
    }
    validate_labels(frame)
    return frame, report


def validate_labels(frame):
    for column in ("department", "urgency"):
        if column not in frame:
            raise ValueError(f"Dataset is missing the '{column}' label column.")
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Dataset contains an empty '{column}' label.")
        if frame[column].nunique() < 2:
            raise ValueError(
                f"The '{column}' column must contain at least two distinct labels."
            )


def load_dataset(path):
    frame, _ = load_dataset_with_report(path)
    return frame
