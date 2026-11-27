import hashlib
import re

from src.database import import_demo_tickets
from src.preprocess import clean_text, has_readable_text

CATEGORIES = (
    "Billing & Payments",
    "Technical Issues",
    "Account & Security",
    "Shipping & Delivery",
)

CATEGORY_ISSUES = {
    "Billing & Payments": (
        "Payment deducted twice",
        "Refund not received",
        "Incorrect invoice",
        "Unauthorized transaction",
    ),
    "Technical Issues": (
        "Application crashing",
        "Website not loading",
        "Technical login errors",
        "Server errors",
    ),
    "Account & Security": (
        "Account locked",
        "Password reset problems",
        "Account hacked",
        "Unauthorized account access",
    ),
    "Shipping & Delivery": (
        "Order not delivered",
        "Package delayed",
        "Wrong delivery address",
        "Missing package",
    ),
}

PRIORITIES = ("Critical", "High", "Medium", "Low")
STATUSES = ("Pending", "In Progress", "Resolved")

_DEPARTMENT_LABELS = {
    "account": "Account & Security",
    "account security": "Account & Security",
    "billing": "Billing & Payments",
    "billing payments": "Billing & Payments",
    "shipping": "Shipping & Delivery",
    "shipping delivery": "Shipping & Delivery",
    "technical": "Technical Issues",
    "technical issues": "Technical Issues",
}

_SUPPORT_RECOMMENDATIONS = {
    "Billing & Payments": (
        "Billing Support",
        "Review the transaction and order records, then follow the billing team's refund or correction process.",
    ),
    "Technical Issues": (
        "Technical Support",
        "Reproduce the reported issue and review relevant application or server diagnostics.",
    ),
    "Account & Security": (
        "Account Security Support",
        "Verify the customer's identity and follow the account security review process.",
    ),
    "Shipping & Delivery": (
        "Shipping Support",
        "Check the carrier tracking details and confirm the delivery address with the customer.",
    ),
}

_EMAIL_PATTERN = re.compile(r"^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$", re.IGNORECASE)


def _normalize_label(label):
    return re.sub(r"[^a-z0-9]+", " ", str(label).strip().lower()).strip()


def category_for_department(label):
    return _DEPARTMENT_LABELS.get(_normalize_label(label), "Other / Unclassified")


def display_priority(label):
    normalized = _normalize_label(label)
    known_priorities = {priority.lower(): priority for priority in PRIORITIES}
    return known_priorities.get(normalized, str(label).strip() or "Other / Unclassified")


def classify_complaint(complaint, department_model, urgency_model):
    cleaned = clean_text(complaint)
    if not has_readable_text(complaint):
        raise ValueError("Enter a customer complaint containing readable text.")
    department = department_model.predict([cleaned])[0]
    urgency = urgency_model.predict([cleaned])[0]
    return str(department), str(urgency)


def validate_customer_submission(customer_name, email, complaint):
    if not str(customer_name or "").strip():
        raise ValueError("Enter the customer's name.")
    if not _EMAIL_PATTERN.fullmatch(str(email or "").strip()):
        raise ValueError("Enter a valid customer email address.")
    if not has_readable_text(complaint):
        raise ValueError("Enter a complaint containing readable text.")


def import_dataset_tickets(frame, database_path=None):
    records = []
    for row in frame.itertuples(index=False):
        ticket_text = str(row.ticket_text).strip()
        department = str(row.department).strip()
        urgency = str(row.urgency).strip()
        identity = "\x1f".join(
            (clean_text(ticket_text), department, urgency)
        )
        source_key = "dataset:" + hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()
        records.append(
            {
                "ticket_text": ticket_text,
                "department": department,
                "urgency": urgency,
                "source_key": source_key,
            }
        )

    if database_path is None:
        return import_demo_tickets(records)
    return import_demo_tickets(records, database_path)


def recommended_support(display_department):
    return _SUPPORT_RECOMMENDATIONS.get(
        display_department,
        (
            "General Support Queue",
            "Review the complaint and route it to the appropriate team.",
        ),
    )
