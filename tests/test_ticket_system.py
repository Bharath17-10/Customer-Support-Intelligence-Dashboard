import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.dashboard import build_dashboard_data
from src.database import (
    CUSTOMER_RECORD,
    DEMO_RECORD,
    add_ticket_reply,
    create_ticket,
    delete_ticket,
    get_all_tickets,
    get_dashboard_statistics,
    get_department_statistics,
    get_priority_statistics,
    get_ticket_by_id,
    get_ticket_replies,
    initialize_database,
    update_ticket_status,
)
from src.preprocess import load_dataset_with_report
from src.ticket_manager import (
    category_for_department,
    classify_complaint,
    import_batch_predictions,
    import_dataset_tickets,
    validate_customer_submission,
)


class TicketSystemTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "support_tickets.db"
        initialize_database(self.database_path)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_only_known_model_labels_map_to_display_categories(self):
        self.assertEqual(category_for_department("Billing"), "Billing & Payments")
        self.assertEqual(category_for_department("Account"), "Account & Security")
        self.assertEqual(
            category_for_department("a new label"),
            "Other / Unclassified",
        )

    def test_complaint_is_preprocessed_before_existing_models_predict(self):
        class RecordingModel:
            def __init__(self, prediction):
                self.prediction = prediction
                self.seen = None

            def predict(self, values):
                self.seen = values[0]
                return [self.prediction]

        department = RecordingModel("Billing")
        urgency = RecordingModel("High")
        result = classify_complaint(
            "My PAYMENT was deducted twice!", department, urgency
        )
        self.assertEqual(result, ("Billing", "High"))
        self.assertEqual(department.seen, "my payment was deducted twice!")
        self.assertEqual(urgency.seen, department.seen)

    def test_customer_input_requires_name_valid_email_and_readable_text(self):
        validate_customer_submission(
            "Customer", "customer@example.com", "Package is delayed."
        )
        with self.assertRaisesRegex(ValueError, "valid customer email"):
            validate_customer_submission("Customer", "invalid", "Package is delayed.")
        with self.assertRaisesRegex(ValueError, "readable text"):
            validate_customer_submission("Customer", "customer@example.com", "!!!")
        with self.assertRaisesRegex(ValueError, "readable text"):
            classify_complaint("!!!", object(), object())

    def test_customer_ticket_defaults_and_status_changes_persist(self):
        ticket_id = create_ticket(
            customer_name="Customer",
            email="customer@example.com",
            ticket_text="Payment was deducted twice.",
            department="Billing",
            urgency="High",
            database_path=self.database_path,
        )
        ticket = get_ticket_by_id(ticket_id, self.database_path)
        self.assertEqual(ticket["status"], "Pending")
        self.assertEqual(ticket["record_type"], CUSTOMER_RECORD)

        update_ticket_status(ticket_id, "In Progress", self.database_path)
        ticket = get_ticket_by_id(ticket_id, self.database_path)
        self.assertEqual(ticket["status"], "In Progress")
        self.assertEqual(len(get_all_tickets(self.database_path)), 1)

    def test_ticket_response_notes_persist_and_are_deleted_with_ticket(self):
        ticket_id = create_ticket(
            customer_name="Customer",
            email="customer@example.com",
            ticket_text="Package is delayed.",
            department="Shipping",
            urgency="High",
            database_path=self.database_path,
        )

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            add_ticket_reply(ticket_id, "  ", self.database_path)
        add_ticket_reply(ticket_id, "We are checking the carrier status.", self.database_path)
        replies = get_ticket_replies(ticket_id, self.database_path)
        self.assertEqual(len(replies), 1)
        self.assertEqual(
            replies[0]["reply_text"], "We are checking the carrier status."
        )

        delete_ticket(ticket_id, self.database_path)
        self.assertEqual(get_ticket_replies(ticket_id, self.database_path), [])

    def test_dashboard_statistics_exclude_demo_examples(self):
        create_ticket(
            customer_name="Customer",
            email="customer@example.com",
            ticket_text="A real submitted complaint.",
            department="Billing",
            urgency="Critical",
            database_path=self.database_path,
        )
        import_dataset_tickets(
            pd.DataFrame(
                [
                    {
                        "ticket_text": "Historical dataset complaint",
                        "department": "Shipping",
                        "urgency": "High",
                    }
                ]
            ),
            self.database_path,
        )

        statistics = get_dashboard_statistics(self.database_path)
        self.assertEqual(statistics["total"], 1)
        self.assertEqual(statistics["pending"], 1)
        self.assertEqual(statistics["critical"], 1)
        self.assertEqual(statistics["demo_imported"], 1)
        self.assertEqual(
            get_department_statistics(database_path=self.database_path),
            [{"label": "Billing", "ticket_count": 1}],
        )
        self.assertEqual(
            get_priority_statistics(database_path=self.database_path),
            [{"label": "Critical", "ticket_count": 1}],
        )
        records = get_all_tickets(self.database_path)
        self.assertEqual(set(records["record_type"]), {CUSTOMER_RECORD, DEMO_RECORD})
        dashboard = build_dashboard_data(
            records,
            get_department_statistics(database_path=self.database_path),
            get_priority_statistics(database_path=self.database_path),
        )
        self.assertEqual(len(dashboard["customer_tickets"]), 1)
        self.assertEqual(dashboard["department_counts"]["Billing & Payments"], 1)

    def test_dataset_import_is_explicitly_idempotent(self):
        frame = pd.DataFrame(
            [
                {
                    "ticket_text": "Refund is delayed.",
                    "department": "Billing",
                    "urgency": "Medium",
                }
            ]
        )
        first = import_dataset_tickets(frame, self.database_path)
        second = import_dataset_tickets(frame, self.database_path)
        self.assertEqual(first, {"imported": 1, "skipped": 0})
        self.assertEqual(second, {"imported": 0, "skipped": 1})
        self.assertEqual(get_dashboard_statistics(self.database_path)["total"], 0)

    def test_batch_predictions_can_be_saved_as_idempotent_demo_tickets(self):
        frame = pd.DataFrame(
            [
                {
                    "ticket_text": "My package is delayed.",
                    "predicted_department": "Shipping",
                    "predicted_urgency": "High",
                },
                {
                    "ticket_text": "I cannot access my account.",
                    "predicted_department": "Account",
                    "predicted_urgency": "Medium",
                },
            ]
        )

        first = import_batch_predictions(frame, b"uploaded csv", self.database_path)
        second = import_batch_predictions(frame, b"uploaded csv", self.database_path)

        self.assertEqual(first, {"imported": 2, "skipped": 0})
        self.assertEqual(second, {"imported": 0, "skipped": 2})
        tickets = get_all_tickets(self.database_path)
        self.assertEqual(set(tickets["record_type"]), {DEMO_RECORD})
        self.assertEqual(
            set(zip(tickets["department"], tickets["urgency"])),
            {("Shipping", "High"), ("Account", "Medium")},
        )
        statistics = get_dashboard_statistics(self.database_path)
        self.assertEqual(statistics["total"], 0)
        self.assertEqual(statistics["demo_imported"], 2)

    def test_selected_batch_query_can_be_saved_without_other_rows(self):
        frame = pd.DataFrame(
            [
                {
                    "ticket_text": "First complaint.",
                    "predicted_department": "Billing",
                    "predicted_urgency": "Low",
                },
                {
                    "ticket_text": "Selected complaint.",
                    "predicted_department": "Shipping",
                    "predicted_urgency": "High",
                },
            ]
        )

        outcome = import_batch_predictions(
            frame.loc[[1]], b"two complaint upload", self.database_path
        )

        self.assertEqual(outcome, {"imported": 1, "skipped": 0})
        tickets = get_all_tickets(self.database_path)
        self.assertEqual(len(tickets), 1)
        self.assertEqual(tickets.iloc[0]["ticket_text"], "Selected complaint.")
        self.assertEqual(tickets.iloc[0]["record_type"], DEMO_RECORD)

    def test_dataset_requires_expected_columns_and_multiple_labels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "tickets.csv"
            pd.DataFrame({"ticket_text": ["A complaint"]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_dataset_with_report(path)

            pd.DataFrame(
                [
                    ["First complaint", "Billing", "High"],
                    ["Second complaint", "Billing", "Low"],
                ],
                columns=["ticket_text", "department", "urgency"],
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "at least two distinct labels"):
                load_dataset_with_report(path)

    def test_dataset_validation_drops_missing_blanks_and_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "tickets.csv"
            pd.DataFrame(
                [
                    [" Payment is delayed! ", "Billing", "High"],
                    [" Payment is delayed! ", "Billing", "High"],
                    ["Application crashes", "Technical", "Low"],
                    [None, "Technical", "Low"],
                ],
                columns=["ticket_text", "department", "urgency"],
            ).to_csv(path, index=False)
            frame, report = load_dataset_with_report(path)

        self.assertEqual(len(frame), 2)
        self.assertEqual(report["input_records"], 4)
        self.assertEqual(report["missing_value_records"], 1)
        self.assertEqual(report["duplicate_records"], 1)
        self.assertEqual(frame.iloc[0]["ticket_text"], "payment is delayed!")

    def test_delete_ticket_removes_record_and_invalid_status_is_rejected(self):
        ticket_id = create_ticket(
            customer_name="Customer",
            email="customer@example.com",
            ticket_text="A complaint.",
            department="Billing",
            urgency="Low",
            database_path=self.database_path,
        )
        with self.assertRaises(ValueError):
            update_ticket_status(ticket_id, "Auto-resolved", self.database_path)
        delete_ticket(ticket_id, self.database_path)
        self.assertIsNone(get_ticket_by_id(ticket_id, self.database_path))


if __name__ == "__main__":
    unittest.main()
