from src.database import CUSTOMER_RECORD
from src.ticket_manager import category_for_department, display_priority


def build_dashboard_data(tickets, department_statistics, priority_statistics):
    customer_tickets = tickets.loc[
        tickets["record_type"].eq(CUSTOMER_RECORD)
    ].copy()
    departments = {}
    for row in department_statistics:
        category = category_for_department(row["label"])
        departments[category] = (
            departments.get(category, 0) + int(row["ticket_count"])
        )

    priorities = {}
    for row in priority_statistics:
        priority = display_priority(row["label"])
        priorities[priority] = (
            priorities.get(priority, 0) + int(row["ticket_count"])
        )

    return {
        "customer_tickets": customer_tickets,
        "department_counts": departments,
        "priority_counts": priorities,
    }
