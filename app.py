import html
import hmac
import os
from pathlib import Path
import pickle
import sqlite3
import uuid

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.database import (
    CUSTOMER_RECORD,
    DEFAULT_DATABASE_PATH,
    DEMO_RECORD,
    STATUSES,
    add_ticket_reply,
    create_ticket,
    get_all_tickets,
    get_dashboard_statistics,
    get_department_statistics,
    get_priority_statistics,
    get_ticket_by_id,
    get_ticket_replies,
    initialize_database,
    update_ticket_status,
)
from src.dashboard import build_dashboard_data
from src.preprocess import clean_text, has_readable_text, load_dataset_with_report
from src.ticket_manager import (
    CATEGORIES,
    CATEGORY_ISSUES,
    PRIORITIES,
    category_for_department,
    classify_complaint,
    display_priority,
    import_dataset_tickets,
    import_batch_predictions,
    recommended_support,
    validate_customer_submission,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATABASE_PATH = DEFAULT_DATABASE_PATH
SAMPLE_COMPLAINTS = [
    ("Billing & Payments", "My payment was deducted but my order is not confirmed."),
    (
        "Technical Issues",
        "The application crashes whenever I open my order history.",
    ),
    ("Account & Security", "I cannot log in because my account has been locked."),
    (
        "Shipping & Delivery",
        "My package has not arrived even though the expected delivery date has passed.",
    ),
]

st.set_page_config(
    page_title="Customer Support Intelligence Dashboard",
    page_icon="🎫",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #10233f;
        --blue: #4779c7;
        --ink: #17263c;
        --muted: #718096;
        --surface: #ffffff;
        --canvas: #f3f6fb;
        --line: #e5eaf2;
    }
    [data-testid="stAppViewContainer"] { background: var(--canvas); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: var(--navy); }
    [data-testid="stSidebar"] * { color: #eef4ff; }
    [data-testid="stSidebar"] [data-baseweb="select"] * { color: #17263c; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }
    h1, h2, h3 { color: var(--navy); letter-spacing: -0.025em; }
    [data-testid="stMain"] [data-testid="stWidgetLabel"],
    [data-testid="stMain"] [data-testid="stWidgetLabel"] p {
        color: var(--ink) !important;
    }
    [data-testid="stMain"] [data-baseweb="input"] > div,
    [data-testid="stMain"] [data-baseweb="textarea"] > div,
    [data-testid="stMain"] [data-baseweb="select"] > div {
        background-color: var(--surface) !important;
        border-color: #cbd5e1 !important;
    }
    [data-testid="stMain"] [data-baseweb="input"] input,
    [data-testid="stMain"] [data-baseweb="textarea"] textarea {
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink);
        caret-color: var(--navy);
    }
    [data-testid="stMain"] [data-testid="stForm"] {
        background: var(--surface);
        border-color: #cbd5e1;
        border-radius: 12px;
    }
    [data-testid="stMetric"] {
        background: var(--surface); border: 1px solid var(--line);
        border-radius: 14px; padding: 18px 20px; box-shadow: 0 3px 14px #10233f0a;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); font-weight: 600; }
    [data-testid="stMetricValue"] { color: var(--navy); }
    [data-testid="stMain"] [data-testid="stButton"] > button {
        color: var(--navy) !important;
        background: var(--surface) !important;
        border-color: #cbd5e1 !important;
        border-radius: 9px;
    }
    [data-testid="stMain"] [data-testid="stButton"] > button:hover {
        background: #eef4ff !important;
        border-color: var(--blue) !important;
    }
    [data-testid="stMain"] [data-testid="stButton"] > button[kind="primary"] {
        background: var(--navy) !important; border-color: var(--navy) !important;
        border-radius: 9px; color: #ffffff !important;
        -webkit-text-fill-color: #ffffff;
    }
    [data-testid="stMain"] [data-testid="stFormSubmitButton"] > button,
    [data-testid="stMain"] [data-testid="stDownloadButton"] > button {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff;
        background: var(--blue) !important;
        border: 1px solid var(--blue) !important;
        border-radius: 9px;
    }
    .category-card, .sample-card {
        background: var(--surface); border: 1px solid var(--line);
        border-radius: 14px; padding: 20px; min-height: 170px;
        box-shadow: 0 3px 14px #10233f0a; margin-bottom: 12px;
    }
    .category-card h3 { margin: 0 0 8px; font-size: 1.05rem; }
    .category-count { font-size: 1.8rem; font-weight: 700; color: var(--navy); }
    .category-issues { color: var(--muted); font-size: .88rem; line-height: 1.55; }
    .sample-label {
        display: inline-block; background: #eaf1fd; color: #254e8a;
        padding: 4px 9px; border-radius: 999px; font-size: .78rem;
        font-weight: 600; margin-bottom: 10px;
    }
    .page-subtitle { color: var(--muted); margin-top: -10px; margin-bottom: 24px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    model_paths = {
        "department": MODEL_DIR / "department_tfidf.joblib",
        "urgency": MODEL_DIR / "urgency_tfidf.joblib",
    }
    missing = [str(path) for path in model_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Trained TF-IDF model files are missing: " + ", ".join(missing)
        )
    return (
        joblib.load(model_paths["department"]),
        joblib.load(model_paths["urgency"]),
    )


def _priority_style(priority):
    styles = {
        "Critical": ("#fde8e8", "#a61b1b"),
        "High": ("#fff0df", "#a94e00"),
        "Medium": ("#fff8d8", "#806300"),
        "Low": ("#e5f6ed", "#176b42"),
    }
    background, foreground = styles.get(priority, ("#edf0f5", "#46546a"))
    return (
        f"background-color: {background}; color: {foreground}; "
        "font-weight: 600;"
    )


def _style_chart(chart):
    chart.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#dce6f5",
        margin=dict(l=12, r=12, t=30, b=12),
        legend_title_text="",
    )
    return chart


def _render_category_cards(department_counts):
    st.subheader("Complaint categories")
    st.caption(
        "Counts use department predictions for customer-submitted tickets. "
        "Imported dataset examples are reported separately."
    )
    columns = st.columns(4)
    for column, category in zip(columns, CATEGORIES):
        issues = " · ".join(CATEGORY_ISSUES[category])
        with column:
            st.markdown(
                f"""
                <div class="category-card">
                    <h3>{html.escape(category)}</h3>
                    <div class="category-count">{department_counts.get(category, 0):,}</div>
                    <div class="category-issues">{html.escape(issues)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    unclassified = department_counts.get("Other / Unclassified", 0)
    if unclassified:
        st.caption(f"Other / Unclassified: {unclassified:,} customer tickets")


def _render_analytics(customer_tickets, dashboard_data):
    st.subheader("Ticket analytics")
    if customer_tickets.empty:
        st.info("Analytics will appear after the first customer ticket is submitted.")
        return

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("#### Tickets by department")
        department_counts = dashboard_data["department_counts"]
        department_frame = pd.DataFrame(
            list(department_counts.items()), columns=["Department", "Tickets"]
        )
        chart = px.pie(
            department_frame,
            names="Department",
            values="Tickets",
            hole=0.58,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(_style_chart(chart), width="stretch")

    with chart_right:
        st.markdown("#### Tickets by priority")
        priority_counts = dashboard_data["priority_counts"]
        priority_order = [value for value in PRIORITIES if value in priority_counts]
        priority_order.extend(
            sorted(value for value in priority_counts if value not in priority_order)
        )
        chart = px.bar(
            x=priority_order,
            y=[priority_counts[value] for value in priority_order],
            labels={"x": "Priority", "y": "Tickets"},
            color=priority_order,
            color_discrete_map={
                "Critical": "#d64545",
                "High": "#ef8b32",
                "Medium": "#e5bd45",
                "Low": "#46a879",
            },
        )
        chart.update_layout(showlegend=False)
        st.plotly_chart(_style_chart(chart), width="stretch")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("#### Tickets by status")
        status_counts = customer_tickets["status"].value_counts()
        status_order = [status for status in STATUSES if status in status_counts]
        chart = px.bar(
            x=status_order,
            y=[int(status_counts[status]) for status in status_order],
            labels={"x": "Status", "y": "Tickets"},
            color=status_order,
            color_discrete_map={
                "Pending": "#4779c7",
                "In Progress": "#e5a545",
                "Resolved": "#46a879",
            },
        )
        chart.update_layout(showlegend=False)
        st.plotly_chart(_style_chart(chart), width="stretch")

    with chart_right:
        st.markdown("#### Tickets received over time")
        received = pd.to_datetime(
            customer_tickets["created_at"], errors="coerce"
        ).dropna()
        if received.empty:
            st.info("No valid ticket dates are available to plot.")
        else:
            daily = (
                received.dt.date.value_counts().sort_index().rename_axis("Date")
                .reset_index(name="Tickets")
            )
            daily["Date"] = daily["Date"].astype(str)
            chart = px.line(
                daily,
                x="Date",
                y="Tickets",
                markers=True,
                labels={"Date": "Date received", "Tickets": "Tickets"},
            )
            chart.update_xaxes(type="category")
            chart.update_traces(line_color="#6ca0f6")
            st.plotly_chart(_style_chart(chart), width="stretch")


def _render_recent_tickets(customer_tickets):
    st.subheader("Recent customer tickets")
    if customer_tickets.empty:
        st.info("No customer-submitted tickets have been received yet.")
        return
    recent = customer_tickets.head(5).copy()
    recent["department"] = recent["department"].map(category_for_department)
    recent["urgency"] = recent["urgency"].map(display_priority)
    recent["created_at"] = pd.to_datetime(
        recent["created_at"], errors="coerce"
    ).dt.strftime("%Y-%m-%d %H:%M")
    recent = recent.rename(
        columns={
            "ticket_id": "Ticket ID",
            "ticket_text": "Customer complaint",
            "department": "Predicted department",
            "urgency": "Priority",
            "status": "Status",
            "created_at": "Date received",
        }
    )
    st.dataframe(
        recent[
            [
                "Ticket ID",
                "Customer complaint",
                "Predicted department",
                "Priority",
                "Status",
                "Date received",
            ]
        ],
        width="stretch",
        hide_index=True,
    )


def _render_overview(statistics, dashboard_data):
    st.title("Customer Support Intelligence Dashboard")
    st.markdown(
        '<div class="page-subtitle">AI-powered complaint classification, prioritization and ticket management.</div>',
        unsafe_allow_html=True,
    )
    metric_sets = [
        [
            ("Total Customer Tickets", statistics["total"], "🎫"),
            ("Pending Tickets", statistics["pending"], "◷"),
            ("In Progress", statistics["in_progress"], "↻"),
            ("Resolved Tickets", statistics["resolved"], "✓"),
        ],
        [
            ("Critical Priority", statistics["critical"], "🔴"),
            ("High Priority", statistics["high"], "🟠"),
            ("Demo Dataset Records", statistics["demo_imported"], "🧪"),
        ],
    ]
    for metrics in metric_sets:
        columns = st.columns(len(metrics))
        for column, (label, value, icon) in zip(columns, metrics):
            with column:
                st.metric(f"{icon}  {label}", f"{value:,}")

    st.write("")
    _render_category_cards(dashboard_data["department_counts"])
    customer_tickets = dashboard_data["customer_tickets"]
    _render_recent_tickets(customer_tickets)
    _render_analytics(customer_tickets, dashboard_data)


def _render_ticket_details(ticket_id, key_suffix):
    try:
        ticket = get_ticket_by_id(ticket_id, DATABASE_PATH)
    except sqlite3.Error as error:
        st.error(f"Could not load ticket details: {error}")
        return
    if ticket is None:
        st.warning("That ticket is no longer available.")
        return

    display_department = category_for_department(ticket["department"])
    team, next_action = recommended_support(display_department)
    st.markdown("---")
    st.subheader(f"Ticket details · {ticket['ticket_id']}")
    if ticket["record_type"] == DEMO_RECORD:
        st.warning(
            "Demo / Dataset Imported — this is a historical example, not a new customer complaint."
        )
    st.info(ticket["ticket_text"])
    identity_left, identity_right = st.columns(2)
    identity_left.markdown(
        f"**Customer**  \n{ticket['customer_name']}  \n**Email**  \n"
        f"{ticket['email'] or 'Not provided'}"
    )
    identity_right.markdown(
        f"**Date received**  \n{ticket['created_at']}  \n"
        f"**Record type**  \n{ticket['record_type']}"
    )
    first, second, third = st.columns(3)
    first.metric("Predicted department", display_department)
    second.metric("Predicted urgency", display_priority(ticket["urgency"]))
    third.metric("Ticket status", ticket["status"])
    st.caption(f"Original model department label: {ticket['department']}")
    if ticket["customer_category"]:
        st.caption(f"Customer-selected category: {ticket['customer_category']}")
    details_left, details_right = st.columns(2)
    details_left.markdown(f"**Recommended support team**  \n{team}")
    details_right.markdown(f"**Suggested next action**  \n{next_action}")

    if ticket["record_type"] == DEMO_RECORD:
        st.caption("Demo records are excluded from live customer ticket counts.")
        return

    with st.form(f"status_form_{key_suffix}_{ticket_id}"):
        new_status = st.selectbox(
            "Update ticket status",
            STATUSES,
            index=STATUSES.index(ticket["status"]),
            key=f"status_{key_suffix}_{ticket_id}",
        )
        submitted = st.form_submit_button("Save status")
    if submitted:
        try:
            update_ticket_status(ticket_id, new_status, DATABASE_PATH)
        except (ValueError, sqlite3.Error) as error:
            st.error(f"Could not update ticket status: {error}")
        else:
            st.success("Ticket status updated.")
            st.rerun()

    st.subheader("Response notes")
    st.caption("Notes are saved in this app only; they are not emailed to the customer.")
    try:
        replies = get_ticket_replies(ticket_id, DATABASE_PATH)
    except sqlite3.Error as error:
        st.error(f"Could not load response notes: {error}")
        return
    if replies:
        for reply in replies:
            st.markdown(f"**{reply['author']} · {reply['created_at']}**")
            st.text(reply["reply_text"])
    else:
        st.info("No response notes have been added to this ticket.")

    with st.form(f"response_note_form_{ticket_id}"):
        response_text = st.text_area(
            "Write a response note",
            key=f"response_note_{ticket_id}",
            max_chars=5000,
            placeholder="Record the response or next step for this ticket...",
        )
        reply_submitted = st.form_submit_button("Save response note", type="primary")
    if reply_submitted:
        try:
            add_ticket_reply(ticket_id, response_text, DATABASE_PATH)
        except (ValueError, sqlite3.Error) as error:
            st.error(f"Could not save response note: {error}")
        else:
            st.success("Response note saved. It was not emailed to the customer.")
            st.rerun()


def _render_ticket_management(tickets):
    st.title("Ticket management")
    st.caption("Search, filter, inspect, and update customer ticket status.")
    if tickets.empty:
        st.info("There are no ticket records yet. Submit a complaint to get started.")
        return

    filters = st.columns(5)
    with filters[0]:
        record_view = st.selectbox(
            "Records",
            ["Customer tickets", "All including demo records"],
        )
    working = tickets.copy()
    if record_view == "Customer tickets":
        working = working[working["record_type"] == CUSTOMER_RECORD]
    working["display_department"] = working["department"].map(
        category_for_department
    )
    working["display_priority"] = working["urgency"].map(display_priority)
    with filters[1]:
        departments = st.multiselect(
            "Department",
            sorted(working["display_department"].unique().tolist()),
            key="department_filter",
        )
    with filters[2]:
        priorities = st.multiselect(
            "Priority",
            sorted(working["display_priority"].unique().tolist()),
            key="priority_filter",
        )
    with filters[3]:
        statuses = st.multiselect(
            "Ticket status",
            sorted(working["status"].unique().tolist()),
            key="status_filter",
        )
    with filters[4]:
        search = st.text_input("Search complaint or Ticket ID", key="ticket_search")

    filtered = working
    if departments:
        filtered = filtered[filtered["display_department"].isin(departments)]
    if priorities:
        filtered = filtered[filtered["display_priority"].isin(priorities)]
    if statuses:
        filtered = filtered[filtered["status"].isin(statuses)]
    if search.strip():
        query = search.strip()
        matches = filtered["ticket_text"].astype(str).str.contains(
            query, case=False, regex=False, na=False
        ) | filtered["ticket_id"].astype(str).str.contains(
            query, case=False, regex=False, na=False
        )
        filtered = filtered[matches]

    if filtered.empty:
        st.info("No tickets match those filters.")
        return

    table = filtered[
        [
            "ticket_id",
            "ticket_text",
            "display_department",
            "display_priority",
            "status",
            "created_at",
            "record_type",
        ]
    ].copy()
    table["created_at"] = pd.to_datetime(
        table["created_at"], errors="coerce"
    ).dt.strftime("%Y-%m-%d %H:%M")
    table["action"] = "Open below"
    table = table.rename(
        columns={
            "ticket_id": "Ticket ID",
            "ticket_text": "Customer complaint",
            "display_department": "Predicted department",
            "display_priority": "Priority",
            "status": "Status",
            "created_at": "Date received",
            "record_type": "Record type",
            "action": "Action",
        }
    )
    styled_table = table.style.apply(
        lambda values: values.map(_priority_style), subset=["Priority"]
    )
    st.dataframe(styled_table, width="stretch", hide_index=True)

    selected_id = st.selectbox(
        "Select ticket to open",
        filtered["ticket_id"].tolist(),
        format_func=lambda value: (
            f"{value} · "
            f"{filtered.loc[filtered['ticket_id'] == value, 'ticket_text'].iloc[0][:70]}"
        ),
        key="ticket_management_selection",
    )
    if st.button("Open selected ticket", type="primary"):
        st.session_state["selected_ticket_id"] = selected_id
    if st.session_state.get("selected_ticket_id") in tickets["ticket_id"].tolist():
        _render_ticket_details(
            st.session_state["selected_ticket_id"],
            "management",
        )


def _populate_complaint(text):
    st.session_state["complaint_input"] = text


def _render_new_ticket(models):
    if models is None:
        st.error("Ticket classification is unavailable because the trained models could not be loaded.")
        return
    department_model, urgency_model = models
    st.title("New customer ticket")
    st.caption(
        "Predictions use the existing trained TF-IDF models and support triage only. "
        "Verify them before acting; only employees can resolve tickets."
    )
    st.subheader("Example Customer Complaints")
    st.caption("These are sample complaints for demonstration, not real customer tickets.")
    sample_columns = st.columns(4)
    for index, (category, complaint) in enumerate(SAMPLE_COMPLAINTS):
        with sample_columns[index]:
            st.markdown(
                f"""
                <div class="sample-card">
                    <div class="sample-label">SAMPLE · {html.escape(category)}</div>
                    <div>{html.escape(complaint)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button(
                "Try Example",
                key=f"try_example_{index}",
                on_click=_populate_complaint,
                args=(complaint,),
                width="stretch",
            )

    st.subheader("Submit a complaint")
    st.session_state.setdefault("ticket_submission_key", uuid.uuid4().hex)
    with st.form("new_ticket_form"):
        customer_name = st.text_input("Customer name", max_chars=120)
        email = st.text_input("Email address", max_chars=254)
        customer_category = st.selectbox(
            "Customer-selected category (optional)",
            ["Not selected", *CATEGORIES],
        )
        complaint = st.text_area(
            "Complaint description",
            key="complaint_input",
            height=130,
            placeholder="Describe the customer's issue...",
        )
        submitted = st.form_submit_button("Analyze & submit ticket", type="primary")

    if submitted:
        try:
            validate_customer_submission(customer_name, email, complaint)
        except ValueError as error:
            st.error(str(error))
        else:
            try:
                department, urgency = classify_complaint(
                    complaint, department_model, urgency_model
                )
            except (ValueError, TypeError, RuntimeError) as error:
                st.error(f"Ticket classification failed; nothing was saved. {error}")
            else:
                try:
                    ticket_id = create_ticket(
                        customer_name=customer_name,
                        email=email,
                        ticket_text=complaint,
                        department=department,
                        urgency=urgency,
                        customer_category=(
                            "" if customer_category == "Not selected" else customer_category
                        ),
                        source_key=(
                            "customer:"
                            + st.session_state["ticket_submission_key"]
                        ),
                        database_path=DATABASE_PATH,
                    )
                except (ValueError, sqlite3.Error) as error:
                    st.error(f"Could not save this ticket: {error}")
                else:
                    st.session_state["last_created_ticket_id"] = ticket_id
                    st.session_state["ticket_submission_key"] = uuid.uuid4().hex
                    st.success(f"Ticket submitted successfully. Ticket ID: {ticket_id}")

    created_id = st.session_state.get("last_created_ticket_id")
    if created_id:
        _render_ticket_details(created_id, "new")


def _render_batch_processing(models):
    if models is None:
        st.error("Batch classification is unavailable because the trained models could not be loaded.")
        return
    department_model, urgency_model = models
    st.title("Batch ticket processing")
    st.caption("Upload a CSV with a `ticket_text` column to classify multiple complaints.")
    uploaded = st.file_uploader(
        "Upload customer complaints",
        type=["csv"],
        key="batch_ticket_upload",
    )
    if uploaded is None:
        st.info("Choose a CSV file to preview and classify its complaints.")
        return

    try:
        data = pd.read_csv(uploaded)
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        return
    if "ticket_text" not in data.columns:
        st.error("CSV must contain a column named 'ticket_text'.")
        return
    if data.empty:
        st.info("The uploaded CSV contains no ticket rows.")
        return

    cleaned = data["ticket_text"].fillna("").astype(str).map(clean_text)
    valid_rows = cleaned.map(has_readable_text)
    data["predicted_department"] = "Other / Unclassified"
    data["predicted_urgency"] = "Other / Unclassified"
    data["classification_error"] = ""
    data.loc[
        ~valid_rows, "classification_error"
    ] = "Complaint is empty or unreadable after preprocessing."
    if valid_rows.any():
        try:
            departments = department_model.predict(cleaned[valid_rows].tolist())
            urgencies = urgency_model.predict(cleaned[valid_rows].tolist())
        except (ValueError, TypeError, RuntimeError) as error:
            st.error(f"Batch model prediction failed: {error}")
            return
        data.loc[valid_rows, "predicted_department"] = departments
        data.loc[valid_rows, "predicted_urgency"] = urgencies
    data["display_department"] = data["predicted_department"].map(
        category_for_department
    )
    data["display_priority"] = data["predicted_urgency"].map(display_priority)

    filters = st.columns(2)
    with filters[0]:
        selected_departments = st.multiselect(
            "Filter by department",
            sorted(data["display_department"].unique().tolist()),
            key="batch_department_filter",
        )
    with filters[1]:
        selected_priorities = st.multiselect(
            "Filter by priority",
            sorted(data["display_priority"].unique().tolist()),
            key="batch_priority_filter",
        )
    visible = data
    if selected_departments:
        visible = visible[visible["display_department"].isin(selected_departments)]
    if selected_priorities:
        visible = visible[visible["display_priority"].isin(selected_priorities)]
    st.dataframe(visible, width="stretch", hide_index=True)
    if not visible.empty:
        selected_index = st.selectbox(
            "Select a query to inspect",
            visible.index.tolist(),
            format_func=lambda row_index: (
                f"Row {row_index + 1}: "
                f"{str(data.loc[row_index, 'ticket_text']).strip()[:100]}"
            ),
            key="batch_query_selection",
        )
        selected_row = data.loc[selected_index]
        st.subheader("Selected query details")
        st.info(str(selected_row["ticket_text"]))
        prediction_columns = st.columns(2)
        prediction_columns[0].metric(
            "Predicted department", selected_row["display_department"]
        )
        prediction_columns[1].metric(
            "Predicted priority", selected_row["display_priority"]
        )
        if selected_row["classification_error"]:
            st.warning(selected_row["classification_error"])
        else:
            with st.form("batch_selected_demo_import_form"):
                st.caption(
                    "Save only this query as a demo ticket. You can then find "
                    "it in Ticket management → All including demo records."
                )
                selected_confirmed = st.checkbox(
                    "I understand this will be marked as a demo record.",
                    key="batch_selected_demo_confirm",
                )
                selected_submitted = st.form_submit_button(
                    "Save selected query as demo"
                )
            if selected_submitted:
                if not selected_confirmed:
                    st.error(
                        "Confirm that this query will be marked as a demo record."
                    )
                else:
                    try:
                        outcome = import_batch_predictions(
                            data.loc[[selected_index]],
                            uploaded.getvalue(),
                            DATABASE_PATH,
                        )
                    except (ValueError, sqlite3.Error) as error:
                        st.error(f"Could not save selected query: {error}")
                    else:
                        st.success(
                            "Selected query saved as a demo ticket. "
                            f"Imported: {outcome['imported']:,}; "
                            f"already saved: {outcome['skipped']:,}."
                        )
    st.download_button(
        "Download classified CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="ticket_predictions.csv",
        mime="text/csv",
        type="primary",
    )
    st.caption(
        "Batch predictions are only a preview unless you explicitly save them as demo records."
    )
    if valid_rows.any():
        with st.form("batch_demo_import_form"):
            st.caption(
                "Save all readable rows from this upload to Ticket management → "
                "All including demo records. Unreadable rows are excluded; demo "
                "records are not counted as live customer tickets."
            )
            confirmed = st.checkbox(
                "I understand these rows will be marked as demo records.",
                key="batch_all_demo_confirm",
            )
            submitted = st.form_submit_button("Save classified tickets as demos")
        if submitted:
            if not confirmed:
                st.error("Confirm that the imported rows will be marked as demo records.")
            else:
                try:
                    outcome = import_batch_predictions(
                        data.loc[valid_rows],
                        uploaded.getvalue(),
                        DATABASE_PATH,
                    )
                except (ValueError, sqlite3.Error) as error:
                    st.error(f"Could not save batch results: {error}")
                else:
                    st.success(
                        f"Saved {outcome['imported']:,} demo tickets; "
                        f"{outcome['skipped']:,} duplicates were already saved. "
                        "Open Ticket management and select "
                        "'All including demo records' to access them."
                    )
    else:
        st.info("There are no readable complaints to save.")


def _render_admin_import():
    st.title("Admin · Dataset import")
    st.caption(
        "Import labelled examples only when needed for a clearly marked demonstration."
    )
    admin_password = os.environ.get("SUPPORT_ADMIN_PASSWORD")
    if not admin_password:
        st.warning(
            "Dataset import is disabled until SUPPORT_ADMIN_PASSWORD is set in the environment."
        )
        return

    if not st.session_state.get("dataset_admin_authenticated", False):
        with st.form("dataset_admin_login"):
            entered_password = st.text_input("Admin password", type="password")
            login_clicked = st.form_submit_button("Unlock dataset import")
        if login_clicked:
            if hmac.compare_digest(entered_password, admin_password):
                st.session_state["dataset_admin_authenticated"] = True
                st.rerun()
            st.error("Invalid admin password.")
        return

    st.success("Admin access enabled for this session.")
    with st.form("dataset_import_form"):
        uploaded = st.file_uploader(
            "Choose a labelled CSV",
            type=["csv"],
            help="Expected columns: ticket_text, department, urgency.",
        )
        confirm_import = st.checkbox(
            "I understand these will be marked as Demo / Dataset Imported records."
        )
        import_clicked = st.form_submit_button(
            "Import examples into demo records",
            type="primary",
        )
    if not import_clicked:
        st.info("No dataset rows are imported until you confirm this action.")
        return
    if uploaded is None:
        st.error("Choose a CSV file before importing.")
        return
    if not confirm_import:
        st.error("Confirm that the imported rows will be marked as demo records.")
        return

    try:
        uploaded.seek(0)
        frame, report = load_dataset_with_report(uploaded)
        outcome = import_dataset_tickets(frame, DATABASE_PATH)
    except (ValueError, sqlite3.Error) as error:
        st.error(f"Dataset import failed: {error}")
        return
    st.success(
        f"Imported {outcome['imported']:,} demo examples; "
        f"skipped {outcome['skipped']:,} duplicates."
    )
    st.caption(
        f"Input rows: {report['input_records']:,} · "
        f"Usable unique rows: {report['usable_records']:,} · "
        f"Duplicate rows removed: {report['duplicate_records']:,}"
    )
    st.warning(
        "Demo records are labelled and excluded from live customer metrics. "
        "Importing does not train or replace either model."
    )


try:
    initialize_database(DATABASE_PATH)
    tickets = get_all_tickets(DATABASE_PATH)
    statistics = get_dashboard_statistics(DATABASE_PATH)
    department_statistics = get_department_statistics(
        database_path=DATABASE_PATH
    )
    priority_statistics = get_priority_statistics(database_path=DATABASE_PATH)
except sqlite3.Error as error:
    st.error(f"Could not read the ticket database: {error}")
    st.stop()

try:
    model_pair = load_models()
    model_error = None
except (
    FileNotFoundError,
    OSError,
    EOFError,
    ValueError,
    AttributeError,
    ImportError,
    pickle.UnpicklingError,
) as error:
    model_pair = None
    model_error = str(error)

dashboard_data = build_dashboard_data(
    tickets, department_statistics, priority_statistics
)

with st.sidebar:
    st.markdown("## 🎫 Support Intelligence")
    st.caption("Customer operations workspace")
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Ticket management",
            "New ticket",
            "Batch processing",
            "Admin · Dataset import",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(
        f"{statistics['total']:,} customer tickets · "
        f"{statistics['demo_imported']:,} demo records"
    )
    if model_error:
        st.warning("Classification models are unavailable.")

if page == "Overview":
    _render_overview(statistics, dashboard_data)
elif page == "Ticket management":
    _render_ticket_management(tickets)
elif page == "New ticket":
    _render_new_ticket(model_pair)
elif page == "Batch processing":
    _render_batch_processing(model_pair)
else:
    _render_admin_import()
