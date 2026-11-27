# Project overview and demo flow

**Problem:** Customer support teams receive tickets that must be assigned to the right department and prioritized.

**Input:** A free-text customer ticket.

**Outputs:** Predicted department (Billing / Technical / Account / Shipping) and urgency (Low / Medium / High / Critical). Use the actual label set present in your chosen dataset.

**Pipeline:**
1. Load and inspect labeled ticket data.
2. Clean text while preserving urgency cues such as punctuation and urgent terms.
3. Split data into training and test sets.
4. Train two classifiers: department and urgency.
5. Compare TF-IDF baseline with sentence embeddings.
6. Evaluate with accuracy, macro-F1, per-class metrics, confusion matrices, and error analysis.
7. Use Streamlit to classify a new ticket or process a CSV.

**Example for explaining the demo (illustrative only):**
Ticket: “My payment was deducted but my order is not confirmed. Please resolve this urgently.”
Expected type of output: department and urgency predictions. The actual model output depends on training data; do not present this example as a verified prediction.
