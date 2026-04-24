# app/gradio_ui.py
"""
Gradio interface for MedScribe Rural.
Simple, accessible UI designed for health workers in low-resource settings.
"""

import gradio as gr
import json
import tempfile
import os
from pathlib import Path

from ocr.extractor import extract_text
from ai.structurer import structure_records
from db.database import init_db, insert_records, log_import, get_stats, get_disease_summary, get_flagged_records
from reports.generator import generate_pdf_report, generate_csv_export

# Initialize DB on startup
init_db()

FACILITIES = [
    "Centre de Santé Ali Sabieh",
    "Centre de Santé Arta",
    "Centre de Santé Dikhil",
    "Centre de Santé Obock",
    "Centre de Santé Tadjourah",
    "Autre / Other",
]


def process_register_image(image_path: str, facility: str) -> tuple:
    """Full pipeline: image → OCR → Gemma 4 → DB → display."""
    if image_path is None:
        return "❌ Please upload an image.", "", ""

    if not facility:
        return "❌ Please select a facility.", "", ""

    try:
        # Step 1: OCR
        yield "⏳ Step 1/3: Extracting text with OCR...", "", ""
        raw_text = extract_text(image_path)

        if not raw_text.strip():
            yield "❌ OCR extracted no text. Check image quality.", "", ""
            return

        # Step 2: AI Structuring
        yield f"⏳ Step 2/3: Gemma 4 is structuring {len(raw_text)} characters...", raw_text, ""
        records = structure_records(raw_text, facility_name=facility)

        if not records:
            yield "⚠️ No patient records could be extracted. Try a clearer image.", raw_text, ""
            return

        # Step 3: Save to DB
        yield f"⏳ Step 3/3: Saving {len(records)} records to database...", raw_text, ""
        n = insert_records(records)
        log_import(filename=Path(image_path).name, facility=facility, n_records=n)

        # Format output
        records_json = json.dumps(records, indent=2, ensure_ascii=False)
        status = (
            f"✅ Successfully extracted and saved **{n} patient records** from {facility}.\n\n"
            f"- Records with flags: {sum(1 for r in records if r.get('flags'))}\n"
            f"- Average confidence: {sum(r.get('confidence', 0) for r in records)/len(records):.0%}"
        )

        yield status, raw_text, records_json

    except Exception as e:
        yield f"❌ Error: {str(e)}", "", ""


def get_dashboard_data():
    """Fetch dashboard stats for display."""
    stats = get_stats()
    summary = get_disease_summary(days=7)
    flagged = get_flagged_records()

    stats_md = f"""
### 📊 Database Overview
| Metric | Value |
|--------|-------|
| Total Records | **{stats['total_records']}** |
| Facilities | **{stats['facilities']}** |
| Flagged Records | **{stats['flagged_records']}** |
| Last Import | {stats['last_import']} |
"""

    if summary:
        disease_md = "### 🦠 Top Diagnoses (Last 7 Days)\n| ICD-10 | Diagnosis | Cases | Facility |\n|--------|-----------|-------|----------|\n"
        for row in summary[:10]:
            disease_md += f"| {row['icd10_code']} | {row['diagnosis']} | **{row['case_count']}** | {row['facility']} |\n"
    else:
        disease_md = "### 🦠 No records yet."

    if flagged:
        flagged_md = f"### ⚑ {len(flagged)} Records Need Review\n"
        for r in flagged[:5]:
            flagged_md += f"- **{r['patient_id']}** ({r['date']}) — {r['diagnosis']} — Confidence: {r['confidence']:.0%}\n"
    else:
        flagged_md = "### ✅ No flagged records."

    return stats_md, disease_md, flagged_md


def generate_report_action(facility: str, days: int, report_type: str):
    """Generate and return a report file."""
    facility_filter = None if facility == "All Facilities" else facility
    try:
        if report_type == "PDF Report":
            path = generate_pdf_report(facility=facility_filter, days=int(days))
        else:
            path = generate_csv_export(facility=facility_filter, days=int(days))

        if path and os.path.exists(path):
            return f"✅ Report generated: {Path(path).name}", path
        else:
            return "⚠️ No data found for the selected period.", None
    except Exception as e:
        return f"❌ Error generating report: {e}", None


# ─── GRADIO UI ────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="MedScribe Rural",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"),
        css="""
            .header { text-align: center; padding: 20px; }
            .badge { display: inline-block; background: #2E86C1; color: white;
                     border-radius: 12px; padding: 2px 10px; font-size: 12px; }
        """
    ) as demo:

        # Header
        gr.Markdown("""
        # 🏥 MedScribe Rural
        ### Offline AI-Powered Medical Register Digitization
        <span class="badge">Gemma 4 E4B</span> &nbsp;
        <span class="badge">100% Offline</span> &nbsp;
        <span class="badge">Multilingual</span>
        ---
        """)

        with gr.Tabs():

            # ── TAB 1: DIGITIZE ──────────────────────────────────────────────
            with gr.Tab("📸 Digitize Register"):
                gr.Markdown("Upload a photo of a handwritten medical register page. Gemma 4 will extract and structure the patient records.")

                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            type="filepath",
                            label="Register Photo",
                            height=350
                        )
                        facility_input = gr.Dropdown(
                            choices=FACILITIES,
                            label="Health Center / Facility",
                            value=FACILITIES[0]
                        )
                        process_btn = gr.Button("🤖 Extract Records with Gemma 4", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        status_output = gr.Markdown(label="Status")
                        with gr.Accordion("Raw OCR Text", open=False):
                            ocr_output = gr.Textbox(label="OCR Output", lines=8, interactive=False)
                        records_output = gr.Code(label="Structured Records (JSON)", language="json", lines=15)

                process_btn.click(
                    fn=process_register_image,
                    inputs=[image_input, facility_input],
                    outputs=[status_output, ocr_output, records_output]
                )

            # ── TAB 2: DASHBOARD ─────────────────────────────────────────────
            with gr.Tab("📊 Dashboard"):
                refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="secondary")

                with gr.Row():
                    stats_output = gr.Markdown()
                    disease_output = gr.Markdown()

                flagged_output = gr.Markdown()

                refresh_btn.click(
                    fn=get_dashboard_data,
                    inputs=[],
                    outputs=[stats_output, disease_output, flagged_output]
                )
                # Auto-load on tab render
                demo.load(get_dashboard_data, outputs=[stats_output, disease_output, flagged_output])

            # ── TAB 3: REPORTS ───────────────────────────────────────────────
            with gr.Tab("📄 Generate Report"):
                gr.Markdown("Generate PDF or CSV reports for health authorities.")

                with gr.Row():
                    report_facility = gr.Dropdown(
                        choices=["All Facilities"] + FACILITIES,
                        label="Facility",
                        value="All Facilities"
                    )
                    report_days = gr.Slider(
                        minimum=7, maximum=90, value=7, step=1,
                        label="Period (days)"
                    )
                    report_type = gr.Radio(
                        choices=["PDF Report", "CSV Export"],
                        value="PDF Report",
                        label="Report Type"
                    )

                report_btn = gr.Button("📥 Generate Report", variant="primary")
                report_status = gr.Markdown()
                report_file = gr.File(label="Download Report")

                report_btn.click(
                    fn=generate_report_action,
                    inputs=[report_facility, report_days, report_type],
                    outputs=[report_status, report_file]
                )

            # ── TAB 4: ABOUT ─────────────────────────────────────────────────
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## About MedScribe Rural

                **MedScribe Rural** digitizes handwritten medical registers from rural health centers
                in the Horn of Africa using AI — fully offline, no internet required.

                ### How It Works
                1. **📸 Capture** — Photograph a register page
                2. **🔍 OCR** — PaddleOCR extracts text (French, Arabic, Somali)
                3. **🤖 AI** — Gemma 4 E4B corrects errors, structures data, assigns ICD-10 codes
                4. **💾 Store** — Records saved to local SQLite database
                5. **📄 Report** — Automated epidemiological reports generated

                ### Why It Matters
                Rural health centers in Djibouti and across the Horn of Africa lack data pipelines.
                Disease outbreaks go undetected for weeks. MedScribe Rural bridges this gap.

                ### Technology
                - **AI Model**: Gemma 4 E4B via Ollama (runs on laptop CPU, 8GB RAM)
                - **OCR**: PaddleOCR (multilingual)
                - **Backend**: Python + FastAPI
                - **UI**: Gradio

                ---
                *Built for the Gemma 4 Good Hackathon 2026 · Track: Health & Sciences*
                """)

    return demo
