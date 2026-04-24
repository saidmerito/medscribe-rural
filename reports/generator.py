# reports/generator.py
"""
Automated report generation for MedScribe Rural.
Produces PDF epidemiological reports and CSV exports from SQLite data.
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from config import REPORTS_DIR, EPIDEMIC_THRESHOLDS
from db.database import get_records, get_disease_summary, get_flagged_records, get_stats

os.makedirs(REPORTS_DIR, exist_ok=True)


# ─── PDF REPORT ──────────────────────────────────────────────────────────────

def generate_pdf_report(facility: str = None, days: int = 7) -> str:
    """Generate a PDF epidemiological report. Returns the file path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    fname = f"medscribe_report_{facility or 'all'}_{timestamp}.pdf"
    fpath = os.path.join(REPORTS_DIR, fname)

    doc = SimpleDocTemplate(fpath, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"],
                                 fontSize=18, textColor=colors.HexColor("#1A5276"),
                                 spaceAfter=6, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"],
                                    fontSize=11, textColor=colors.HexColor("#2E86C1"),
                                    spaceAfter=4, alignment=TA_CENTER)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"],
                                   fontSize=13, textColor=colors.HexColor("#1A5276"),
                                   spaceBefore=16, spaceAfter=6)
    normal = styles["Normal"]
    alert_style = ParagraphStyle("Alert", parent=styles["Normal"],
                                 fontSize=10, textColor=colors.HexColor("#C0392B"),
                                 backColor=colors.HexColor("#FADBD8"),
                                 borderPadding=6, spaceAfter=4)

    stats = get_stats()
    summary = get_disease_summary(days=days)
    records = get_records(facility=facility, days=days)
    flagged = get_flagged_records()

    # ── Epidemic alerts
    alerts = []
    for row in summary:
        threshold = EPIDEMIC_THRESHOLDS.get(row["icd10_code"])
        if threshold and row["case_count"] >= threshold:
            alerts.append(row)

    story = []

    # Header
    story.append(Paragraph("MedScribe Rural", title_style))
    story.append(Paragraph(f"Epidemiological Report — {facility or 'All Facilities'}", subtitle_style))
    story.append(Paragraph(f"Period: Last {days} days | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2E86C1")))
    story.append(Spacer(1, 0.4*cm))

    # Summary stats
    story.append(Paragraph("📊 Summary Statistics", heading_style))
    stat_data = [
        ["Total Records", "Facilities", "Flagged for Review", "Last Import"],
        [str(stats["total_records"]), str(stats["facilities"]),
         str(stats["flagged_records"]), str(stats["last_import"])[:16]]
    ]
    stat_table = Table(stat_data, colWidths=[4.2*cm]*4)
    stat_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E86C1")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#EBF5FB"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#AED6F1")),
        ("ROWHEIGHT", (0, 0), (-1, -1), 20),
    ]))
    story.append(stat_table)
    story.append(Spacer(1, 0.4*cm))

    # Epidemic alerts
    if alerts:
        story.append(Paragraph("🚨 Epidemic Alerts", heading_style))
        for alert in alerts:
            story.append(Paragraph(
                f"⚠ ALERT: {alert['case_count']} cases of {alert['diagnosis']} "
                f"({alert['icd10_code']}) at {alert['facility']} — "
                f"exceeds threshold of {EPIDEMIC_THRESHOLDS[alert['icd10_code']]}",
                alert_style
            ))
        story.append(Spacer(1, 0.3*cm))

    # Disease breakdown
    story.append(Paragraph("🦠 Disease Breakdown", heading_style))
    if summary:
        dis_data = [["ICD-10", "Diagnosis", "Cases", "Facility"]]
        for row in summary[:20]:
            dis_data.append([row["icd10_code"], row["diagnosis"],
                             str(row["case_count"]), row["facility"]])
        dis_table = Table(dis_data, colWidths=[2*cm, 7*cm, 2*cm, 6.4*cm])
        dis_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A5276")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#EBF5FB"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#AED6F1")),
            ("ROWHEIGHT", (0, 0), (-1, -1), 18),
        ]))
        story.append(dis_table)
    else:
        story.append(Paragraph("No records found for this period.", normal))

    story.append(Spacer(1, 0.4*cm))

    # Flagged records
    if flagged:
        story.append(Paragraph("⚑ Records Requiring Manual Review", heading_style))
        story.append(Paragraph(
            f"{len(flagged)} record(s) have low confidence or uncertain fields. "
            "Please verify these entries in the system.", normal))

    # Footer
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#AED6F1")))
    story.append(Paragraph(
        "Generated by MedScribe Rural — Powered by Gemma 4 E4B (offline AI) | Gemma 4 Good Hackathon 2026",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8,
                       textColor=colors.grey, alignment=TA_CENTER)
    ))

    doc.build(story)
    return fpath


# ─── CSV EXPORT ──────────────────────────────────────────────────────────────

def generate_csv_export(facility: str = None, days: int = 30) -> str:
    """Export records as CSV. Returns the file path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    fname = f"medscribe_export_{facility or 'all'}_{timestamp}.csv"
    fpath = os.path.join(REPORTS_DIR, fname)

    records = get_records(facility=facility, days=days)

    if not records:
        return None

    fieldnames = [
        "id", "patient_id", "facility", "date", "age", "sex",
        "chief_complaint", "diagnosis", "icd10_code", "treatment",
        "outcome", "confidence", "flags", "extracted_at"
    ]

    with open(fpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            # Parse flags back to string for CSV
            if isinstance(record.get("flags"), str):
                try:
                    record["flags"] = ", ".join(json.loads(record["flags"]))
                except Exception:
                    pass
            writer.writerow(record)

    return fpath
