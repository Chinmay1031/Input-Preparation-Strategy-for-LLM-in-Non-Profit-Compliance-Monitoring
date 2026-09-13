"""
s3_field_extractor.py
---------------------
Strategy 3 — Structured field extraction.
Extracts only compliance-relevant figures and builds a compact
structured narrative. This is the hypothesis strategy of the thesis.
Expected tokens: ~600 to 1,000 per document.
"""

import re
from src.ingestion.document_schema import (
    ParsedDocument,
    DOCUMENT_TYPE_AUP_REPORT,
    DOCUMENT_TYPE_FINANCIAL_STATEMENT,
)
from src.ingestion.budget_extractor import summarise_budget_lines


def _yoy(current, prior):
    """Compute year-on-year percentage change."""
    if prior and prior != 0:
        pct = ((current - prior) / abs(prior)) * 100
        return f"{pct:+.0f}%"
    return "N/A"


def _extract_figures_from_text(text: str) -> dict:
    """
    Extracts financial figures by scanning each line for known labels.
    Handles space-separated digit groups common in pdfplumber output.
    """
    figures = {}
    lines = text.split('\n')

    TARGETS = {
        "Donations received":  ("revenue_current",        "revenue_prior"),
        "Employee costs":      ("employee_costs_current",  "employee_costs_prior"),
        "Staff welfare":       ("staff_welfare_current",   "staff_welfare_prior"),
        "Profit for the year": ("profit_current",          "profit_prior"),
    }

    def extract_nums(line):
        """
        Extract numbers from a line, handling space-separated digit groups
        common in pdfplumber output e.g. '4 437 800' becomes 4437800.
        Only accepts numbers between 1,000 and 50,000,000.
        """
        # Join space-separated digit groups into full numbers
        # Matches patterns like '4 437 800' or '792 646'
        joined = re.sub(
            r'(\d{1,3})((?:\s\d{3})+)',
            lambda m: m.group(1) + m.group(2).replace(' ', ''),
            line
        )
        # Extract all numeric tokens from the joined line
        tokens = re.findall(r'\d+', joined)
        results = []
        for token in tokens:
            try:
                v = float(token)
                if 1_000 <= v <= 50_000_000:
                    results.append(v)
            except ValueError:
                continue
        return results

    for line in lines:
        stripped = line.strip()

        for label, (cur_key, pri_key) in TARGETS.items():
            if stripped.lower().startswith(label.lower()):
                nums = extract_nums(stripped)
                if len(nums) >= 2:
                    figures[cur_key] = nums[0]
                    figures[pri_key] = nums[1]
                elif len(nums) == 1:
                    figures[cur_key] = nums[0]
                break

        # Cash equivalents
        if stripped.lower().startswith("cash and cash equivalent"):
            nums = extract_nums(stripped)
            if len(nums) >= 2:
                figures["cash_current"] = nums[0]
                figures["cash_prior"]   = nums[1]
            elif len(nums) == 1:
                figures["cash_current"] = nums[0]

        # Donations paid — standalone Donations line in expenses
        if re.match(r'^Donations\s+\d', stripped, re.IGNORECASE):
            nums = extract_nums(stripped)
            if nums:
                figures["donations_paid"] = nums[0]

    return figures


def prepare_s3(doc: ParsedDocument) -> str:
    """
    Builds a compact structured compliance narrative from extracted
    financial figures. Sends only what matters — nothing else.
    """
    lines = []
    lines.append("GRANTEE COMPLIANCE SUMMARY")
    lines.append(f"Grantee:       {doc.grantee_name or 'Unknown'}")
    lines.append(f"Document type: {doc.document_type or 'Unknown'}")
    lines.append(f"Fiscal year:   {doc.fiscal_year or 'Unknown'}")
    lines.append(f"Currency:      {doc.currency or 'Unknown'}")
    lines.append(f"Auditor:       {doc.auditor or 'Not identified'}")
    lines.append("")

    # ── Financial statement path ──────────────────────────────────────────────
    if doc.document_type == DOCUMENT_TYPE_FINANCIAL_STATEMENT:
        figs = doc.financial_figures

        # fallback to line-by-line extraction if table parsing yielded nothing
        if not figs:
            figs = _extract_figures_from_text(doc.full_text)

        lines.append("KEY FINANCIAL INDICATORS:")

        if "revenue_current" in figs:
            lines.append(
                f"  Revenue:         {doc.currency} {figs['revenue_current']:>12,.0f}"
                + (f"  (prior: {figs.get('revenue_prior', 0):,.0f},"
                   f" YoY: {_yoy(figs['revenue_current'], figs.get('revenue_prior', 0))})"
                   if "revenue_prior" in figs else "")
            )

        if "employee_costs_current" in figs:
            lines.append(
                f"  Employee costs:  {doc.currency} {figs['employee_costs_current']:>12,.0f}"
                + (f"  (prior: {figs.get('employee_costs_prior', 0):,.0f},"
                   f" YoY: {_yoy(figs['employee_costs_current'], figs.get('employee_costs_prior', 0))})"
                   if "employee_costs_prior" in figs else "")
            )

        if "staff_welfare_current" in figs:
            lines.append(
                f"  Staff welfare:   {doc.currency} {figs['staff_welfare_current']:>12,.0f}"
                + (f"  (prior: {figs.get('staff_welfare_prior', 0):,.0f},"
                   f" YoY: {_yoy(figs['staff_welfare_current'], figs.get('staff_welfare_prior', 0))})"
                   if "staff_welfare_prior" in figs else "")
            )

        if "donations_paid" in figs:
            lines.append(
                f"  Donations paid:  {doc.currency} {figs['donations_paid']:>12,.0f}"
                f"  (grantee disbursing funds to others — review required)"
            )

        if "profit_current" in figs:
            lines.append(
                f"  Net profit:      {doc.currency} {figs['profit_current']:>12,.0f}"
            )

        if "cash_current" in figs:
            lines.append(
                f"  Cash balance:    {doc.currency} {figs['cash_current']:>12,.0f}"
            )

        lines.append("")

        # Audit opinion
        audit_text = doc.get_section_text("audit_report").lower()
        if "nothing has come to our attention" in audit_text:
            lines.append("AUDIT / REVIEW OPINION:")
            lines.append("  Clean limited assurance conclusion — no material misstatements.")
        elif "unmodified" in audit_text or "fairly present" in audit_text:
            lines.append("AUDIT / REVIEW OPINION:")
            lines.append("  Unmodified audit opinion issued.")
        elif "qualified" in audit_text:
            lines.append("AUDIT / REVIEW OPINION:")
            lines.append("  QUALIFIED opinion — see audit report.")
        lines.append("")

        # Going concern
        gc_text = (doc.get_section_text("directors_report") +
                   doc.get_section_text("notes")).lower()
        if "going concern" in gc_text and "not aware" in gc_text:
            lines.append("GOING CONCERN: No doubt expressed by directors or reviewer.")
        elif "substantial doubt" in gc_text or "going concern" in gc_text:
            lines.append("GOING CONCERN: Concern flagged — review required.")
        lines.append("")

        # Related parties — only specific meaningful lines
        notes_text = doc.get_section_text("notes")
        if "related part" in notes_text.lower():
            lines.append("RELATED PARTY / REVENUE SOURCE:")
            for ln in notes_text.split("\n"):
                stripped = ln.strip()
                if (len(stripped) > 5 and len(stripped) < 120 and
                    any(kw in stripped.lower() for kw in [
                        "hasso plattner", "revenue received",
                        "spac", "embark", "common interest",
                        "donations received", "recovery"
                    ])):
                    lines.append(f"  {stripped}")
            lines.append("")

    # ── AUP report path ───────────────────────────────────────────────────────
    elif doc.document_type == DOCUMENT_TYPE_AUP_REPORT:
        figs = doc.financial_figures
        lines.append("ENGAGEMENT SUMMARY:")
        lines.append(
            f"  Type: Agreed-Upon Procedures (ISRS 4400) — "
            f"no audit opinion expressed."
        )
        if "procedure_count" in figs:
            lines.append(f"  Procedures performed: {int(figs['procedure_count'])}")
        if "no_exceptions_count" in figs:
            lines.append(
                f"  Procedures with no exceptions: "
                f"{int(figs['no_exceptions_count'])} of "
                f"{int(figs.get('procedure_count', '?'))}"
            )
        if "donations_received" in figs:
            lines.append(
                f"  Total donations recorded: "
                f"{doc.currency} {figs['donations_received']:,.0f}"
            )
        if "total_expenses" in figs:
            lines.append(
                f"  Total operating expenses: "
                f"{doc.currency} {figs['total_expenses']:,.0f}"
            )
        if "closing_balance" in figs:
            lines.append(
                f"  Closing balance: "
                f"{doc.currency} {figs['closing_balance']:,.0f}"
            )
        lines.append("")

        # Budget summary from extracted lines
        if doc.budget_lines:
            summary = summarise_budget_lines(doc.budget_lines)
            lines.append("BUDGET VS ACTUAL SUMMARY:")
            if summary.get("overspend_count"):
                lines.append(
                    f"  Overspend items (>10% over budget): "
                    f"{summary['overspend_count']}"
                )
            if summary.get("unbudgeted_count"):
                lines.append(
                    f"  Unbudgeted items (budget=0, actual>0): "
                    f"{summary['unbudgeted_count']}"
                )
            if summary.get("top_overspends"):
                lines.append("  Top overspends:")
                for bl in summary["top_overspends"][:3]:
                    pct = f"{bl.variance_pct:.0%}" if bl.variance_pct else "N/A"
                    lines.append(
                        f"    {bl.category[:45]:<45} "
                        f"actual={bl.actual:>10,.0f}  "
                        f"budget={bl.budget:>10,.0f}  "
                        f"variance={pct}"
                    )
            lines.append("")

        # Key procedure findings
        purpose_text = doc.get_section_text("aup_purpose")
        if "no exceptions" in purpose_text.lower():
            lines.append(
                "AUDITOR CONCLUSION: All agreed procedures "
                "returned no exceptions."
            )
        lines.append("")

        # Pass-through check
        aup_text = doc.full_text.lower()
        if "bank statement" in aup_text and "annexure a" in aup_text:
            lines.append(
                "PASS-THROUGH NOTE: Difference between bank receipts "
                "and Annexure A donations noted. "
                "Attributed to separate project. Verify independently."
            )

    lines.append("")
    lines.append(
        "NOTE: This summary was generated by structured field extraction. "
        "Verify all figures against the source document."
    )

    return "\n".join(lines).strip()