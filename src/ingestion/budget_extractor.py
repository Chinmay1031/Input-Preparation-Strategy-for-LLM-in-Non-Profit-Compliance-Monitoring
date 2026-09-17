
import re
from typing import List, Optional, Tuple
from .document_schema import BudgetLine, ParsedDocument

COUNTRY_IDENTIFIERS = {
    "CONGO":   ["CONGO", "ODZALA"],
    "GABON":   ["GABON"],
    "MOROCCO": ["MOROCCO", "MORROCCO"],                              
    "NAMIBIA": ["NAMIBIA"],
    "SENEGAL": ["SENEGAL"],
    "RWANDA":  ["RWANDA"],
    "GENERAL": ["OPS", "DEV:", "M&E", "STAFF", "TRAVEL"],
}

SIGNIFICANT_VARIANCE_PCT = 0.20                
OVERSPEND_THRESHOLD = 0.10                                 


def _clean_number(raw: str) -> Optional[float]:
    if not raw:
        return None
    cleaned = re.sub(r'[€$£,\s]', '', str(raw).strip())
                                          
    if cleaned.startswith('(') and cleaned.endswith(')'):
        cleaned = '-' + cleaned[1:-1]
    try:
        return float(cleaned)
    except ValueError:
        return None


def _detect_country(line_text: str) -> Optional[str]:
    upper = line_text.upper()
    for country, keywords in COUNTRY_IDENTIFIERS.items():
        if any(kw in upper for kw in keywords):
            return country
    return None


def extract_budget_lines_from_text(annexure_a_text: str) -> List[BudgetLine]:
    budget_lines = []
    lines = annexure_a_text.split('\n')

                                                                  
    number_pattern = re.compile(
        r'^(.+?)\s+([\d,.\(\)]+)\s+([\d,.\(\)]+)\s+([\d,.\(\)]+)\s*(.*)$'
    )

    current_country = None

    for line in lines:
        line = line.strip()
        if not line or len(line) < 5:
            continue

        detected_country = _detect_country(line)
        if detected_country:
            current_country = detected_country

        match = number_pattern.match(line)
        if not match:
            continue

        category_raw = match.group(1).strip()
        actual_raw   = match.group(2)
        budget_raw   = match.group(3)
        variance_raw = match.group(4)
        comment      = match.group(5).strip() if match.group(5) else None

        if any(h in category_raw.upper() for h in [
            'CATEGORY', 'PROCEDURE', 'ACTUAL', 'BUDGET', 'VARIANCE', 'NO.'
        ]):
            continue

        actual   = _clean_number(actual_raw)
        budget   = _clean_number(budget_raw)
        variance = _clean_number(variance_raw)

        if actual is None or budget is None:
            continue

        variance_pct = None
        if budget and budget != 0:
            variance_pct = abs(variance or 0) / abs(budget)

        budget_line = BudgetLine(
            category=category_raw,
            actual=actual,
            budget=budget,
            variance=variance or (actual - budget),
            variance_pct=variance_pct,
            comment=comment or None,
            country=_detect_country(category_raw) or current_country,
            is_overspend=(actual > budget * (1 + OVERSPEND_THRESHOLD)) if budget > 0 else False,
            is_unbudgeted=(budget == 0 and actual > 0),
        )

        budget_lines.append(budget_line)

    return budget_lines


def extract_budget_lines_from_tables(tables: List) -> List[BudgetLine]:
    budget_lines = []
    current_country = None

    for table in tables:
        if not table:
            continue

        for row in table:
            if not row or len(row) < 3:
                continue

            cells = [str(c).strip() if c else "" for c in row]

            first = cells[0].upper()
            if any(h in first for h in [
                'CATEGORY', 'NO.', 'PROCEDURE', 'ITEM', 'DATE PER',
                'AMOUNT PER', 'DESCRIPTION', 'CURRENCY', 'ANNEXURE B',
                'EURO EQUIVALENT', 'PETTY CASH'
            ]):
                continue
            if not first or first in ['', 'NONE']:
                continue

                                                                          
                                                                       
            if re.match(r'^\d+$', first):
                continue

            country = _detect_country(cells[0])
            if country:
                current_country = country

            numbers = []
            comment = None
            for cell in cells[1:4]:                                 
                n = _clean_number(cell)
                if n is not None:
                    numbers.append(n)

            if len(cells) > 4:
                for cell in cells[4:]:
                    if cell and len(cell) > 5 and not any(c.isdigit() for c in cell[:3]):
                        comment = cell
                        break

            if len(numbers) < 2:
                continue

            actual  = numbers[0]
            budget  = numbers[1] if len(numbers) > 1 else 0
            variance = numbers[2] if len(numbers) > 2 else (actual - budget)

                                                                         
                                                                            
            if budget > 10_000_000:
                continue

            variance_pct = None
            if budget and budget != 0:
                variance_pct = abs(variance) / abs(budget)

            bl = BudgetLine(
                category=cells[0],
                actual=actual,
                budget=budget,
                variance=variance,
                variance_pct=variance_pct,
                comment=comment,
                country=_detect_country(cells[0]) or current_country,
                is_overspend=(actual > budget * 1.10) if budget > 0 else False,
                is_unbudgeted=(budget == 0 and actual > 0),
            )
            budget_lines.append(bl)

    return budget_lines


def summarise_budget_lines(budget_lines: List[BudgetLine]) -> dict:
    if not budget_lines:
        return {}

    total_actual = sum(bl.actual for bl in budget_lines if bl.actual)
    total_budget = sum(bl.budget for bl in budget_lines if bl.budget)
    overspends   = [bl for bl in budget_lines if bl.is_overspend]
    unbudgeted   = [bl for bl in budget_lines if bl.is_unbudgeted]

    top_overspends = sorted(
        overspends,
        key=lambda x: abs(x.variance),
        reverse=True
    )[:5]

    country_totals = {}
    for bl in budget_lines:
        if bl.country:
            if bl.country not in country_totals:
                country_totals[bl.country] = {"actual": 0, "budget": 0}
            country_totals[bl.country]["actual"]  += bl.actual or 0
            country_totals[bl.country]["budget"]  += bl.budget or 0

    return {
        "total_actual":      total_actual,
        "total_budget":      total_budget,
        "total_variance":    total_actual - total_budget,
        "budget_utilisation": total_actual / total_budget if total_budget else None,
        "overspend_count":   len(overspends),
        "unbudgeted_count":  len(unbudgeted),
        "top_overspends":    top_overspends,
        "unbudgeted_items":  unbudgeted,
        "country_totals":    country_totals,
    }
