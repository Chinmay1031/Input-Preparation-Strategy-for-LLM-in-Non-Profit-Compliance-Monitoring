import sys
sys.path.insert(0, '.')
import re

from src.output_processing.faithfulness_checker import (
    _extract_figures, _best_match_score
)
from src.ingestion import parse_document

doc = parse_document(
    'data/pdfs/Somos+2024+Audited+Financial+Statements+-+Final.pdf',
    'somos'
)

                                                 
for line in doc.full_text.split('\n'):
    if '480' in line and '210' in line:
        print('RAW LINE:', repr(line.strip()[:120]))
        print('EXTRACTED:', _extract_figures(line))
        print()