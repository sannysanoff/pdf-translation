#!/usr/bin/env python3
"""
Extract table data from existing docling.json to create tables.json
"""

import json
import os

def extract_tables_from_docling(docling_path):
    """Extract table data from docling.json and save as tables.json"""
    with open(docling_path, 'r') as f:
        doc_data = json.load(f)
    
    tables = []
    
    if 'tables' in doc_data:
        for table in doc_data['tables']:
            prov = table['prov'][0] if table['prov'] else None
            if prov and 'bbox' in prov:
                bbox = prov['bbox']
                table_data = {
                    'bbox': {'l': bbox['l'], 't': bbox['t'], 'r': bbox['r'], 'b': bbox['b']},
                    'table_cells': []
                }
                
                # Extract table cells from data.table_cells
                if 'data' in table and 'table_cells' in table['data']:
                    for cell in table['data']['table_cells']:
                        cell_bbox = cell['bbox']
                        table_data['table_cells'].append({
                            'bbox': {'l': cell_bbox['l'], 't': cell_bbox['t'], 'r': cell_bbox['r'], 'b': cell_bbox['b']},
                            'text': cell['text'],
                            'row_span': cell['row_span'],
                            'col_span': cell['col_span'],
                            'column_header': cell['column_header'],
                            'row_header': cell['row_header']
                        })
                
                tables.append(table_data)
    
    return tables

if __name__ == '__main__':
    docling_path = '/Users/san/Fun/translate-pdf/workspace/ukr-1/0045/docling.json'
    output_dir = '/Users/san/Fun/translate-pdf/workspace/ukr-1/0045'
    
    tables = extract_tables_from_docling(docling_path)
    
    with open(os.path.join(output_dir, 'tables.json'), 'w') as f:
        json.dump(tables, f, indent=2)
    
    print(f"Extracted {len(tables)} table(s) to tables.json")