# -*- coding: utf-8 -*-
import csv
import re

def insert_commas(text):
    m = re.search(r'\bvisit\b', text, flags=re.IGNORECASE)
    if not m:
        return text 
    idx = m.end()
    prefix = text[:idx].rstrip()
    if not prefix.endswith(','):
        prefix += ','

    suffix = text[idx:].strip()
    #  ['CDRSB', '0.0', 'ADAS11', '10.0', ...])
    tokens = suffix.split()
    pairs = []
    for i in range(0, len(tokens), 2):
        if i + 1 < len(tokens):
            pairs.append(f"{tokens[i]} {tokens[i+1]}")
        else:
            pairs.append(tokens[i])
    new_suffix = ', '.join(pairs)
    return prefix + ' ' + new_suffix

input_csv = '/data1/ADP-DiT/dataset/AD_meta/csvfile/image_text.csv'
output_csv = '/data1/ADP-DiT/dataset/AD_meta/csvfile/metadata_modified.csv'

with open(input_csv, newline='', encoding='utf-8') as f_in, open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)
    
    header = next(reader)
    writer.writerow(header)
    
    for row in reader:
        original_text = row[1]
        modified_text = insert_commas(original_text)
        row[1] = modified_text
        writer.writerow(row)

print("CSV complete.")

# import csv

# input_csv = '/data1/ADP-DiT/dataset/AD_meta/csvfile/metadata_modified.csv'
# output_csv = '/data1/ADP-DiT/dataset/AD_meta/csvfile/metadata_modified_lowercase_first_visit.csv'

# with open(input_csv, 'r', encoding='utf-8', newline='') as fin, \
#      open(output_csv, 'w', encoding='utf-8', newline='') as fout:
#     reader = csv.reader(fin)
#     writer = csv.writer(fout)

#     # 헤더 처리 (첫 줄)
#     header = next(reader)
#     writer.writerow(header)

#     # 각 행의 두 번째 칼럼(text_en)에서 "First visit"를 "first visit"로 변경
#     for row in reader:
#         row[1] = row[1].replace("First visit", "first visit")
#         writer.writerow(row)

# print("CSV 파일의 'First visit' 문자열이 'first visit'로 변경된 새 파일을 생성했습니다.")
