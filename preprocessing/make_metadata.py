import pandas as pd
import re

# 기존 CSV 파일 경로
existing_csv_path = "Final_A_with_Descriptive_Text.csv"  # 기존 CSV 경로
image_csv_path = "All_Subjects_FHQ.csv"  # PTID를 포함한 CSV 경로
output_csv_path = "joined_output.csv"  # 병합된 결과를 저장할 CSV 경로

# Step 1: CSV 파일 읽기
existing_df = pd.read_csv(existing_csv_path)
image_df = pd.read_csv(image_csv_path)

# Step 2: 병합 기준 열 이름 통일
# 'Subject'를 'PTID'로 변경하여 병합 준비
existing_df.rename(columns={'Subject': 'PTID'}, inplace=True)

# Step 3: 두 데이터프레임 병합
merged_df = pd.merge(existing_df, image_df, on='PTID', how='inner')  # 'inner' 병합 (공통값만 포함)

# Step 4: 병합 결과 저장
merged_df.to_csv(output_csv_path, index=False)

print(f"Merged CSV saved to {output_csv_path}")

# Step 1: A와 image_data.csv 읽기
a_df = pd.read_csv(existing_csv_path)
image_data_df = pd.read_csv(image_csv_path)
image_data_df.rename(columns={'PTID': 'Subject'}, inplace=True)

# Step 2: A의 Image Path에서 Subject 추출
def extract_subject(image_path):
    # 정규식을 사용해 n+_S_n+ 형식의 문자열 추출
    match = re.search(r'\d+_S_\d+', image_path)
    return match.group(0) if match else None

image_data_df = image_data_df.drop_duplicates(subset=["Subject"], keep="first")

# Apply 함수로 Subject 열 생성
a_df['Subject'] = a_df['Image Path'].apply(extract_subject)

# Step 3: A와 image_data.csv 병합 (A를 기준으로)
merged_df = pd.merge(a_df, image_data_df, on="Subject", how="left")  # A를 기준으로 병합

# Step 4: Text 열 삭제
merged_df.drop(columns=["Text"], inplace=True)

# Step 5: 최종 결과 저장
merged_df.to_csv(output_csv_path, index=False)


image_data_csv_path = "All_Subjects_FHQ.csv"  # image_data.csv 파일 경로
b_csv_path = "All_Subjects_PTDEMOG.csv"  # B 파일 경로
output_csv_path = "final_output_.csv"  # 최종 저장 파일 경로

# Step 1: A와 image_data.csv 읽기
a_df = pd.read_csv(output_csv_path)
image_data_df = pd.read_csv(image_data_csv_path)
b_df = pd.read_csv(b_csv_path)

# Step 2: A의 Image Path에서 Subject 추출
def extract_subject(image_path):
    # 정규식을 사용해 n+_S_n+ 형식의 문자열 추출
    match = re.search(r'\d+_S_\d+', image_path)
    return match.group(0) if match else None

# Apply 함수로 Subject 열 생성
a_df['Subject'] = a_df['Image Path'].apply(extract_subject)
image_data_df.rename(columns={"PTID": "Subject"}, inplace=True)

# Step 3: image_data.csv에서 중복 제거 (Subject 기준)
image_data_df = image_data_df.drop_duplicates(subset=["Subject"], keep="first")

# Step 4: A와 image_data.csv 병합 (A를 기준으로)
merged_df = pd.merge(a_df, image_data_df, on="Subject", how="inner")

# Step 5: B.csv에서 PTRACCAT 열까지 선택
ptraccat_index = b_df.columns.get_loc("PTRACCAT")  # PTRACCAT 열의 인덱스 찾기
b_df = b_df.iloc[:, :ptraccat_index + 1]  # PTRACCAT 열 포함 이전까지 선택
b_df.rename(columns={"PTID": "Subject"}, inplace=True)

# Step 6: B.csv에서 중복 제거 (Subject 기준)
b_df = b_df.drop_duplicates(subset=["Subject"], keep="first")

# Step 7: A와 B 병합
merged_df = pd.merge(merged_df, b_df, on="Subject", how="inner")

# Step 8: 중복된 열 처리 (_x, _y 제거)
# Identify and resolve duplicate columns
for col in merged_df.columns:
    if "_x" in col and col.replace("_x", "_y") in merged_df.columns:
        merged_df[col.replace("_x", "")] = merged_df[col]  # Use `_x` column and rename
        merged_df.drop(columns=[col, col.replace("_x", "_y")], inplace=True)  # Drop both `_x` and `_y`

# Step 9: Text 열 삭제
merged_df.drop(columns=["Text"], inplace=True)

# Step 10: 최종 결과 저장
merged_df.to_csv(output_csv_path, index=False)

print(f"Merged CSV saved to {output_csv_path}")

output_csv_path = "Filtered_Data_Dict.csv"

required_columns = [
    "FHQSOURCE", "FHQPROV", "FHQMOM", "FHQMOMAD", "FHQDAD", "FHQDADAD",
    "FHQSIB", "ID", "SITEID", "USERDATE", "USERDATE2", "update_stamp",
    "PTSOURCE", "PTGENDER", "PTDOB", "PTDOBYY", "PTHAND", "PTMARRY", "PTEDUCAT",
    "PTWORKHS", "PTWORK", "PTNOTRT", "PTRTYR", "PTHOME", "PTTLANG", "PTPLANG",
    "PTADBEG", "PTCOGBEG", "PTADDX", "PTETHCAT", "PTRACCAT", "PHASE", "RID",
    "VISCODE", "VISCODE2", "VISDATE"
]

# CSV 파일 읽기
data_dict = a_df

# 필터링: FLDNAME이 필요한 컬럼 리스트에 포함된 행만 남기기
filtered_data = data_dict[data_dict['FLDNAME'].isin(required_columns)]

# 중복 제거: FLDNAME 기준으로 첫 번째 행만 유지
filtered_data_unique = filtered_data.drop_duplicates(subset="FLDNAME", keep="first")

# 필터링된 데이터 저장
filtered_data_unique.to_csv(output_csv_path, index=False)

print(f"Filtered CSV saved at {output_csv_path}")


df = pd.read_csv('image_text_temp.csv')

# CSV 읽기
a_df = df
b_df = merged_df
output_csv_path = 'ImgText.csv'
# B에서 Subject와 Date 추출
def extract_subject_and_date(image_path):
    match = re.search(r'(\d+_S_\d+)_(\d{4}-\d{2}-\d{2})', image_path)
    if match:
        subject = match.group(1)
        date = match.group(2)
        return pd.Series([subject, date])
    return pd.Series([None, None])

b_df[['Subject', 'Date']] = b_df['Image Path'].apply(extract_subject_and_date)

# A와 B의 join
merged_df = pd.merge(
    a_df, b_df,
    left_on=['Subject', 'Acq Date'],  # A의 Subject, Acq Date
    right_on=['Subject', 'Date'],    # B에서 추출한 Subject, Date
    how='inner'  # inner join
)

# 결과 저장
merged_df.to_csv(output_csv_path, index=False)
print(f"Joined CSV saved to {output_csv_path}")

df = pd.read_csv('ImgText.csv')

columns_to_keep = [
    "Image Path", "Subject", "Month_to_Visit", "Group", "Sex", "Age", "PTGENDER", "PTETHCAT", "PTRACCAT", "PTMARRY", "FHQMOM", "FHQDAD", "FHQSIB", 
    "PTHAND", "PTWORK", "PTHOME", "PTEDUCAT", "PTCOGBEG", "PTDOBYY", "PTADBEG", "PTRTYR"
]

filtered_df = df[columns_to_keep].copy()

filtered_df['CognitiveDecline_Age'] = filtered_df['PTCOGBEG'] - filtered_df['PTDOBYY']  # 인지 저하 시작 나이
filtered_df['Alzheimer_Age'] = filtered_df['PTADBEG'] - filtered_df['PTDOBYY']          # 알츠하이머 발병 나이
filtered_df['Cognitive_to_Alzheimer'] = filtered_df['PTADBEG'] - filtered_df['PTCOGBEG']  # 인지저하~알츠하이머까지 걸린 시간(년)
filtered_df['Retirement_to_Cognitive'] = filtered_df['PTCOGBEG'] - filtered_df['PTRTYR']  # 은퇴~인지저하까지 걸린 시간(년)

# 범주형 데이터 값 설명 정의
category_descriptions = {
    "Group": {
    "AD": "Alzheimer's Disease",
    "MCI": "Mild Cognitive Impairment",
    "CN": "Cognitive Normal"},
    "Sex": {"M":"Male", "F":"Female"},
    "PTETHCAT": {1: "Hispanic", 2: "Non-Hispanic"},
    "PTRACCAT": {1: "White", 2: "African American", 3: "Asian", 4: "Other"},
    "PTMARRY": {1: "Married", 2: "Single", 3: "Divorced", 4: "Widowed"},
    "FHQMOM": {0: "Mother: No dementia", 1: "Mother: Dementia"},
    "FHQDAD": {0: "Father: No dementia", 1: "Father: Dementia"},
    "FHQSIB": {0: "Sibling: No dementia", 1: "Sibling: Dementia"},
    "PTHAND": {1: "Right-handed", 2: "Left-handed", 3: "Ambidextrous"},
    "PTWORK": {0: "Unemployed", 1: "Employed"},
    "PTHOME": {1: "Living independently", 2: "Living with family", 3: "Living in care facility"}
}


# 범주형 데이터를 Text로 변환 및 이어붙이기
def generate_text(row):
    text_parts = []
    for col, mapping in category_descriptions.items():
        if col in row and not pd.isna(row[col]):
            value = row[col]
            if value in mapping:
                text_parts.append(mapping[value])
    return ", ".join(text_parts)

# Text 컬럼 생성
filtered_df['Text'] = filtered_df.apply(generate_text, axis=1)
output_csv_path = 'CATARI_df.csv'
# 결과 저장
filtered_df[['Image Path', 'Text']].to_csv(output_csv_path, index=False)
print(f"Final CSV with Text column saved to {output_csv_path}")

# 입력 파일 경로
input_csv_path = "ImgText.csv"  # 기존 데이터가 저장된 CSV 파일
output_metadata_path = "metadata.csv"  # Image Path와 수치형 데이터만 포함한 파일

# 읽어오기
df = pd.read_csv(input_csv_path)\

# 2. Image Path와 수치형 데이터 컬럼만 포함한 데이터프레임 생성
numeric_columns = [
    'Age', 'Month_to_Visit', 'PTEDUCAT', 'PTCOGBEG', 'PTADBEG', 'PTRTYR',
    'CognitiveDecline_Age', 'Alzheimer_Age', 'Cognitive_to_Alzheimer', 'Retirement_to_Cognitive'
]

filtered_numeric_columns = ['Image Path']

# 각 수치형 컬럼에 대해 값 검증
for col in numeric_columns:
    if col in df.columns:  # 컬럼 존재 확인
        # 값이 모두 양수이거나 결측치가 없는지 확인
        if (df[col] >= 0).all() and not df[col].isnull().any():
            filtered_numeric_columns.append(col)

# 필터링된 데이터프레임 생성
metadata_df = df[filtered_numeric_columns]

# 저장
metadata_df.to_csv(output_metadata_path, index=False)
print(f"Metadata CSV saved to {output_metadata_path}")

import re

df = pd.read_csv('image_Text.csv')
output_csv_path = "processed_image_Text.csv"  # 출력 CSV 파일 경로

# CSV 읽기
df = pd.read_csv(input_csv_path)

# 나이와 기간 추출 함수
def process_text(row):
    text_parts = row['Text'].split(', ')  # ', ' 기준으로 분할

    # 3번째 요소에서 나이 추출
    age_part = text_parts[2]  # 예: "90 years old"
    age_match = re.search(r'(\d+)\s*years\s*old', age_part)
    age = int(age_match.group(1)) if age_match else None

    # 4번째 요소에서 첫 방문으로부터 몇 달인지 추출
    visit_part = text_parts[3]  # 예: "11 months from first visit"
    months_match = re.search(r'(\d+)\s*months', visit_part)
    months_from_first_visit = int(months_match.group(1)) if months_match else 0  # 첫 방문이면 0으로 처리

    # 나머지 텍스트 다시 합치기 (3,4번째 요소 제외)
    new_text_parts = text_parts[:2] + text_parts[4:]
    new_text = ', '.join(new_text_parts)

    return pd.Series([new_text, age, months_from_first_visit])

# Text 처리 및 새로운 컬럼 생성
df[['Text', 'Age', 'MonthsFromFirstVisit']] = df.apply(process_text, axis=1)

# 결과 저장
df.to_csv(output_csv_path, index=False)
df[['Image Path', 'Text']].to_csv('IMAGE_TEXT.csv', index=False)
print(f"Processed CSV saved to {output_csv_path}")