import pandas as pd

# Parameters
csv_path = 'Final_A_with_Descriptive_Text.csv'  # 원본 CSV 경로
temp_csv_path = 'image_text_temp.csv'  # 생성될 CSV 파일 경로
subject_counts = {"CN": 25, "MCI": 25, "AD": 50}  # 그룹별 선택 수

def select_subjects_by_timestamps(csv_path, subject_counts):
    """
    CSV에서 그룹별로 Timestamp가 많은 Subject를 선택.

    Args:
        csv_path (str): 원본 CSV 파일 경로.
        subject_counts (dict): 그룹별 선택할 Subject 수.

    Returns:
        pd.DataFrame: 선택된 Subject 데이터프레임.
    """
    # 원본 CSV 읽기
    csv_data = pd.read_csv(csv_path)

    # Acq Date 형식 변환
    csv_data['Acq Date'] = pd.to_datetime(csv_data['Acq Date'], format='%Y/%m/%d').dt.strftime('%Y-%m-%d')

    # 그룹별 Subject 선택
    selected_subjects = []
    for group, count in subject_counts.items():
        group_data = csv_data[csv_data['Group'] == group]
        if group_data.empty:
            print(f"Warning: No subjects found for group {group}")
            continue

        # Timestamp 개수 기준으로 Subject 정렬
        group_data['Timestamp Count'] = group_data.groupby('Subject')['Acq Date'].transform('count')
        group_sorted = group_data.sort_values(by=['Subject', 'Timestamp Count', 'Acq Date'], ascending=[True, False, True])
        selected_subjects.append(group_sorted.head(count * len(group_data['Acq Date'].unique())))

    return pd.concat(selected_subjects)


if __name__ == "__main__":
    # Step 1: Subject 선택
    selected_subjects_df = select_subjects_by_timestamps(csv_path, subject_counts)

    # Step 2: Temp CSV 저장
    selected_subjects_df.to_csv(temp_csv_path, index=False)
    print(f"Temporary dataset CSV saved at {temp_csv_path}")