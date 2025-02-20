# -*- coding: utf-8 -*-
import csv
import re
import os
import numpy as np  # 추가
from datetime import datetime
import subprocess
from concurrent.futures import ProcessPoolExecutor

# Constants
CSV_FILE = "/workspace/dataset/AD3_meta/csvfile/image_text.csv"
IMAGE_ROOT = "/workspace/"

# 변환 쌍에 따른 폴더 매핑
CONVERSION_FOLDER_MAP = {
    ("Cognitive Normal", "Cognitive Normal"): "CNtoCN",
    ("Cognitive Normal", "Mild Cognitive Impairment"): "CNtoMC",
    ("Mild Cognitive Impairment", "Mild Cognitive Impairment"): "MCtoMC",
    ("Mild Cognitive Impairment", "Alzheimer's Disease"): "MCtoAD",
    ("Alzheimer's Disease", "Alzheimer's Disease"): "ADtoAD"
}

ALLOWED_PAIRS = {
    ("Cognitive Normal", "Cognitive Normal"),
    ("Cognitive Normal", "Mild Cognitive Impairment"),
    ("Mild Cognitive Impairment", "Mild Cognitive Impairment"),
    ("Mild Cognitive Impairment", "Alzheimer's Disease"),
    ("Alzheimer's Disease", "Alzheimer's Disease")
}

# 모든 변환된 month 차이를 저장할 리스트 (전역 변수)
month_differences = []

# 각 변환 쌍별 month 차이를 저장할 딕셔너리
conversion_month_differences = {pair: [] for pair in ALLOWED_PAIRS}


def calculate_month_difference(date1, date2):
    """두 날짜 간의 차이를 개월 단위로 계산"""
    return abs((date2.year - date1.year) * 12 + date2.month - date1.month)


def extract_images_and_prompts(csv_file, condition):
    """
    CSV 파일에서 모든 이미지 경로와 프롬프트, 그리고 'first visit'인 항목을 추출합니다.
    """
    cond2 = "first visit"

    if not os.path.isfile(csv_file):
        print(f"CSV file not found: {csv_file}")
        return [], []

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_image_prompts = []
        first_visit_image_prompts = []
        for row in reader:
            image_path = row['image_path'].strip()
            text_en = row['text_en'].strip()
            abs_path = os.path.normpath(os.path.join(IMAGE_ROOT, image_path))
            all_image_prompts.append((abs_path, text_en))
            if condition in text_en and cond2 in text_en:
                first_visit_image_prompts.append((abs_path, text_en))
    return all_image_prompts, first_visit_image_prompts


def extract_latest_image_and_prompt(subject_id, all_image_prompts):
    """주어진 subject_id에 대해 최신 날짜의 이미지 경로와 프롬프트를 추출합니다."""
    latest_date = None
    latest_image_path = None
    latest_prompt = None

    for row in all_image_prompts:
        ipath = row[0]
        prompt = row[1]
        ipath_abs = os.path.abspath(os.path.join(IMAGE_ROOT, ipath))

        if f'_S_{subject_id}_' in ipath:
            date_match = re.search(r'_(\d{4}-\d{2}-\d{2})_', ipath)
            if date_match:
                date_str = date_match.group(1)
                try:
                    date_val = datetime.strptime(date_str, '%Y-%m-%d')
                    if latest_date is None or date_val > latest_date:
                        latest_date = date_val
                        latest_image_path = ipath_abs
                        latest_prompt = prompt
                except Exception as e:
                    print(f"Date parsing failed for {date_str}: {e}")
                    continue

    if latest_image_path:
        print(f"Latest image for subject_id {subject_id}: {latest_image_path} (Date: {latest_date.strftime('%Y-%m-%d')})")
    else:
        print(f"No images found for subject_id {subject_id}")

    return latest_image_path, latest_prompt, latest_date


def extract_target_path(image_path, all_image_prompts):
    """
    이미지 경로에서 subject_id를 추출하고, 해당 subject의 최신 이미지와 프롬프트를 반환합니다.
    또한 input 이미지와 target 이미지 간의 month 차이를 계산합니다.
    """
    subject_id_match = re.search(r'\d{3}_S_(\d{4})', image_path)
    if subject_id_match:
        subject_id = subject_id_match.group(1)
        print(f"Extracted subject_id: {subject_id} from image_path: {image_path}")

        latest_image_path, latest_prompt, latest_date = extract_latest_image_and_prompt(subject_id, all_image_prompts)

        # 현재 input 이미지의 날짜 추출
        date_match = re.search(r'_(\d{4}-\d{2}-\d{2})_', image_path)
        if date_match:
            input_date = datetime.strptime(date_match.group(1), '%Y-%m-%d')
        else:
            print(f"Could not extract date from image_path: {image_path}")
            return None, None, None

        # Month 차이 계산 및 반환
        if latest_date:
            month_diff = calculate_month_difference(input_date, latest_date)
            print(f"Month difference between input ({input_date.strftime('%Y-%m-%d')}) and target ({latest_date.strftime('%Y-%m-%d')}): {month_diff} months")
        else:
            month_diff = None

        return latest_image_path, latest_prompt, month_diff
    else:
        print(f"Could not extract subject_id from image_path: {image_path}")
        return None, None, None


# get_condition 함수 (텍스트에서 조건을 추출)
def get_condition(text):
    if "Alzheimer's Disease" in text:
        return "Alzheimer's Disease"
    elif "Mild Cognitive Impairment" in text:
        return "Mild Cognitive Impairment"
    elif "Cognitive Normal" in text:
        return "Cognitive Normal"
    else:
        return None


# run_experiment 함수는 기존 코드에 따라 정의되어 있다고 가정합니다.
def run_experiment(image_path, prompt, target_path, target_prompt, conversion_pair):
    # 이 함수는 기존의 인퍼런스 실행 코드를 포함합니다.
    # 실제 코드 내용은 생략합니다.
    print(f"Running experiment for conversion: {conversion_pair}")
    # 예시: subprocess.run([...])
    pass

def run_single_inference(command, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # GPU ID 설정
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(command)}")
        print(f"Error Output: {e.stderr}")
        raise e

def main():
    cond1_list = [
        "Cognitive Normal",
        "Mild Cognitive Impairment",
        "Alzheimer's Disease"
    ]

    for condition in cond1_list:
        print(f"Starting experiments for condition: {condition}")
        all_image_prompts, first_visit_image_prompts = extract_images_and_prompts(CSV_FILE, condition)
        if not first_visit_image_prompts:
            print(f"No matching 'first visit' data found for condition: {condition}")
            continue

        for image_path, prompt in first_visit_image_prompts:
            target_path, target_prompt, month_diff = extract_target_path(image_path, all_image_prompts)

            # input image와 target이 동일한 경우 건너뜁니다.
            if image_path == target_path:
                print(f"Skipping inference for most recent first visit image: {image_path}")
                continue

            # input condition과 target condition을 비교
            input_condition = condition
            target_condition = get_condition(target_prompt)
            if target_condition is None:
                print(f"Skipping image: {image_path} due to unknown target condition.")
                continue

            conversion_pair = (input_condition, target_condition)
            if conversion_pair not in ALLOWED_PAIRS:
                print(f"Skipping inference for subject {image_path}: conversion {conversion_pair} is not allowed.")
                continue

            # 변환 쌍별 month 차이가 존재하면 해당 딕셔너리에 추가
            if month_diff is not None:
                conversion_month_differences[conversion_pair].append(month_diff)
                month_differences.append(month_diff)

            # 실행
            run_experiment(image_path, prompt, target_path, target_prompt, conversion_pair)

    # 전체 변환에 대한 통계 출력 (Mean ± Std Dev)
    if month_differences:
        overall_mean = np.mean(month_differences)
        overall_std = np.std(month_differences)
        print(f"\n===== Overall Month Difference Statistics =====")
        print(f"Mean ± Std Dev: {overall_mean:.2f} ± {overall_std:.2f} months")
    else:
        print("No month differences were recorded.")

    # 각 변환 쌍별 통계 출력 (Mean ± Std Dev)
    print("\n===== Conversion Pair Month Difference Statistics =====")
    for conv_pair, diffs in conversion_month_differences.items():
        if diffs:
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs)
            print(f"Conversion {conv_pair}: {mean_diff:.2f} ± {std_diff:.2f} months (n = {len(diffs)})")
        else:
            print(f"Conversion {conv_pair}: No data recorded.")

    print("All experiments completed.")

if __name__ == "__main__":
    main()
