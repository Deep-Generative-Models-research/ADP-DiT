#!/bin/bash

input_dir=""      # DICOM 파일 경로
output_dir=""    # 변환된 NIfTI 파일 저장 경로

# 출력 디렉토리 생성
mkdir -p "$output_dir"

# DICOM 디렉토리 순회 및 처리할 디렉토리 목록 생성
find "$input_dir" -mindepth 4 -type d | while read -r dir; do
    # DICOM 파일이 있는 디렉토리만 목록에 추가
    if ls "$dir"/*.dcm >/dev/null 2>&1; then
        echo "$dir"
    fi
done > dir_list.txt

# 병렬 실행 함수 정의
process_dir() {
    dir="$1"
    # 경로를 분석하여 Subject ID와 Timestamp 추출
    rel_path=${dir#"$input_dir/"}  # 입력 경로의 상대 경로
    subject=$(echo "$rel_path" | cut -d'/' -f1)   # 첫 번째 디렉토리를 Subject ID로
    timestamp=$(echo "$rel_path" | cut -d'/' -f3) # 세 번째 디렉토리를 Timestamp로 간주

    # 출력 디렉토리와 파일 이름 설정
    subject_output_dir="$output_dir/$subject"
    mkdir -p "$subject_output_dir"
    output_file="$subject_output_dir/${subject}_${timestamp}.nii.gz"

    # dcm2niix 실행
    echo "Processing Subject: $subject, Timestamp: $timestamp"
    dcm2niix -z y -o "$subject_output_dir" -f "${subject}_${timestamp}" "$dir"
}

export -f process_dir
export input_dir output_dir

# GNU parallel을 사용하여 병렬 실행
parallel --joblog joblog.txt -j 20 process_dir :::: dir_list.txt

echo "All conversions completed!"