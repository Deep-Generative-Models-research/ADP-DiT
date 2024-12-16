import os
from glob import glob
import ants
from multiprocessing import Pool

# NIfTI 데이터 경로
nifti_dir = "i"
output_base_dir = ""
error_log_file = "error.txt"
os.makedirs(output_base_dir, exist_ok=True)

# 오류 로그 초기화
with open(error_log_file, "w") as f:
    f.write("")

# 병렬로 처리할 함수
def process_subject(subject_dir):
    subject = os.path.basename(subject_dir)
    registered_subject_dir = os.path.join(output_base_dir, subject)

    # 이미 처리된 subject는 건너뛰기
    if os.path.exists(registered_subject_dir):
        print(f"Subject {subject} already processed. Skipping...")
        return

    print(f"Processing subject: {subject}")

    try:
        # Timestamp별 파일 그룹화
        timestamps = {}
        for filepath in glob(os.path.join(subject_dir, "*.nii.gz")):
            filename = os.path.basename(filepath)
            timestamp = filename.split("_")[1]  # 파일명에서 timestamp 추출
            if timestamp not in timestamps:
                timestamps[timestamp] = []
            timestamps[timestamp].append(filepath)

        # 각 Timestamp 그룹 처리
        for timestamp, files in timestamps.items():
            print(f"  Processing timestamp: {timestamp}")

            # 정렬된 파일 저장 디렉토리
            os.makedirs(registered_subject_dir, exist_ok=True)

            # 기준 파일 복사
            reference = sorted(files)[0]  # 가장 먼저 발견된 파일을 기준으로 설정
            reference_output_path = os.path.join(registered_subject_dir, os.path.basename(reference))
            print(f"    Copying reference file: {os.path.basename(reference)}")
            ants.image_write(ants.image_read(reference), reference_output_path)

            # 나머지 파일 정합
            for moving in files:
                if moving != reference:
                    output_file = os.path.join(registered_subject_dir, os.path.basename(moving))

                    # 정렬 작업 수행
                    fixed = ants.image_read(reference)
                    moving_image = ants.image_read(moving)

                    print(f"    Registering {os.path.basename(moving)} to {os.path.basename(reference)}...")
                    registration = ants.registration(fixed=fixed, moving=moving_image, type_of_transform="SyN")
                    aligned_image = registration["warpedmovout"]

                    # 정렬된 파일 저장
                    ants.image_write(aligned_image, output_file)
                    print(f"    Final aligned image saved to: {output_file}")

    except Exception as e:
        # 오류 발생 시 error.txt에 기록
        print(f"Error processing subject {subject}: {e}")
        with open(error_log_file, "a") as f:
            f.write(f"{subject}: {e}\n")
        return

# 병렬 처리 설정
if __name__ == "__main__":
    nifti_dirs = sorted(glob(os.path.join(nifti_dir, "*")))
    nifti_dirs = [d for d in nifti_dirs if os.path.isdir(d)]  # 디렉토리만 필터링

    # 병렬 처리 - Pool 사용
    with Pool(processes=8) as pool:  # 시스템에 따라 적절히 프로세스 개수 조정
        pool.map(process_subject, nifti_dirs)

    print("All alignments completed!")