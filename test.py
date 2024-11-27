import os
import pandas as pd
import shutil

# 경로 설정
csv_file_path = '/workspace/dataset/AD/csvfile/image_text.csv'
output_csv_path = '/workspace/dataset/AD/csvfile/image_text_150.csv'
images_dir = '/workspace/dataset/AD/images'
output_images_dir = '/workspace/dataset/AD/images_150'

# CSV 파일 읽기
if not os.path.exists(csv_file_path):
    raise FileNotFoundError(f"CSV 파일이 존재하지 않습니다: {csv_file_path}")

# 데이터 로드
image_text_df = pd.read_csv(csv_file_path)

# Z-coordinate 150인 데이터 필터링
filtered_df = image_text_df[image_text_df['text_en'].str.contains('Z-coordinate 150', na=False)]

# 필터링된 데이터를 새 CSV 파일로 저장
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
filtered_df.to_csv(output_csv_path, index=False)
print(f"Filtered CSV 저장 완료: {output_csv_path}")

# 이미지 복사
if not os.path.exists(images_dir):
    raise FileNotFoundError(f"이미지 폴더가 존재하지 않습니다: {images_dir}")

os.makedirs(output_images_dir, exist_ok=True)

for image_path in filtered_df['image_path']:
    src_image_path = os.path.join(images_dir, os.path.basename(image_path))
    dest_image_path = os.path.join(output_images_dir, os.path.basename(image_path))

    if os.path.exists(src_image_path):
        shutil.copy2(src_image_path, dest_image_path)
        print(f"이미지 복사 완료: {src_image_path} -> {dest_image_path}")
    else:
        print(f"이미지 파일이 존재하지 않습니다: {src_image_path}")

print(f"모든 작업이 완료되었습니다. 필터링된 이미지가 저장된 폴더: {output_images_dir}")