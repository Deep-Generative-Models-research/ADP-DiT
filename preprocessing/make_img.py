import os
import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image, ImageOps


# Parameters
nifti_dir = 'z_adjusted'  # NIfTI 파일들의 루트 디렉토리
output_dir = 'dataset/images'  # PNG 이미지 저장 경로
heights = [170] #list(range(70, 161, 10))  # 추출할 높이 (100~200, 10씩 증가)
csv_path = 'Final_A_with_Descriptive_Text.csv'  # 원본 CSV 경로
temp_csv_path = 'dataset/csvfile/image_text_temp.csv'  # 생성될 CSV 파일 경로
final_csv_path = 'dataset/csvfile/image_text.csv'  # 생성될 CSV 파일 경로


def save_semi_stretched_mirrored_image(slice_2d, output_path, size=(256, 256)):
    """
    슬라이스를 세로로 늘리고, 가로로 패딩을 추가하고, 좌우 반전한 PNG 이미지를 저장.

    Args:
        slice_2d (ndarray): 2D 슬라이스 데이터.
        output_path (str): 저장 경로.
        size (tuple): 최종 이미지 크기 (기본값: 256x256).
    """
    # Normalize data to 0-255
    slice_2d = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d)) * 255
    slice_2d = slice_2d.astype(np.uint8)

    # Convert to PIL Image
    pil_image = Image.fromarray(slice_2d)

    # 좌우 반전
    mirrored_image = ImageOps.mirror(pil_image)

    # 세로 방향 늘리기
    stretched_height = size[1]
    aspect_ratio = slice_2d.shape[1] / slice_2d.shape[0]
    stretched_width = int(stretched_height * aspect_ratio)
    stretched_image = mirrored_image.resize((stretched_width, stretched_height), Image.Resampling.LANCZOS)

    # 가로 방향 패딩 추가
    padded_image = ImageOps.pad(stretched_image, size, method=Image.Resampling.LANCZOS, color=(0))

    # 저장
    padded_image.save(output_path)
    print(f"Saved image: {output_path}")


def nifti_to_png_and_csv(nifti_dir, output_dir, heights, csv_path, temp_csv_path):
    """
    NIfTI 데이터를 PNG 이미지로 변환하고, 지정된 높이에서 2D 슬라이스를 추출 및 저장.

    Args:
        nifti_dir (str): NIfTI 파일이 저장된 디렉토리.
        output_dir (str): PNG 이미지를 저장할 디렉토리.
        heights (list): 추출할 슬라이스 높이 (Z-coordinates).
        csv_path (str): 텍스트 매핑용 CSV 파일 경로.
        temp_csv_path (str): 생성할 임시 CSV 파일 경로.
    """
    # Step 1: Output 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 2: Final CSV 읽기
    final_csv = pd.read_csv(csv_path)
    final_csv['Acq Date'] = pd.to_datetime(final_csv['Acq Date']).dt.strftime('%Y-%m-%d')

    image_data = []

    # Step 3: NIfTI 파일 처리
    for subject_folder in os.listdir(nifti_dir):
        subject_path = os.path.join(nifti_dir, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        for nifti_file in os.listdir(subject_path):
            if nifti_file.endswith('.nii.gz'):
                # 파일명에서 Subject 및 Timestamp 추출
                parts = nifti_file.replace('.nii.gz', '').split('_')
                subject_id = '_'.join(parts[:3])  # 002_S_1018
                timestamp = '_'.join(parts[3:])  # 2006-11-29_10_00_05.0
                nifti_path = os.path.join(subject_path, nifti_file)

                # NIfTI 데이터를 NumPy 배열로 변환 및 정렬
                img = nib.load(nifti_path).get_fdata()
                oriented_data = np.flip(np.transpose(img, (1, 0, 2)), axis=0)  # 기본 orientation 수정

                # Step 4: 지정된 높이에서 2D 슬라이스 추출 및 처리
                for z in heights:
                    if z >= oriented_data.shape[2]:
                        print(f"Warning: Height {z} out of bounds for {nifti_file}")
                        continue

                    slice_2d = oriented_data[:, :, z]
                    image_name = f"{subject_id}_{timestamp}_{z}.png"
                    image_path = os.path.join(output_dir, image_name)

                    # 새 방식으로 이미지 저장 (세로 늘리기 + 가로 패딩 + 좌우 반전)
                    save_semi_stretched_mirrored_image(slice_2d, image_path, size=(256, 256))

                    # 텍스트 매핑
                    text_row = final_csv[(final_csv['Subject'] == subject_id) &
                                         (final_csv['Acq Date'] == timestamp.split('_')[0])]
                    if not text_row.empty:
                        text = text_row.iloc[0]['Text']# + f", Z-coordinate {z}"
                        # 강제로 "./dataset"으로 시작하도록 경로 설정
                        relative_path = os.path.join("./dataset", os.path.relpath(image_path, start=output_dir))
                        image_data.append({'Image Path': relative_path, 'Subject': subject_id, 
                                           'Timestamp': timestamp, 'Z Index': z, 'Text': text})

    # 임시 CSV 저장
    temp_df = pd.DataFrame(image_data)
    temp_df.to_csv(temp_csv_path, index=False)
    print(f"Temporary dataset CSV saved at {temp_csv_path}")


def create_and_sort_csv(image_dir, temp_csv_path, final_csv_path):
    """
    PNG 이미지 경로와 텍스트 데이터를 매핑하여 CSV 파일을 생성하고 정렬.

    Args:
        image_dir (str): PNG 이미지가 저장된 루트 디렉토리.
        temp_csv_path (str): 임시 CSV 파일 경로.
        final_csv_path (str): 생성할 최종 CSV 파일 경로.
    """
    # Final CSV 읽기
    temp_csv = pd.read_csv(temp_csv_path)

    # Z Index를 정렬 및 Image Name 기준 추가 정렬
    temp_csv['Image Name'] = temp_csv['Image Path'].apply(lambda x: os.path.basename(x).split('_')[0])
    sorted_df = temp_csv.sort_values(by=['Image Name']).reset_index(drop=True)

    # 최종 CSV 저장
    sorted_df[['Image Path', 'Text']].to_csv(final_csv_path, index=False)
    print(f"Final sorted dataset CSV saved at {final_csv_path}")


if __name__ == "__main__":
    # Step 1: NIfTI to PNG and Temporary CSV
    nifti_to_png_and_csv(nifti_dir, output_dir, heights, csv_path, temp_csv_path)

    # Step 2: Create and Sort Final CSV
    create_and_sort_csv(output_dir, temp_csv_path, final_csv_path)