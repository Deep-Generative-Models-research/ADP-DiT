import os
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import img_as_float
import matplotlib.pyplot as plt
import numpy as np

# 이미지 로드 및 Grayscale 변환 함수
def load_and_preprocess_gray(image_path):
    img = imread(image_path)
    if len(img.shape) == 3:  # RGB 이미지
        img = rgb2gray(img)  # Grayscale 변환
    return img_as_float(img).astype(np.float32)

# 상위 1% 추출 및 이미지 출력
def display_top_1_percent_images(csv_files):
    for file in csv_files:
        print(f"Processing: {file}")
        df = pd.read_csv(file)

        # 상위 1% 필터링
        fid_threshold = df["FID"].quantile(0.999)
        mse_threshold = df["MSE"].quantile(0.999)
        ssim_threshold = df["SSIM"].quantile(0.999)

        top_rows = df[
            (df["FID"] >= fid_threshold) |
            (df["MSE"] >= mse_threshold) |
            (df["SSIM"] >= ssim_threshold)
        ]
        top_rows = df.nlargest(10, "FID")

        # 이미지 출력
        for _, row in top_rows.iterrows():
            print(row["input_image_path"], row["target_image_path"], sep='\n')
            input_img = load_and_preprocess_gray(row["input_image_path"])
            output_img = load_and_preprocess_gray(row["output_image_path"])
            target_img = load_and_preprocess_gray(row["target_image_path"])
            difference_map = np.abs(output_img - target_img)

            # 출력
            fig, ax = plt.subplots(1, 4, figsize=(16, 4))
            fig.suptitle(f"Input: {row['input_image_path']}\nFID: {row['FID']:.2f}, MSE: {row['MSE']:.2f}, SSIM: {row['SSIM']:.4f}")

            ax[0].imshow(input_img, cmap="gray")
            ax[0].axis("off")
            ax[0].set_title("Input")

            ax[1].imshow(output_img, cmap="gray")
            ax[1].axis("off")
            ax[1].set_title("Output")

            ax[2].imshow(target_img, cmap="gray")
            ax[2].axis("off")
            ax[2].set_title("Target")

            ax[3].imshow(difference_map, cmap="gray")
            ax[3].axis("off")
            ax[3].set_title("Difference Map")

            plt.tight_layout()
            plt.show()

def load_csv_files(file_path="csv_files.txt"):
    with open(file_path, "r") as f:
        csv_files = [line.strip() for line in f if line.strip()]  # 빈 줄 제외
    return csv_files

# CSV 파일 리스트 로드
csv_files = load_csv_files()

display_top_1_percent_images(csv_files)