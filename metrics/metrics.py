import pandas as pd
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
from scipy.linalg import norm
from torchvision import models
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Resize
import os

# FID 계산용 함수
class FIDCalculator:
    def __init__(self):
        self.model = models.inception_v3(pretrained=True, transform_input=False)
        self.model.fc = torch.nn.Identity()
        self.model.eval()

    def calculate_features(self, images):
        with torch.no_grad():
            return self.model(images).cpu().numpy()

    def calculate_fid(self, real_features, generated_features):
        mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)

        diff = mu1 - mu2
        cov_mean, _ = np.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real

        return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * cov_mean)

# 이미지 비교 함수
def calculate_metrics(input_img, output_img, target_img):
    # SSIM
    ssim_val = ssim(target_img, output_img, data_range=output_img.max() - output_img.min())

    # MSE
    mse_val = np.mean((output_img - target_img) ** 2)

    return ssim_val, mse_val

# 이미지 전처리 함수
def preprocess_image(img_path, img_size=299):
    img = imread(img_path)
    if len(img.shape) == 2:  # Gray-scale 이미지를 RGB로 변환
        img = np.stack([img] * 3, axis=-1)
    img_tensor = ToTensor()(img).unsqueeze(0)  # Tensor 변환 및 배치 추가
    img_tensor = Resize((img_size, img_size))(img_tensor)
    return img_tensor

# 메인 함수
def evaluate_images(csv_path, output_csv_path):
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)
    fid_calculator = FIDCalculator()
    
    # 결과 저장용 리스트
    results = []

    # FID 계산용 Feature 저장
    target_features = []
    output_features = []

    for idx, row in df.iterrows():
        input_path, output_path, target_path = row['input'], row['output'], row['target']
        
        # 이미지 읽기
        input_img = imread(input_path)
        output_img = imread(output_path)
        target_img = imread(target_path)

        # SSIM & MSE 계산
        ssim_val, mse_val = calculate_metrics(input_img, output_img, target_img)

        # FID 계산을 위한 Feature 추출
        target_tensor = preprocess_image(target_path)
        output_tensor = preprocess_image(output_path)

        target_features.append(fid_calculator.calculate_features(target_tensor))
        output_features.append(fid_calculator.calculate_features(output_tensor))

        # 결과 저장
        results.append({
            "input_path": input_path,
            "output_path": output_path,
            "target_path": target_path,
            "SSIM": ssim_val,
            "MSE": mse_val
        })

    # FID 계산
    target_features = np.vstack(target_features)
    output_features = np.vstack(output_features)
    fid_val = fid_calculator.calculate_fid(target_features, output_features)

    # 결과를 DataFrame으로 저장
    results_df = pd.DataFrame(results)
    results_df['FID'] = fid_val  # 동일 FID 값 적용

    # 평균값 계산
    mean_values = results_df[['SSIM', 'MSE']].mean().to_dict()
    mean_values['FID'] = fid_val
    print("평균값:", mean_values)

    # 결과 CSV 저장
    results_df.to_csv(output_csv_path, index=False)
    print(f"결과 저장 완료: {output_csv_path}")