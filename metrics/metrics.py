# numactl --interleave=all python metrics/metrics.py
import os
# numactl 환경에서 과도한 스레드 생성 방지를 위해 환경 변수 설정
os.environ["OMP_NUM_THREADS"] = "128"
os.environ["MKL_NUM_THREADS"] = "128"

import pandas as pd
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.io import imread
from skimage.transform import resize
from skimage.color import gray2rgb
from scipy.linalg import sqrtm
from torchvision import models
import torch
from torchvision.transforms import ToTensor, Compose, Normalize
import concurrent.futures
import threading

# 쓰레드 로컬 변수 (각 쓰레드마다 FIDCalculator 인스턴스를 생성하기 위함)
thread_local = threading.local()

# FID 계산용 클래스
class FIDCalculator:
    def __init__(self):
        # Inception v3 모델 초기화 (ImageNet 가중치 사용)
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.model = self.model.float()  # 모델 dtype을 float32로 설정

    def calculate_features(self, images):
        print(f"Image tensor dtype before model: {images.dtype}")
        images = images.float()  # 입력 텐서 dtype을 float32로 설정
        with torch.no_grad():
            features = self.model(images).cpu().numpy()
        print(f"Features dtype: {features.dtype}")
        return features

    def calculate_fid(self, real_features, generated_features):
        mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
        diff = mu1 - mu2
        cov_mean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real
        return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * cov_mean)

# 이미지 비교 함수 (SSIM, MSE & PSNR)
def calculate_metrics_ssim_mse(output_img, target_img, resize_dim=(256, 256)):
    # 그레이스케일 이미지는 RGB로 변환
    if output_img.ndim == 2:
        output_img = gray2rgb(output_img)
    if target_img.ndim == 2:
        target_img = gray2rgb(target_img)
    # 이미지 dtype을 float32로 변경
    output_img = output_img.astype(np.float32)
    target_img = target_img.astype(np.float32)
    print(f"Output image shape: {output_img.shape}, dtype: {output_img.dtype}")
    print(f"Target image shape: {target_img.shape}, dtype: {target_img.dtype}")
    # 채널 축 설정
    channel_axis = -1 if output_img.ndim == 3 else None
    # 작은 이미지에 대한 win_size 조정
    min_dim = min(output_img.shape[0], output_img.shape[1])
    if min_dim < 7:
        win_size = min_dim
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            win_size = 3
    else:
        win_size = 7
    # SSIM 계산
    try:
        ssim_val = ssim(
            target_img,
            output_img,
            data_range=output_img.max() - output_img.min(),
            channel_axis=channel_axis,
            win_size=win_size
        )
    except TypeError:
        ssim_val = ssim(
            target_img,
            output_img,
            data_range=output_img.max() - output_img.min(),
            multichannel=(channel_axis is not None)
        )
    mse_val = np.mean((output_img - target_img) ** 2)
    psnr_val = psnr(target_img, output_img, data_range=output_img.max() - output_img.min())
    return ssim_val, mse_val, psnr_val

# 이미지 전처리 함수 for FID (정규화 추가)
def preprocess_image_fid(img_path, img_size=299):
    from skimage.io import imread
    from skimage.transform import resize
    from skimage.color import gray2rgb
    img = imread(img_path)
    if img.ndim == 2:
        img = gray2rgb(img)
    img = resize(img, (img_size, img_size), anti_aliasing=True)
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

# 각 행을 처리하는 워커 함수
def process_row(idx, row, original_base_path, new_base_path):
    try:
        input_path_original = row['input_image_path']
        output_path_original = row['output_image_path']
        target_path_original = row['target_image_path']
        prompt = row['prompt']

        if os.path.isabs(input_path_original):
            input_path = input_path_original.replace(original_base_path, new_base_path)
        else:
            input_path = os.path.join(new_base_path, input_path_original)
        if os.path.isabs(output_path_original):
            output_path = output_path_original.replace(original_base_path, new_base_path)
        else:
            output_path = os.path.join(new_base_path, output_path_original)
        if os.path.isabs(target_path_original):
            target_path = target_path_original.replace(original_base_path, new_base_path)
        else:
            target_path = os.path.join(new_base_path, target_path_original)
        input_path = os.path.normpath(input_path)
        output_path = os.path.normpath(output_path)
        target_path = os.path.normpath(target_path)
        print(f"\nProcessing row {idx}:")
        print(f"Input path: {input_path}")
        print(f"Output path: {output_path}")
        print(f"Target path: {target_path}")
        print(f"Prompt: {prompt}")
        if (not os.path.exists(input_path) or not os.path.exists(output_path) or not os.path.exists(target_path)):
            print(f"Row {idx}: 이미지 파일이 누락되어 건너뜁니다.")
            return None
        # 이미지 읽기
        output_img = imread(output_path)
        target_img = imread(target_path)
        ssim_val, mse_val, psnr_val = calculate_metrics_ssim_mse(output_img, target_img)
        target_tensor = preprocess_image_fid(target_path)
        output_tensor = preprocess_image_fid(output_path)
        # 각 쓰레드마다 FIDCalculator 인스턴스를 생성 (한번 생성되면 재사용)
        if not hasattr(thread_local, 'fid_calculator'):
            thread_local.fid_calculator = FIDCalculator()
        fid_calc = thread_local.fid_calculator
        target_feature = fid_calc.calculate_features(target_tensor)
        output_feature = fid_calc.calculate_features(output_tensor)
        return {
            "input_path": input_path,
            "output_path": output_path,
            "target_path": target_path,
            "prompt": prompt,
            "SSIM": ssim_val,
            "MSE": mse_val,
            "PSNR": psnr_val,
            "target_feature": target_feature,
            "output_feature": output_feature
        }
    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        return None

def evaluate_images():
    """
    /data1/ADG-DiT/results/XXX/results.csv 에서 데이터를 읽고,
    (동일 input_path 중) SSIM이 가장 큰 결과만 골라 FID, 메트릭을 계산합니다.
    최종 결과는 /data1/ADG-DiT/results/evaluation/{dataset}/{dataset}_best_results.csv 로 저장하며,
    평균값은 /data1/ADG-DiT/results/evaluation/evaluate.csv 에 누적 기록합니다.
    """
    evaluations = [
        {"dataset": "ADtoAD", "input_csv": "/data1/ADG-DiT/results/ADtoAD/results.csv"},
        {"dataset": "MCtoAD", "input_csv": "/data1/ADG-DiT/results/MCtoAD/results.csv"},
        {"dataset": "MCtoMC", "input_csv": "/data1/ADG-DiT/results/MCtoMC/results.csv"},
        {"dataset": "CNtoMC", "input_csv": "/data1/ADG-DiT/results/CNtoMC/results.csv"},
        {"dataset": "CNtoCN", "input_csv": "/data1/ADG-DiT/results/CNtoCN/results.csv"}
    ]
    evaluation_root = "/data1/ADG-DiT/results/evaluation"
    summary_csv_path = os.path.join(evaluation_root, "evaluate.csv")
    original_base_path = '/workspace'
    new_base_path = '/data1/ADG-DiT'
    for eval_ in evaluations:
        dataset = eval_["dataset"]
        input_csv = eval_["input_csv"]
        print(f"\n===== Processing Dataset: {dataset} =====")
        print(f"Input CSV: {input_csv}")
        required_columns = ['input_image_path', 'output_image_path', 'target_image_path', 'prompt']
        try:
            df = pd.read_csv(input_csv, usecols=required_columns, skipinitialspace=True)
        except Exception as e:
            print(f"CSV 파일 읽기 오류 ({input_csv}): {e}")
            continue
        df.columns = df.columns.str.strip()
        print("CSV 열 이름:", df.columns.tolist())
        print("CSV 데이터 샘플:")
        print(df.head())
        all_results = []
        # 병렬 처리: 각 행을 처리하는 작업을 쓰레드풀에 제출
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for idx, row in df.iterrows():
                futures.append(executor.submit(process_row, idx, row, original_base_path, new_base_path))
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res is not None:
                    all_results.append(res)
        if not all_results:
            print(f"{dataset}: 유효한 결과가 없어 FID 계산을 건너뜁니다.")
            continue
        all_results_df = pd.DataFrame(all_results)
        best_results_df = (
            all_results_df.sort_values(by="SSIM", ascending=False)
                          .groupby("input_path", as_index=False)
                          .head(1)
        )
        best_target_features = np.vstack(best_results_df["target_feature"])
        best_output_features = np.vstack(best_results_df["output_feature"])
        # FID 계산은 각 데이터셋에 대해 단일 값으로 계산
        fid_calc_main = FIDCalculator()
        fid_val = fid_calc_main.calculate_fid(best_target_features, best_output_features)
        best_results_df["FID"] = fid_val
        mean_values = {
            "SSIM": best_results_df["SSIM"].mean(),
            "MSE":  best_results_df["MSE"].mean(),
            "PSNR": best_results_df["PSNR"].mean(),
            "FID":  fid_val,
            "Dataset": dataset
        }
        print("\n베스트 결과 기준 평균값:", mean_values)
        dataset_dir = os.path.join(evaluation_root, dataset)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            print(f"결과 저장 디렉터리 생성: {dataset_dir}")
        best_csv_path = os.path.join(dataset_dir, f"{dataset}_best_results.csv")
        columns_to_save = [
            "input_path", "output_path", "target_path", "prompt",
            "SSIM", "MSE", "PSNR", "FID"
        ]
        best_save_df = best_results_df[columns_to_save].copy()
        if os.path.exists(best_csv_path):
            best_save_df.to_csv(best_csv_path, mode='a', index=False, header=False)
            print(f"베스트 결과가 기존 CSV 파일에 추가되었습니다: {best_csv_path}")
        else:
            best_save_df.to_csv(best_csv_path, mode='w', index=False, header=True)
            print(f"베스트 결과가 새 CSV 파일로 저장되었습니다: {best_csv_path}")
        summary_df = pd.DataFrame([{
            "Dataset": mean_values["Dataset"],
            "SSIM": mean_values["SSIM"],
            "MSE": mean_values["MSE"],
            "PSNR": mean_values["PSNR"],
            "FID": mean_values["FID"]
        }])
        if not os.path.exists(evaluation_root):
            os.makedirs(evaluation_root)
        if os.path.exists(summary_csv_path):
            cols = ["Dataset"] + [c for c in summary_df.columns if c != "Dataset"]
            summary_df = summary_df[cols]
            summary_df.to_csv(summary_csv_path, mode='a', index=False, header=False)
            print(f"평균값이 기존 평가 CSV 파일에 추가되었습니다: {summary_csv_path}")
        else:
            cols = ["Dataset"] + [c for c in summary_df.columns if c != "Dataset"]
            summary_df = summary_df[cols]
            summary_df.to_csv(summary_csv_path, mode='w', index=False, header=True)
            print(f"평균값이 새 평가 CSV 파일로 저장되었습니다: {summary_csv_path}")

if __name__ == "__main__":
    evaluate_images()
