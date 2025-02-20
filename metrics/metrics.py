import pandas as pd
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.io import imread
from skimage.transform import resize
from skimage.color import gray2rgb
from scipy.linalg import sqrtm
from torchvision import models
import torch
from torchvision.transforms import ToTensor, Resize

# FID 계산용 클래스
class FIDCalculator:
    def __init__(self):
        # Inception v3 모델 초기화
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.model = self.model.float()  # 모델의 dtype을 float32로 설정

    def calculate_features(self, images):
        print(f"Image tensor dtype before model: {images.dtype}")
        images = images.float()  # 입력 텐서의 dtype을 float32로 설정
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
    # Convert grayscale images to RGB
    if output_img.ndim == 2:
        output_img = gray2rgb(output_img)
    if target_img.ndim == 2:
        target_img = gray2rgb(target_img)

    # Ensure images are in float32 format
    output_img = output_img.astype(np.float32)
    target_img = target_img.astype(np.float32)

    # Print image shapes and data types
    print(f"Output image shape: {output_img.shape}, dtype: {output_img.dtype}")
    print(f"Target image shape: {target_img.shape}, dtype: {target_img.dtype}")

    # Determine channel_axis
    if output_img.ndim == 3:
        channel_axis = -1  # Last axis
    else:
        channel_axis = None

    # Adjust win_size if necessary
    min_dim = min(output_img.shape[0], output_img.shape[1])
    if min_dim < 7:
        win_size = min_dim  # Must be an odd integer
        if win_size % 2 == 0:
            win_size -= 1  # Make it odd
        if win_size < 3:
            win_size = 3  # Minimum allowed win_size
    else:
        win_size = 7

    # SSIM calculation
    try:
        ssim_val = ssim(
            target_img,
            output_img,
            data_range=output_img.max() - output_img.min(),
            channel_axis=channel_axis,
            win_size=win_size
        )
    except TypeError:
        # For older versions of skimage
        ssim_val = ssim(
            target_img,
            output_img,
            data_range=output_img.max() - output_img.min(),
            multichannel=(channel_axis is not None)
        )

    # MSE calculation
    mse_val = np.mean((output_img - target_img) ** 2)

    # PSNR calculation
    psnr_val = psnr(target_img, output_img, data_range=output_img.max() - output_img.min())

    return ssim_val, mse_val, psnr_val

# 이미지 전처리 함수 for FID
def preprocess_image_fid(img_path, img_size=299):
    from skimage.io import imread
    from skimage.transform import resize
    from skimage.color import gray2rgb

    img = imread(img_path)
    if img.ndim == 2:
        img = gray2rgb(img)
    img = resize(img, (img_size, img_size), anti_aliasing=True)

    from torchvision.transforms import ToTensor
    img_tensor = ToTensor()(img).unsqueeze(0)  # Tensor 변환 및 배치 추가
    return img_tensor

def evaluate_images():
    """
    이 함수는 /mnt/ssd/ADP-DiT/results/XXX/results.csv 에서 데이터를 읽고,
    (동일 input_path 중) SSIM이 가장 큰 결과만 골라 FID, 메트릭을 계산합니다.
    최종 결과는 /mnt/ssd/ADP-DiT/results/evaluation/{dataset}/{dataset}_best_results.csv 형태로 저장하고,
    평균값은 /mnt/ssd/ADP-DiT/results/evaluation/evaluate.csv 에 누적 기록합니다.
    """
    # 여러 CSV 파일 경로 정의
    evaluations = [
        {
            "dataset": "ADtoAD",
            "input_csv": "/mnt/ssd/ADP-DiT/results/ADtoAD/results.csv",
        },
        {
            "dataset": "MCtoMC",
            "input_csv": "/mnt/ssd/ADP-DiT/results/MCtoMC/results.csv",
        },
        {
            "dataset": "CNtoCN",
            "input_csv": "/mnt/ssd/ADP-DiT/results/CNtoCN/results.csv",
        }
    ]

    # 최종 평가 요약을 저장할 CSV 파일 경로
    # => /mnt/ssd/ADP-DiT/results/evaluation/evaluate.csv
    evaluation_root = "/mnt/ssd/ADP-DiT/results/evaluation"
    summary_csv_path = os.path.join(evaluation_root, "evaluate.csv")

    # FID 계산기 초기화
    fid_calculator = FIDCalculator()

    for eval_ in evaluations:
        dataset = eval_["dataset"]
        input_csv = eval_["input_csv"]

        print(f"\n===== Processing Dataset: {dataset} =====")
        print(f"Input CSV: {input_csv}")

        # CSV 파일 읽기 (필요한 열만 읽기)
        required_columns = ['input_image_path', 'output_image_path', 'target_image_path', 'prompt']
        try:
            df = pd.read_csv(input_csv, usecols=required_columns, skipinitialspace=True)
        except FileNotFoundError:
            print(f"입력 CSV 파일이 존재하지 않습니다: {input_csv}")
            continue
        except ValueError as ve:
            print(f"입력 CSV 파일에 필요한 열이 없습니다: {ve}")
            continue
        except Exception as e:
            print(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
            continue

        # 열 이름 공백 제거
        df.columns = df.columns.str.strip()

        # 데이터의 첫 몇 행을 출력하여 확인
        print("CSV 열 이름:", df.columns.tolist())
        print("CSV 데이터 샘플:")
        print(df.head())

        # -----------------------------
        # (1) 모든 결과를 임시로 저장할 리스트
        # -----------------------------
        all_results = []

        for idx, row in df.iterrows():
            input_path_original = row['input_image_path']
            output_path_original = row['output_image_path']
            target_path_original = row['target_image_path']
            prompt = row['prompt']

            # /workspace/ => /mnt/ssd/ADP-DiT/ 로 치환 (절대경로 변환)
            original_base_path = '/workspace'
            new_base_path = '/mnt/ssd/ADP-DiT'

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

            # 경로 정규화
            input_path = os.path.normpath(input_path)
            output_path = os.path.normpath(output_path)
            target_path = os.path.normpath(target_path)

            print(f"\nProcessing row {idx}:")
            print(f"Input path: {input_path}")
            print(f"Output path: {output_path}")
            print(f"Target path: {target_path}")
            print(f"Prompt: {prompt}")

            # 이미지 파일 존재 여부 확인
            if (not os.path.exists(input_path) or
                not os.path.exists(output_path) or
                not os.path.exists(target_path)):
                print("이미지 파일이 누락되어 현재 행을 건너뜁니다.")
                continue

            try:
                # SSIM, MSE & PSNR 계산용
                output_img = imread(output_path)
                target_img = imread(target_path)

                ssim_val, mse_val, psnr_val = calculate_metrics_ssim_mse(output_img, target_img)

                # FID feature
                target_tensor = preprocess_image_fid(target_path)
                output_tensor = preprocess_image_fid(output_path)
                target_feature = fid_calculator.calculate_features(target_tensor)
                output_feature = fid_calculator.calculate_features(output_tensor)

                all_results.append({
                    "input_path": input_path,
                    "output_path": output_path,
                    "target_path": target_path,
                    "prompt": prompt,
                    "SSIM": ssim_val,
                    "MSE": mse_val,
                    "PSNR": psnr_val,
                    "target_feature": target_feature,
                    "output_feature": output_feature
                })

            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue

        if not all_results:
            print(f"{dataset}: 유효한 결과가 없어 FID 계산을 건너뜁니다.")
            continue

        all_results_df = pd.DataFrame(all_results)

        # 동일 input_path 그룹 중 SSIM 가장 큰 것 선택
        best_results_df = (
            all_results_df.sort_values(by="SSIM", ascending=False)
                          .groupby("input_path", as_index=False)
                          .head(1)
        )

        # 골라낸 행으로 FID 계산
        best_target_features = np.vstack(best_results_df["target_feature"])
        best_output_features = np.vstack(best_results_df["output_feature"])
        fid_val = fid_calculator.calculate_fid(best_target_features, best_output_features)

        # 공통 FID 열
        best_results_df["FID"] = fid_val

        mean_values = {
            "SSIM": best_results_df["SSIM"].mean(),
            "MSE":  best_results_df["MSE"].mean(),
            "PSNR": best_results_df["PSNR"].mean(),
            "FID":  fid_val,
            "Dataset": dataset
        }
        print("\n베스트 결과 기준 평균값:", mean_values)

        # ---------------------------
        # (2) 결과 CSV 저장
        # => /mnt/ssd/ADP-DiT/results/evaluation/{dataset}/{dataset}_best_results.csv
        # ---------------------------
        dataset_dir = os.path.join(evaluation_root, dataset)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            print(f"결과 저장 디렉터리 생성: {dataset_dir}")

        best_csv_path = os.path.join(dataset_dir, f"{dataset}_best_results.csv")

        # 저장할 컬럼
        columns_to_save = [
            "input_path", "output_path", "target_path", "prompt",
            "SSIM", "MSE", "PSNR", "FID"
        ]
        best_save_df = best_results_df[columns_to_save].copy()

        # CSV에 이어서 기록(Append)
        if os.path.exists(best_csv_path):
            best_save_df.to_csv(best_csv_path, mode='a', index=False, header=False)
            print(f"베스트 결과가 기존 CSV 파일에 추가되었습니다: {best_csv_path}")
        else:
            best_save_df.to_csv(best_csv_path, mode='w', index=False, header=True)
            print(f"베스트 결과가 새 CSV 파일로 저장되었습니다: {best_csv_path}")

        # ---------------------------
        # (3) /mnt/ssd/ADP-DiT/results/evaluation/evaluate.csv 에 평균값 추가
        # ---------------------------
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
