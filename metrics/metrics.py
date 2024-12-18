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
from torchvision.transforms import ToTensor, Resize
import os

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
    img = imread(img_path)
    if img.ndim == 2:
        img = gray2rgb(img)
    img = resize(img, (img_size, img_size), anti_aliasing=True)
    img_tensor = ToTensor()(img).unsqueeze(0)  # Tensor 변환 및 배치 추가
    return img_tensor

# 메인 평가 함수
def evaluate_images():
    # 여러 CSV 파일 경로 정의
    evaluations = [
        {
            "dataset": "ADtoAD",
            "input_csv": "/home/juneyonglee/Desktop/ADG-DiT/results/ADtoAD/results.csv",
            "output_csv": "/home/juneyonglee/Desktop/ADG-DiT/metrics/results_ADtoAD.csv"
        },
        {
            "dataset": "MCtoMC",
            "input_csv": "/home/juneyonglee/Desktop/ADG-DiT/results/MCtoMC/results.csv",
            "output_csv": "/home/juneyonglee/Desktop/ADG-DiT/metrics/results_MCtoMC.csv"
        },
        {
            "dataset": "CNtoCN",
            "input_csv": "/home/juneyonglee/Desktop/ADG-DiT/results/CNtoCN/results.csv",
            "output_csv": "/home/juneyonglee/Desktop/ADG-DiT/metrics/results_CNtoCN.csv"
        }
    ]

    # 평가 요약을 저장할 CSV 파일 경로
    summary_csv_path = "/home/juneyonglee/Desktop/ADG-DiT/metrics/evaluate.csv"

    # FID 계산기 초기화
    fid_calculator = FIDCalculator()

    # 요약 결과 저장용 리스트
    summary_results = []

    for eval in evaluations:
        dataset = eval["dataset"]
        input_csv = eval["input_csv"]
        output_csv = eval["output_csv"]

        print(f"\n===== Processing Dataset: {dataset} =====")
        print(f"Input CSV: {input_csv}")
        print(f"Output CSV: {output_csv}")

        # CSV 파일 읽기 (필요한 열만 읽기)
        required_columns = ['input_image_path', 'output_image_path', 'target_image_path', 'prompt']
        try:
            df = pd.read_csv(input_csv, usecols=required_columns, skipinitialspace=True)
        except FileNotFoundError:
            print(f"입력 CSV 파일이 존재하지 않습니다: {input_csv}")
            # 빈 CSV 파일 생성 (옵션)
            create_empty_csv = False  # True로 설정하면 빈 CSV 파일을 생성합니다.
            if create_empty_csv:
                df = pd.DataFrame(columns=required_columns)
                df.to_csv(input_csv, index=False)
                print(f"빈 입력 CSV 파일이 생성되었습니다: {input_csv}")
            else:
                continue
        except ValueError as ve:
            print(f"입력 CSV 파일에 필요한 열이 없습니다: {ve}")
            continue
        except Exception as e:
            print(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
            continue

        # 열 이름 공백 제거
        df.columns = df.columns.str.strip()

        # 열 이름 확인
        print("CSV 열 이름:", df.columns.tolist())

        # 데이터의 첫 몇 행을 출력하여 확인
        print("CSV 데이터 샘플:")
        print(df.head())

        # 결과 저장용 리스트
        results = []

        # FID 계산용 Feature 저장
        target_features = []
        output_features = []

        for idx, row in df.iterrows():
            # 원본 경로 추출
            input_path_original = row['input_image_path']
            output_path_original = row['output_image_path']
            target_path_original = row['target_image_path']
            prompt = row['prompt']  # prompt 추출

            # 절대 경로인지 상대 경로인지 확인 후 변환
            original_base_path = '/workspace'
            new_base_path = '/home/juneyonglee/Desktop/ADG-DiT'

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

            # 절대 경로 정규화
            input_path = os.path.normpath(input_path)
            output_path = os.path.normpath(output_path)
            target_path = os.path.normpath(target_path)

            # 디버깅 정보 출력
            print(f"\nProcessing row {idx}:")
            print(f"Input path: {input_path}")
            print(f"Output path: {output_path}")
            print(f"Target path: {target_path}")
            print(f"Prompt: {prompt}")

            # 이미지 파일 존재 여부 확인
            missing = False
            if not os.path.exists(input_path):
                print(f"Input image not found: {input_path}")
                missing = True
            if not os.path.exists(output_path):
                print(f"Output image not found: {output_path}")
                missing = True
            if not os.path.exists(target_path):
                print(f"Target image not found: {target_path}")
                missing = True
            if missing:
                continue  # 누락된 파일이 있는 경우 현재 행을 건너뜀

            try:
                # SSIM, MSE & PSNR을 위한 이미지 읽기 (원본 크기 유지)
                output_img = imread(output_path)
                target_img = imread(target_path)

                # SSIM, MSE & PSNR 계산
                ssim_val, mse_val, psnr_val = calculate_metrics_ssim_mse(output_img, target_img)

                # FID 계산을 위한 Feature 추출 (299x299)
                target_tensor = preprocess_image_fid(target_path)
                output_tensor = preprocess_image_fid(output_path)

                target_features.append(fid_calculator.calculate_features(target_tensor))
                output_features.append(fid_calculator.calculate_features(output_tensor))

                # 결과 저장
                results.append({
                    "input_path": input_path,
                    "output_path": output_path,
                    "target_path": target_path,
                    "prompt": prompt,  # prompt 추가
                    "SSIM": ssim_val,
                    "MSE": mse_val,
                    "PSNR": psnr_val
                })
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue

        if len(target_features) == 0 or len(output_features) == 0:
            print("FID 계산을 위한 유효한 이미지가 없습니다.")
            continue

        # FID 계산
        target_features = np.vstack(target_features)
        output_features = np.vstack(output_features)
        fid_val = fid_calculator.calculate_fid(target_features, output_features)

        # 결과를 DataFrame으로 저장
        results_df = pd.DataFrame(results)
        results_df['FID'] = fid_val  # 동일 FID 값 적용

        # 평균값 계산
        mean_values = results_df[['SSIM', 'MSE', 'PSNR']].mean().to_dict()
        mean_values['FID'] = fid_val
        mean_values['Dataset'] = dataset  # 어떤 데이터셋인지 추가
        print("\n평균값:", mean_values)

        # 결과 CSV 저장 (덮어쓰지 않고 추가)
        try:
            # Output directory 생성
            output_dir = os.path.dirname(output_csv)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Output directory created: {output_dir}")

            # 파일이 존재하는지 확인
            if os.path.exists(output_csv):
                # 파일이 존재하면 헤더 없이 추가
                results_df[['input_path', 'output_path', 'target_path', 'prompt', 'SSIM', 'MSE', 'PSNR', 'FID']].to_csv(
                    output_csv, mode='a', index=False, header=False
                )
                print(f"결과가 기존 CSV 파일에 추가되었습니다: {output_csv}")
            else:
                # 파일이 존재하지 않으면 새로 생성하고 헤더 포함
                results_df[['input_path', 'output_path', 'target_path', 'prompt', 'SSIM', 'MSE', 'PSNR', 'FID']].to_csv(
                    output_csv, mode='w', index=False, header=True
                )
                print(f"결과가 새 CSV 파일으로 저장되었습니다: {output_csv}")
        except Exception as e:
            print(f"결과 CSV 파일을 저장하는 중 오류가 발생했습니다: {e}")

        # **평균값을 evaluate.csv에 추가**
        try:
            # 출력 디렉터리 추출
            eval_output_dir = os.path.dirname(summary_csv_path)
            if not os.path.exists(eval_output_dir):
                os.makedirs(eval_output_dir)
                print(f"Output directory for summary created: {eval_output_dir}")

            # 평가 요약을 담을 DataFrame 생성
            summary_df = pd.DataFrame([{
                "Dataset": mean_values.get("Dataset", dataset),
                "SSIM": mean_values.get("SSIM"),
                "MSE": mean_values.get("MSE"),
                "PSNR": mean_values.get("PSNR"),
                "FID": mean_values.get("FID")
            }])

            # 평가 요약 CSV 파일 저장 (덮어쓰지 않고 추가)
            if os.path.exists(summary_csv_path):
                # 'Dataset'을 첫 번째 열로 재배치
                if 'Dataset' in summary_df.columns:
                    columns = ['Dataset'] + [col for col in summary_df.columns if col != 'Dataset']
                    summary_df = summary_df[columns]
                summary_df.to_csv(
                    summary_csv_path, mode='a', index=False, header=False
                )
                print(f"평균값이 기존 평가 CSV 파일에 추가되었습니다: {summary_csv_path}")
            else:
                # 'Dataset'을 첫 번째 열로 재배치
                if 'Dataset' in summary_df.columns:
                    columns = ['Dataset'] + [col for col in summary_df.columns if col != 'Dataset']
                    summary_df = summary_df[columns]
                summary_df.to_csv(
                    summary_csv_path, mode='w', index=False, header=True
                )
                print(f"평균값이 새 평가 CSV 파일으로 저장되었습니다: {summary_csv_path}")
        except Exception as e:
            print(f"평균값 CSV 파일을 저장하는 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    evaluate_images()
