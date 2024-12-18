import pandas as pd
import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.linalg import sqrtm
from torchvision import models
import torch
from torchvision.transforms import ToTensor
from matplotlib.colors import LinearSegmentedColormap

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
    # 그레이스케일 이미지를 RGB로 변환
    if output_img.ndim == 2:
        output_img = gray2rgb(output_img)
    if target_img.ndim == 2:
        target_img = gray2rgb(target_img)

    # 이미지가 float32 형식인지 확인
    output_img = output_img.astype(np.float32)
    target_img = target_img.astype(np.float32)

    # 이미지 형상과 데이터 타입 출력 (디버깅용)
    print(f"Output image shape: {output_img.shape}, dtype: {output_img.dtype}")
    print(f"Target image shape: {target_img.shape}, dtype: {target_img.dtype}")

    # 채널 축 결정
    if output_img.ndim == 3:
        channel_axis = -1  # 마지막 축
    else:
        channel_axis = None

    # win_size 조정
    min_dim = min(output_img.shape[0], output_img.shape[1])
    if min_dim < 7:
        win_size = min_dim  # 홀수여야 함
        if win_size % 2 == 0:
            win_size -= 1  # 홀수로 변경
        if win_size < 3:
            win_size = 3  # 최소 win_size
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
        # skimage의 이전 버전용
        ssim_val = ssim(
            target_img,
            output_img,
            data_range=output_img.max() - output_img.min(),
            multichannel=(channel_axis is not None)
        )

    # MSE 계산
    mse_val = np.mean((output_img - target_img) ** 2)

    # PSNR 계산
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

# 차이 맵 생성 및 저장 함수
def generate_difference_map(output_image_path, target_image_path, difference_map_path, cmap='black_red', prompt=""):
    try:
        output_img = imread(output_image_path)
        target_img = imread(target_image_path)

        if output_img.ndim == 2:
            output_img = gray2rgb(output_img)
        if target_img.ndim == 2:
            target_img = gray2rgb(target_img)

        if output_img.shape != target_img.shape:
            target_img = resize(target_img, output_img.shape[:2], anti_aliasing=True, preserve_range=True).astype(output_img.dtype)

        difference = output_img.astype(np.float32) - target_img.astype(np.float32)
        difference_magnitude = np.sum(np.abs(difference), axis=2)

        cmap_custom = LinearSegmentedColormap.from_list('black_red', ['black', 'red'])

        plt.figure(figsize=(8, 6))
        plt.imshow(difference_magnitude, cmap=cmap_custom)
        plt.colorbar(label='Difference Magnitude')
        plt.axis('off')
        plt.title(f'Difference Map (Output - Target)\nPrompt: {prompt}')

        # plt.savefig(difference_map_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"차이 맵 저장 완료: {difference_map_path}")
    except Exception as e:
        print(f"차이 맵 생성 중 오류 발생: {e}")

# 컴포지트 이미지 생성 함수
def create_composite_image(diff_results, diff_dir, dataset, images_per_row=4):
    try:
        num_sets = len(diff_results)
        if num_sets == 0:
            print(f"{dataset} 데이터셋에 대한 차이 맵 세트가 없습니다.")
            return

        fig, axes = plt.subplots(num_sets, 4, figsize=(16, 4 * num_sets))
        if num_sets == 1:
            axes = [axes]

        for i, image_set in enumerate(diff_results):
            input_img = imread(image_set['input_image'])
            output_img = imread(image_set['output_image'])
            target_img = imread(image_set['target_image'])

            if input_img.ndim == 2:
                input_img = gray2rgb(input_img)
            if output_img.ndim == 2:
                output_img = gray2rgb(output_img)
            if target_img.ndim == 2:
                target_img = gray2rgb(target_img)

            difference = output_img.astype(np.float32) - target_img.astype(np.float32)
            difference_magnitude = np.sum(np.abs(difference), axis=2)

            axes[i][0].imshow(input_img)
            axes[i][0].set_title('Input Image')
            axes[i][0].axis('off')

            axes[i][1].imshow(output_img)
            axes[i][1].set_title('Output Image')
            axes[i][1].axis('off')

            axes[i][2].imshow(target_img)
            axes[i][2].set_title('Target Image')
            axes[i][2].axis('off')

            cmap_custom = LinearSegmentedColormap.from_list('black_red', ['black', 'red'])
            axes[i][3].imshow(difference_magnitude, cmap=cmap_custom)
            prompt = image_set.get('prompt', image_set.get('Prompt', 'No Prompt'))
            axes[i][3].set_title(f'Difference Map\nPrompt: {prompt}')
            axes[i][3].axis('off')

        plt.tight_layout()
        composite_image_path = os.path.join(diff_dir, f"{dataset}_comparison.png")
        plt.savefig(composite_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"컴포지트 이미지 저장 완료: {composite_image_path}")
    except Exception as e:
        print(f"컴포지트 이미지 생성 중 오류 발생: {e}")

# 개별 데이터셋 처리 함수
def process_dataset(dataset_info, summary_results, top_percentage=5):
    """
    개별 데이터셋에 대해 상위 성능 이미지 쌍을 식별하고 차이 맵을 생성합니다.

    Parameters:
    - dataset_info: 딕셔너리 형태의 데이터셋 정보 (dataset name, input_csv, output_csv)
    - summary_results: 평가 요약을 저장할 리스트
    - top_percentage: 상위 몇 퍼센트를 선택할지 결정 (default: 5)
    """
    dataset = dataset_info["dataset"]
    output_csv = dataset_info["output_csv"]

    print(f"\n===== Processing Dataset: {dataset} =====")
    print(f"Output CSV: {output_csv}")

    # 출력 CSV 파일 읽기
    try:
        df = pd.read_csv(output_csv)
    except FileNotFoundError:
        print(f"출력 CSV 파일이 존재하지 않습니다: {output_csv}")
        return []  # 빈 리스트 반환
    except Exception as e:
        print(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
        return []  # 빈 리스트 반환

    # 필요한 열 확인
    required_columns = ['input_path', 'output_path', 'target_path', 'prompt', 'SSIM', 'MSE', 'PSNR', 'FID']
    if not all(col in df.columns for col in required_columns):
        print(f"CSV 파일에 필요한 열이 없습니다. 필요한 열: {required_columns}")
        return []  # 빈 리스트 반환

    # 상위 5% 선택 (SSIM 기준)
    top_n = max(1, int(len(df) * top_percentage / 100))
    top_df = df.nlargest(top_n, 'SSIM')

    print(f"총 이미지 쌍: {len(df)}")
    print(f"상위 {top_percentage}% ({len(top_df)}) 이미지 쌍 선택")

    # 결과 디렉토리 생성
    diff_dir = os.path.join(os.path.dirname(output_csv), f"{dataset}_diff_maps")
    if not os.path.exists(diff_dir):
        os.makedirs(diff_dir)
        print(f"차이 맵 저장 디렉토리 생성: {diff_dir}")

    # 차이 맵 결과 저장용 리스트
    diff_results = []

    # 상위 이미지 쌍에 대해 차이 맵 생성
    for idx, row in top_df.iterrows():
        input_path = row['input_path']
        output_path = row['output_path']
        target_path = row['target_path']

        # 차이 맵 파일명 생성
        image_name = os.path.splitext(os.path.basename(output_path))[0]
        difference_map_path = os.path.join(diff_dir, f"{image_name}_diff_map.png")

        # Prompt 추출
        prompt = row.get('prompt', row.get('Prompt', 'No Prompt'))

        # 차이 맵 생성 및 저장
        generate_difference_map(output_path, target_path, difference_map_path, prompt=prompt)

        # 차이 맵 정보 저장 (prompt 추가)
        diff_results.append({
            "input_image": input_path,
            "output_image": output_path,
            "target_image": target_path,
            "difference_map": difference_map_path,
            "Dataset": dataset,
            "prompt": prompt  # 추가된 부분
        })

    # 차이 맵 결과를 CSV로 저장 (5개의 열: input_image, output_image, target_image, difference_map, prompt)
    if diff_results:
        diff_csv_path = os.path.join(os.path.dirname(output_csv), f"{dataset}_diff_results.csv")
        diff_df = pd.DataFrame(diff_results)
        try:
            if os.path.exists(diff_csv_path):
                # 파일이 존재하면 헤더 없이 추가
                diff_df.to_csv(diff_csv_path, mode='a', index=False, header=False)
                print(f"차이 맵 결과가 기존 CSV 파일에 추가되었습니다: {diff_csv_path}")
            else:
                # 파일이 존재하지 않으면 새로 생성하고 헤더 포함
                diff_df.to_csv(diff_csv_path, mode='w', index=False, header=True)
                print(f"차이 맵 결과가 새 CSV 파일으로 저장되었습니다: {diff_csv_path}")
        except Exception as e:
            print(f"차이 맵 결과 CSV 파일을 저장하는 중 오류가 발생했습니다: {e}")
    else:
        print("상위 5% 이미지 쌍에 대한 차이 맵 결과가 없습니다.")

    # 평균값 계산
    mean_values = top_df[['SSIM', 'MSE', 'PSNR']].mean().to_dict()
    mean_values['FID'] = top_df['FID'].mean()
    mean_values['Dataset'] = dataset  # 어떤 데이터셋인지 추가
    print("\n평균값:", mean_values)

    # 평균값을 summary_results에 추가
    summary_results.append(mean_values)

    # 차이 맵을 하나의 이미지로 나열하여 저장
    create_composite_image(diff_results, diff_dir, dataset)

    return diff_results  # HTML 리포트 생성을 위해 반환

# 평가 요약 생성 함수
def generate_evaluation_summary(summary_results, summary_csv_path):
    """
    평가 요약 결과를 summary CSV 파일에 저장합니다.

    Parameters:
    - summary_results: 평가 요약을 저장할 리스트
    - summary_csv_path: 평가 요약 CSV 파일 경로
    """
    try:
        summary_df = pd.DataFrame(summary_results)
        # 'Dataset'을 첫 번째 열로 재배치
        if 'Dataset' in summary_df.columns:
            # 'Dataset'을 첫 번째 열로 설정
            columns = ['Dataset'] + [col for col in summary_df.columns if col != 'Dataset']
            summary_df = summary_df[columns]
        if os.path.exists(summary_csv_path):
            # 파일이 존재하면 헤더 없이 추가
            summary_df.to_csv(summary_csv_path, mode='a', index=False, header=False)
            print(f"평균값이 기존 평가 CSV 파일에 추가되었습니다: {summary_csv_path}")
        else:
            # 파일이 존재하지 않으면 새로 생성하고 헤더 포함
            summary_df.to_csv(summary_csv_path, mode='w', index=False, header=True)
            print(f"평균값이 새 평가 CSV 파일으로 저장되었습니다: {summary_csv_path}")
    except Exception as e:
        print(f"평균값 CSV 파일을 저장하는 중 오류가 발생했습니다: {e}")

# 메인 평가 함수
def main():
    # 평가 파일들 정의
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

    # 요약 결과 저장용 리스트
    summary_results = []

    # FID 계산기 초기화
    fid_calculator = FIDCalculator()

    # 전체 차이 맵 결과 저장용 리스트 (HTML 리포트용)
    all_diff_results = []

    # 각 데이터셋에 대해 처리
    for eval in evaluations:
        diff_results = process_dataset(eval, summary_results, top_percentage=5)
        all_diff_results.extend(diff_results)

    # 평가 요약 CSV 파일에 저장
    if summary_results:
        generate_evaluation_summary(summary_results, summary_csv_path)
    else:
        print("평가 요약 결과가 없습니다.")

if __name__ == "__main__":
    main()
