import pandas as pd
import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from skimage.color import gray2rgb, rgb2gray
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.feature import canny
from scipy.linalg import sqrtm
from torchvision import models
import torch
from torchvision.transforms import ToTensor
from matplotlib.colors import LinearSegmentedColormap

def generate_difference_map(
    input_path, output_path, target_path, difference_map_path, prompt="No Prompt"
):
    """
    input_path, output_path, target_path 이미지를 불러와서
    (1) 네 개 서브플롯(입력/타겟/출력/차이)으로 시각화하고
    (2) 에지(Edge) 부분만 차이 맵을 계산하여 black->red로 표시한 이미지를 그림에 포함합니다.
    """
    # 1) 이미지 읽기
    input_img = imread(input_path)
    output_img = imread(output_path)
    target_img = imread(target_path)

    # 채널 수 / 크기가 다를 경우 보정
    if input_img.ndim == 2:
        input_img = gray2rgb(input_img)
    if output_img.ndim == 2:
        output_img = gray2rgb(output_img)
    if target_img.ndim == 2:
        target_img = gray2rgb(target_img)

    # 리사이즈 (출력이미지와 타겟이미지는 원래 비교 대상으로 동일 크기를 맞추고 있음)
    # 하지만 입력 이미지의 크기가 다를 수 있으므로, 여기서는 일단 그대로 둠.
    # 필요하다면 같은 크기로 맞추고 싶으면 아래처럼 주석 해제:
    #
    # if output_img.shape != target_img.shape:
    #     output_img = resize(output_img, (target_img.shape[0], target_img.shape[1]), anti_aliasing=True)

    # float32 변환
    output_img_f = output_img.astype(np.float32)
    target_img_f = target_img.astype(np.float32)

    # 2) 차이 계산(그레이스케일 기반)
    from skimage.color import rgb2gray
    output_gray = rgb2gray(output_img_f)
    target_gray = rgb2gray(target_img_f)
    diff_map_gray = np.abs(output_gray - target_gray)

    # 3) 에지 추출(canny)
    from skimage.feature import canny
    edges = canny(target_gray / 255.0, sigma=2.0)

    # 에지 부분만 차이 남기기
    diff_map_edges = diff_map_gray * edges.astype(np.float32)

    # 4) 검정→빨간색 컬러맵
    cmap_red = LinearSegmentedColormap.from_list("black_red", [(0, "black"), (1, "red")])

    # 5) 시각화(1×4 subplot)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # (1) Input Image
    axes[0].imshow(input_img.astype(np.uint8))
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # (2) Target Image
    axes[1].imshow(target_img.astype(np.uint8))
    axes[1].set_title("Target Image")
    axes[1].axis("off")

    # (3) Output Image
    axes[2].imshow(output_img.astype(np.uint8))
    axes[2].set_title("Output Image")
    axes[2].axis("off")

    # (4) Difference on Edges
    diff_map_display = diff_map_edges.clip(0, 255).astype(np.uint8)
    im = axes[3].imshow(diff_map_display, cmap=cmap_red)
    axes[3].set_title("Diff(Edges)")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(prompt)
    plt.tight_layout()

    # 저장 (DPI=300으로 화질 개선)
    plt.savefig(difference_map_path, dpi=300)
    plt.close(fig)
    print(f"차이 맵이 저장되었습니다: {difference_map_path}")


def create_csv_if_not_exists(csv_path, required_columns):
    parent_dir = os.path.dirname(csv_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        print(f"디렉터리를 생성했습니다: {parent_dir}")

    if not os.path.exists(csv_path):
        print(f"CSV 파일이 존재하지 않아 새로 생성합니다: {csv_path}")
        empty_df = pd.DataFrame(columns=required_columns)
        empty_df.to_csv(csv_path, index=False)


def generate_evaluation_summary(summary_results, summary_csv_path):
    parent_dir = os.path.dirname(summary_csv_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        print(f"디렉터리를 생성했습니다: {parent_dir}")

    try:
        summary_df = pd.DataFrame(summary_results)
        if 'Dataset' in summary_df.columns:
            columns = ['Dataset'] + [col for col in summary_df.columns if col != 'Dataset']
            summary_df = summary_df[columns]

        if os.path.exists(summary_csv_path):
            summary_df.to_csv(summary_csv_path, mode='a', index=False, header=False)
            print(f"평균값이 기존 평가 CSV 파일에 추가되었습니다: {summary_csv_path}")
        else:
            summary_df.to_csv(summary_csv_path, mode='w', index=False, header=True)
            print(f"평균값이 새 평가 CSV 파일로 저장되었습니다: {summary_csv_path}")
    except Exception as e:
        print(f"평가 요약 CSV 파일을 저장하는 중 오류가 발생했습니다: {e}")


def create_composite_image(diff_results, diff_dir, dataset):
    """
    diff_results 안에 들어있는 difference_map 경로들을 이용해,
    한 장의 PNG로 모아서 저장합니다. (위→아래 세로 배치)
    """
    if not diff_results:
        print("차이 맵 결과가 없어 composite 이미지 생성을 건너뜁니다.")
        return

    n = len(diff_results)
    fig, axes = plt.subplots(n, 1, figsize=(5, 5*n))

    if n == 1:
        axes = [axes]

    for i, item in enumerate(diff_results):
        diff_map_path = item["difference_map"]
        diff_img = imread(diff_map_path)
        axes[i].imshow(diff_img)
        axes[i].axis("off")
        axes[i].set_title(os.path.basename(diff_map_path), fontsize=8)

    composite_path = os.path.join(diff_dir, f"{dataset}_composite_diff.png")
    plt.tight_layout()
    plt.savefig(composite_path, dpi=300)
    plt.close(fig)
    print(f"여러 차이 맵을 합친 이미지를 저장했습니다: {composite_path}")


def process_dataset(dataset_info, summary_results, top_percentage=5):
    """
    - input_csv: 입력(이미지 경로) CSV
    - 여기서 읽은 데이터를 바탕으로 '상위 5%' 차이 맵 생성
    - 결과물은 /mnt/ssd/ADG-DiT/results/evaluation/{dataset}/ 아래에 저장
    """
    dataset = dataset_info["dataset"]
    input_csv = dataset_info["input_csv"]  # 원본 이미지 경로가 들어있는 CSV

    print(f"\n===== Processing Dataset: {dataset} =====")
    print(f"Input CSV: {input_csv}")

    # (1) CSV 읽기
    required_columns = ['input_path', 'output_path', 'target_path', 'prompt', 'SSIM', 'MSE', 'PSNR', 'FID']
    try:
        df = pd.read_csv(input_csv, usecols=required_columns, skipinitialspace=True)
    except FileNotFoundError:
        print(f"입력 CSV 파일이 존재하지 않습니다: {input_csv}")
        return []
    except ValueError as ve:
        print(f"입력 CSV 파일에 필요한 열이 없습니다: {ve}")
        return []
    except Exception as e:
        print(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
        return []

    if df.empty:
        print(f"CSV 파일에 데이터가 없습니다: {input_csv}")
        return []

    # (2) 상위 5% (SSIM) 계산
    top_n = max(1, int(len(df) * top_percentage / 100))
    top_df = df.nlargest(top_n, 'SSIM')

    print(f"총 이미지 쌍: {len(df)}")
    print(f"상위 {top_percentage}% ({len(top_df)}) 이미지 쌍 선택")

    # (3) evaluation/{dataset} 폴더 생성
    evaluation_root = "/mnt/ssd/ADG-DiT/results/evaluation"
    dataset_dir = os.path.join(evaluation_root, dataset)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"결과 저장 디렉토리 생성: {dataset_dir}")

    # diff_map 폴더
    diff_dir = os.path.join(dataset_dir, "diff_maps")
    if not os.path.exists(diff_dir):
        os.makedirs(diff_dir)
        print(f"차이 맵 저장 디렉토리 생성: {diff_dir}")

    diff_results = []

    # (4) 차이 맵 생성
    for idx, row in top_df.iterrows():
        input_path = row['input_path']
        output_path = row['output_path']
        target_path = row['target_path']
        prompt = row.get('prompt', 'No Prompt')

        image_name = os.path.splitext(os.path.basename(output_path))[0]
        difference_map_path = os.path.join(diff_dir, f"{image_name}.png")

        # 입력, 출력, 타겟 이미지를 모두 넘겨서 4-subplot으로 표시
        generate_difference_map(input_path, output_path, target_path, difference_map_path, prompt=prompt)

        diff_results.append({
            "input_image": input_path,
            "output_image": output_path,
            "target_image": target_path,
            "difference_map": difference_map_path,
            "Dataset": dataset,
            "prompt": prompt
        })

    # (5) 차이 맵 결과 CSV
    diff_csv_path = os.path.join(dataset_dir, f"{dataset}_diff_results.csv")
    diff_df = pd.DataFrame(diff_results)
    try:
        if os.path.exists(diff_csv_path):
            diff_df.to_csv(diff_csv_path, mode='a', index=False, header=False)
            print(f"차이 맵 결과가 기존 CSV 파일에 추가되었습니다: {diff_csv_path}")
        else:
            diff_df.to_csv(diff_csv_path, mode='w', index=False, header=True)
            print(f"차이 맵 결과가 새 CSV 파일로 저장되었습니다: {diff_csv_path}")
    except Exception as e:
        print(f"차이 맵 결과 CSV 파일을 저장하는 중 오류가 발생했습니다: {e}")

    # (6) 평균값 계산
    mean_values = top_df[['SSIM', 'MSE', 'PSNR']].mean().to_dict()
    mean_values['FID'] = top_df['FID'].mean()
    mean_values['Dataset'] = dataset
    print("\n평균값:", mean_values)

    summary_results.append(mean_values)

    # (7) 상위 diff_results를 한 장에 합치는 함수 호출
    create_composite_image(diff_results, diff_dir, dataset)

    return diff_results


def main():
    """
    difference_map.py 실행 시, 아래 evaluations에 정의된 CSV(입력 경로)들을 읽어
    /mnt/ssd/ADG-DiT/results/evaluation/ 폴더 아래에 결과(차이맵, CSV 등)를 생성.
    """
    evaluations = [
        {
            "dataset": "ADtoAD",
            "input_csv": "/mnt/ssd/ADG-DiT/results/evaluation/ADtoAD/ADtoAD_best_results.csv",
        },
        {
            "dataset": "MCtoMC",
            "input_csv": "/mnt/ssd/ADG-DiT/results/evaluation/MCtoMC/MCtoMC_best_results.csv",
        },
        {
            "dataset": "CNtoCN",
            "input_csv": "/mnt/ssd/ADG-DiT/results/evaluation/CNtoCN/CNtoCN_best_results.csv",
        }
    ]

    # 최종 평가 요약 CSV
    evaluation_root = "/mnt/ssd/ADG-DiT/results/evaluation"
    summary_csv_path = os.path.join(evaluation_root, "evaluate.csv")

    summary_results = []

    for eval_ in evaluations:
        process_dataset(eval_, summary_results, top_percentage=5)

    generate_evaluation_summary(summary_results, summary_csv_path)


if __name__ == "__main__":
    main()
