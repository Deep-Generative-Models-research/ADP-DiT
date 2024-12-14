# -*- coding: utf-8 -*-
import csv
import re
import os
from datetime import datetime
import subprocess
from multiprocessing import Pool, cpu_count  # 추가된 부분

# Constants
CSV_FILE = "/mnt/ssd/ADG-DiT/dataset/AD2/csvfile/image_text.csv"
NEGATIVE = "low quality, blurry, distorted anatomy, extra artifacts, non-medical objects, unrelated symbols, missing brain regions, incorrect contrast, cartoonish, noise, grainy patches"
DIT_WEIGHT = "/mnt/ssd/ADG-DiT/ADG-DiT_256_2_ADoldversion/003-dit_XL_2/checkpoints/e4800.pt"
LOAD_KEY = "module"
MODEL = "DiT-256/2"
IMAGE_ROOT = "/mnt/ssd/ADG-DiT"
CONDITION_FOLDER_MAP = {
    "Alzheimer Disease": "ADtoAD",
    "Cognitive Normal": "CNtoCN",
    "Mild Cognitive Impairment": "MCtoMC"
}


def extract_images_and_prompts(csv_file, condition):
    cond2 = "First visit"
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        image_prompts = [(os.path.abspath(os.path.join(IMAGE_ROOT, row['image_path'].strip())), row['text_en'].strip())
                         for row in reader if condition in row['text_en'].strip() and cond2 in row['text_en'].strip()]
    return image_prompts


def extract_target_path(image_path, rows):
    match_id = re.search(r'(\d{3}_S_\d{4})', image_path)
    if not match_id:
        return None

    subject_id = match_id.group(1)
    latest_date = None
    target_path = None

    for row in rows:
        ipath = row['image_path'].strip() if isinstance(row, dict) else row[0]
        ipath_abs = os.path.abspath(os.path.join(IMAGE_ROOT, ipath))

        if subject_id in ipath:
            date_match = re.search(r'_(\d{4}-\d{2}-\d{2})_', ipath)
            if date_match:
                date_str = date_match.group(1)
                try:
                    date_val = datetime.strptime(date_str, '%Y-%m-%d')
                    if latest_date is None or date_val > latest_date:
                        latest_date = date_val
                        target_path = ipath_abs  
                except Exception as e:
                    print(f"Date parsing failed for {date_str}: {e}")
                    continue

    return target_path


def run_experiment(args):
    image_path, prompt, target_path, condition = args
    folder_name = CONDITION_FOLDER_MAP.get(condition, "results")
    img_base = os.path.splitext(os.path.basename(image_path))[0]  
    output_dir = os.path.join("results", folder_name, img_base)
    os.makedirs(output_dir, exist_ok=True)

    for cfg_scale in range(6, 11):  
        for infer_steps in range(100, 101, 1):  
            output_img_path = os.path.join(output_dir, f"output_cfg{cfg_scale}_steps{infer_steps}.png")
            command = [
                "python", "sample_t2i.py",
                "--infer-mode", "fa",
                "--model", MODEL,
                "--prompt", prompt,
                "--negative", NEGATIVE,
                "--image-path", image_path,
                "--no-enhance",
                "--dit-weight", DIT_WEIGHT,
                "--load-key", LOAD_KEY,
                "--cfg-scale", str(cfg_scale),
                "--infer-steps", str(infer_steps),
                "--results-dir", output_dir  
            ]
            try:
                result = subprocess.run(command, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"Experiment with --cfg-scale={cfg_scale} and --infer-steps={infer_steps} failed.")
                print(f"Error output: {e.stderr}")


def main():
    cond1_list = ["Alzheimer Disease", "Cognitive Normal", "Mild Cognitive Impairment"]
    total_args = []

    for condition in cond1_list:
        image_prompts = extract_images_and_prompts(CSV_FILE, condition)
        if not image_prompts:
            continue
        for image_path, prompt in image_prompts:
            target_path = extract_target_path(image_path, image_prompts)
            if not image_path or not prompt or not target_path:
                continue
            total_args.append((image_path, prompt, target_path, condition))

    # 병렬 실행 (멀티프로세싱)
    num_processes = min(len(total_args), cpu_count() - 1)  # CPU 코어 수만큼 제한
    print(f"Starting {num_processes} parallel workers")
    with Pool(num_processes) as pool:
        pool.map(run_experiment, total_args)

    print("All experiments completed.")


if __name__ == "__main__":
    main()
