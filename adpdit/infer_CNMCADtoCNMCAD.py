# -*- coding: utf-8 -*-
import csv
import re
import os
from datetime import datetime
import subprocess
from concurrent.futures import ProcessPoolExecutor

# Constants
CSV_FILE = "/workspace/dataset/AD3/csvfile/image_text.csv"
NEGATIVE = "low quality, blurry, distorted anatomy, extra artifacts, non-medical objects, unrelated symbols, missing brain regions, incorrect contrast, cartoonish, noise, grainy patches"
DIT_WEIGHT = "/workspace/log_EXP_dit_g_2_AD3/001-dit_g_2/checkpoints/final.pt/mp_rank_00_model_states.pt"
LOAD_KEY = "module"
MODEL = "DiT-g/2"
IMAGE_ROOT = "/workspace/"


# 기존 단일 condition 폴더 매핑 대신 변환 쌍에 따른 폴더 매핑을 정의합니다.
CONVERSION_FOLDER_MAP = {
    # ("Cognitive Normal", "Cognitive Normal"): "CNtoCN",
    # ("Cognitive Normal", "Mild Cognitive Impairment"): "CNtoMC",
    ("Mild Cognitive Impairment", "Mild Cognitive Impairment"): "MCtoMC",
    ("Mild Cognitive Impairment", "Alzheimer Disease"): "MCtoAD",
    ("Alzheimer Disease", "Alzheimer Disease"): "ADtoAD"
}

# 허용되는 변환 쌍
ALLOWED_PAIRS = {
    # ("Cognitive Normal", "Cognitive Normal"),
    # ("Cognitive Normal", "Mild Cognitive Impairment"),
    ("Mild Cognitive Impairment", "Mild Cognitive Impairment"),
    ("Mild Cognitive Impairment", "Alzheimer Disease"),
    ("Alzheimer Disease", "Alzheimer Disease")
}

def get_condition(text):
    """텍스트에서 조건 문자열을 추출합니다."""
    if "Alzheimer Disease" in text:
        return "Alzheimer Disease"
    elif "Mild Cognitive Impairment" in text:
        return "Mild Cognitive Impairment"
    elif "Cognitive Normal" in text:
        return "Cognitive Normal"
    else:
        return None

def extract_images_and_prompts(csv_file, condition):
    """
    CSV 파일에서 모든 이미지 경로와 프롬프트, 그리고 'first visit'인 항목을 추출합니다.
    """
    cond2 = "first visit"

    if not os.path.isfile(csv_file):
        print(f"CSV file not found: {csv_file}")
        return [], []

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_image_prompts = []
        first_visit_image_prompts = []
        for row in reader:
            image_path = row['image_path'].strip()
            text_en = row['text_en'].strip()
            abs_path = os.path.normpath(os.path.join(IMAGE_ROOT, image_path))
            all_image_prompts.append((abs_path, text_en))
            if condition in text_en and cond2 in text_en:
                first_visit_image_prompts.append((abs_path, text_en))
    return all_image_prompts, first_visit_image_prompts

def extract_latest_image_and_prompt(subject_id, all_image_prompts):
    """주어진 subject_id에 대해 최신 날짜의 이미지 경로와 프롬프트를 추출합니다."""
    latest_date = None
    latest_image_path = None
    latest_prompt = None

    for row in all_image_prompts:
        ipath = row[0]
        prompt = row[1]
        ipath_abs = os.path.abspath(os.path.join(IMAGE_ROOT, ipath))

        if f'_S_{subject_id}_' in ipath:
            date_match = re.search(r'_(\d{4}-\d{2}-\d{2})_', ipath)
            if date_match:
                date_str = date_match.group(1)
                try:
                    date_val = datetime.strptime(date_str, '%Y-%m-%d')
                    if latest_date is None or date_val > latest_date:
                        latest_date = date_val
                        latest_image_path = ipath_abs
                        latest_prompt = prompt
                except Exception as e:
                    print(f"Date parsing failed for {date_str}: {e}")
                    continue

    if latest_image_path:
        print(f"Latest image for subject_id {subject_id}: {latest_image_path}")
    else:
        print(f"No images found for subject_id {subject_id}")

    return latest_image_path, latest_prompt

def extract_target_path(image_path, all_image_prompts):
    """
    이미지 경로에서 subject_id를 추출하고, 해당 subject의 최신 이미지와 프롬프트를 반환합니다.
    """
    subject_id_match = re.search(r'\d{3}_S_(\d{4})', image_path)
    if subject_id_match:
        subject_id = subject_id_match.group(1)
        print(f"Extracted subject_id: {subject_id} from image_path: {image_path}")
        return extract_latest_image_and_prompt(subject_id, all_image_prompts)
    else:
        print(f"Could not extract subject_id from image_path: {image_path}")
        return None, None

def run_single_inference(command, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # GPU ID 설정
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(command)}")
        print(f"Error Output: {e.stderr}")
        raise e

def run_experiment(image_path, prompt, target_path, target_prompt, conversion_pair):
    # conversion_pair: (input_condition, target_condition)
    folder_name = CONVERSION_FOLDER_MAP.get(conversion_pair, "results")
    img_base = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join("results", folder_name, img_base)
    complete_marker = os.path.join(output_dir, "complete.txt")

    # 만약 output_dir에 완료 표시 파일이 있으면 이미 처리된 것으로 간주하고 건너뜁니다.
    if os.path.exists(complete_marker):
        print(f"Skipping experiment for {image_path} as output already exists in {output_dir}.")
        return

    os.makedirs(output_dir, exist_ok=True)

    log_file_path = os.path.join("results", folder_name, "experiment_log.txt")
    csv_file_path = os.path.join("results", folder_name, "results.csv")

    if not os.path.isfile(csv_file_path):
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['input_image_path', 'prompt', 'output_image_path', 'target_image_path'])

    # 여러 cfg_scale 값과 infer_steps를 사용합니다.
    cfg_scales = [1, 2, 3]
    infer_steps_list = range(90, 111, 1)
    gpu_ids = list(range(1))  # GPU 0 사용

    # 각 GPU에 대해 max_workers=1인 executor 생성
    gpu_executors = {gpu_id: ProcessPoolExecutor(max_workers=1) for gpu_id in gpu_ids}
    futures = []
    global_task_idx = 0

    for cfg_scale in cfg_scales:
        for infer_steps in infer_steps_list:
            gpu_id = gpu_ids[global_task_idx % len(gpu_ids)]
            global_task_idx += 1

            output_img_path = os.path.join(output_dir, f"output_cfg{cfg_scale}.0_steps{infer_steps}_idx0.png")

            log_message = (
                f"\n{'='*50}\n"
                f"Conversion: {conversion_pair[0]} -> {conversion_pair[1]}\n"
                f"Image Path: {image_path}\n"
                f"Prompt: {target_prompt}\n"
                f"Output Image Path: {output_img_path}\n"
                f"Target Image Path: {target_path}\n"
                f"Running experiment with --cfg-scale={cfg_scale}, --infer-steps={infer_steps} on GPU {gpu_id}\n"
                f"{'='*50}\n"
            )
            print(log_message)
            with open(log_file_path, 'a', encoding='utf-8') as log_file:
                log_file.write(log_message)

            command = [
                "python", "sample_t2i.py",
                "--infer-mode", "fa",
                "--model", MODEL,
                "--prompt", target_prompt,
                "--negative", NEGATIVE,
                "--image-path", image_path,
                "--no-enhance",
                "--dit-weight", DIT_WEIGHT,
                "--load-key", LOAD_KEY,
                "--cfg-scale", str(cfg_scale),
                "--infer-steps", str(infer_steps),
                "--results-dir", output_dir
            ]

            future = gpu_executors[gpu_id].submit(run_single_inference, command, gpu_id)
            future.cfg_scale = cfg_scale
            future.infer_steps = infer_steps
            future.image_path = image_path
            future.prompt = target_prompt
            future.output_img_path = output_img_path
            future.target_path = target_path
            futures.append(future)

    for future in futures:
        try:
            future.result()
            with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([future.image_path, future.prompt, future.output_img_path, future.target_path])
            print(f"Experiment with --cfg-scale={future.cfg_scale} and --infer-steps={future.infer_steps} completed successfully.")
        except Exception as e:
            with open(log_file_path, 'a', encoding='utf-8') as log_file:
                log_file.write(str(e) + '\n')
            print(f"Experiment with --cfg-scale={future.cfg_scale} and --infer-steps={future.infer_steps} failed.")
            print(f"Error: {str(e)}")

    for executor in gpu_executors.values():
        executor.shutdown()

    # 실험이 모두 완료되면 완료 표시 파일 생성
    with open(complete_marker, "w", encoding="utf-8") as f:
        f.write("complete")
    print(f"Experiment for {image_path} is marked complete in {output_dir}.")


def main():
    cond1_list = [
        "Cognitive Normal",
        "Mild Cognitive Impairment",
        "Alzheimer Disease"
    ]

    for condition in cond1_list:
        print(f"Starting experiments for condition: {condition}")
        all_image_prompts, first_visit_image_prompts = extract_images_and_prompts(CSV_FILE, condition)
        if not first_visit_image_prompts:
            print(f"No matching 'first visit' data found for condition: {condition}")
            continue

        for image_path, prompt in first_visit_image_prompts:
            target_path, target_prompt = extract_target_path(image_path, all_image_prompts)

            # input image와 target이 동일한 경우 건너뜁니다.
            if image_path == target_path:
                print(f"Skipping inference for most recent first visit image: {image_path}")
                continue

            # input condition은 현재 루프의 condition, target condition은 target_prompt에서 추출합니다.
            input_condition = condition
            target_condition = get_condition(target_prompt)
            if target_condition is None:
                print(f"Skipping image: {image_path} due to unknown target condition.")
                continue

            conversion_pair = (input_condition, target_condition)
            if conversion_pair not in ALLOWED_PAIRS:
                print(f"Skipping inference for subject {image_path}: conversion {conversion_pair} is not allowed.")
                continue

            run_experiment(image_path, prompt, target_path, target_prompt, conversion_pair)

    print("All experiments completed.")

if __name__ == "__main__":
    main()
