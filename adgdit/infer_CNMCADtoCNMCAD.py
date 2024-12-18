# -*- coding: utf-8 -*-
import csv
import re
import os
from datetime import datetime
import subprocess
from concurrent.futures import ProcessPoolExecutor

# Constants
CSV_FILE = "/workspace/dataset/AD2/csvfile/image_text.csv"
NEGATIVE = "low quality, blurry, distorted anatomy, extra artifacts, non-medical objects, unrelated symbols, missing brain regions, incorrect contrast, cartoonish, noise, grainy patches"
DIT_WEIGHT = "/workspace/ADG-DiT_G_2_AD2/001-dit_g_2/checkpoints/e1900.pt"
LOAD_KEY = "module"
MODEL = "DiT-g/2"
IMAGE_ROOT = "/workspace/"
CONDITION_FOLDER_MAP = {
    "Alzheimer Disease": "ADtoAD",
    "Cognitive Normal": "CNtoCN",
    "Mild Cognitive Impairment": "MCtoMC"
}


def extract_images_and_prompts(csv_file, condition):
    """
    Extract both all IMAGE_PATHS and PROMPTS, and those that match 'First visit' based on specified conditions from the CSV file.

    Returns:
        all_image_prompts (list): All image paths and prompts.
        first_visit_image_prompts (list): Image paths and prompts that match 'First visit'.
    """
    cond2 = "First visit"

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
    """Extract the image path and prompt for the latest date for a given subject ID."""
    latest_date = None
    latest_image_path = None
    latest_prompt = None

    for row in all_image_prompts:
        ipath = row[0]
        prompt = row[1]
        ipath_abs = os.path.abspath(os.path.join(IMAGE_ROOT, ipath))

        # Match the subject_id correctly
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
    Extract the target image path and prompt based on the subject_id extracted from image_path.
    """
    # Updated regex with a capturing group for subject_id
    subject_id_match = re.search(r'\d{3}_S_(\d{4})', image_path)
    if subject_id_match:
        subject_id = subject_id_match.group(1)  # Extract the subject_id from the capturing group
        print(f"Extracted subject_id: {subject_id} from image_path: {image_path}")  # Debug print
        return extract_latest_image_and_prompt(subject_id, all_image_prompts)
    else:
        print(f"Could not extract subject_id from image_path: {image_path}")
        return None, None


def run_single_inference(command):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(command)}")
        print(f"Error Output: {e.stderr}")
        raise e


def run_experiment(image_path, prompt, target_path, target_prompt, condition):
    folder_name = CONDITION_FOLDER_MAP.get(condition, "results")
    img_base = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join("results", folder_name, img_base)
    os.makedirs(output_dir, exist_ok=True)

    log_file_path = os.path.join("results", folder_name, "experiment_log.txt")
    csv_file_path = os.path.join("results", folder_name, "results.csv")

    if not os.path.isfile(csv_file_path):
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Updated CSV header: Removed 'input_text', renamed 'target_prompt' to 'prompt'
            writer.writerow(['input_image_path','prompt', 'output_image_path', 'target_image_path'])

    # cfg_scales = range(6, 11)
    # cfg_scales = range(6, 9)
    cfg_scales = range(9, 11)
    infer_steps_list = range(100, 101, 1)

    max_workers = len(cfg_scales) * len(infer_steps_list)
    tasks = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for cfg_scale in cfg_scales:
            for infer_steps in infer_steps_list:
                output_img_path = os.path.join(output_dir, f"output_cfg{cfg_scale}.0_steps{infer_steps}_idx0.png")

                log_message = (
                    f"\n{'='*50}\n"
                    f"Condition: {condition}\n"
                    f"Image Path: {image_path}\n"
                    f"Prompt: {target_prompt}\n"  # Renamed to 'Prompt: target_prompt'
                    f"Output Image Path: {output_img_path}\n"
                    f"Target Image Path: {target_path}\n"
                    f"Running experiment with --cfg-scale={cfg_scale} and --infer-steps={infer_steps} on single GPU\n"
                    f"{'='*50}\n"
                )

                print(log_message)
                with open(log_file_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(log_message)

                command = [
                    "python", "sample_t2i.py",
                    "--infer-mode", "fa",
                    "--model", MODEL,
                    "--prompt", target_prompt,  # Use target_prompt as the prompt
                    "--negative", NEGATIVE,
                    "--image-path", image_path,
                    "--no-enhance",
                    "--dit-weight", DIT_WEIGHT,
                    "--load-key", LOAD_KEY,
                    "--cfg-scale", str(cfg_scale),
                    "--infer-steps", str(infer_steps),
                    "--results-dir", output_dir
                ]

                future = executor.submit(run_single_inference, command)
                # Attach metadata to the future for later reference
                future.cfg_scale = cfg_scale
                future.infer_steps = infer_steps
                future.image_path = image_path
                # future.prompt = prompt  # Original prompt removed
                future.prompt = target_prompt  # Use target_prompt as 'prompt'
                future.output_img_path = output_img_path
                future.target_path = target_path
                future.target_prompt = target_prompt
                tasks.append(future)

        for future in tasks:
            try:
                future.result()
                with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    # Updated CSV writing: Remove original 'prompt', use 'target_prompt' as 'prompt'
                    writer.writerow([future.image_path,future.target_prompt, future.output_img_path, future.target_path ])
                print(f"Experiment with --cfg-scale={future.cfg_scale} and --infer-steps={future.infer_steps} completed successfully.")
            except Exception as e:
                with open(log_file_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(str(e) + '\n')
                print(f"Experiment with --cfg-scale={future.cfg_scale} and --infer-steps={future.infer_steps} failed.")
                print(f"Error: {str(e)}")


def main():
    cond1_list = [
        "Alzheimer Disease",
        "Cognitive Normal",
        "Mild Cognitive Impairment"
    ]

    for condition in cond1_list:
        print(f"Starting experiments for condition: {condition}")

        all_image_prompts, first_visit_image_prompts = extract_images_and_prompts(CSV_FILE, condition)
        if not first_visit_image_prompts:
            print(f"No matching 'First visit' data found for condition: {condition}")
            continue

        for image_path, prompt in first_visit_image_prompts:
            # Extract target_path and target_prompt using all_image_prompts
            target_path, target_prompt = extract_target_path(image_path, all_image_prompts)

            if not image_path or not prompt or not target_path or not target_prompt:
                print(f"Skipping image: {image_path} due to missing information.")
                continue

            # Pass all 5 arguments to the run_experiment function
            run_experiment(image_path, prompt, target_path, target_prompt, condition)

    print("All experiments completed.")


if __name__ == "__main__":
    main()
