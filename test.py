# -*- coding: utf-8 -*-
import os
from PIL import Image

# ���� �̹������� �ִ� ���͸�
input_dir = "/mnt/ssd/datset/images"
# ���� ũ�ӵ� �̹����� ������ ���͸�
output_dir = "/mnt/ssd/datset/images_cropped"

# ���� �̵��� �ȼ� ��
shift_up = 20

# ��� ���� ���͸��� ������ ����
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"��� ���� ���͸��� �����߽��ϴ�: {output_dir}")

# ���͸� �� ��� ���Ͽ� ���� ó��
for filename in os.listdir(input_dir):
    # Ȯ���ڰ� �̹������� Ȯ�� (��: png, jpg, jpeg)
    if not (filename.lower().endswith(".png") or 
            filename.lower().endswith(".jpg") or
            filename.lower().endswith(".jpeg")):
        continue

    # ���� �̹��� ���
    img_path = os.path.join(input_dir, filename)

    try:
        # �̹��� �ҷ����� & RGB ��ȯ
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        # Center Crop���� short_side ����
        short_side = min(width, height)

        # �߾�(top) ��� �� shift_up ����
        center_top = (height - short_side) // 2
        top = center_top - shift_up
        if top < 0:
            top = 0

        left = (width - short_side) // 2
        right = left + short_side
        bottom = top + short_side

        # bottom�� �̹��� ������ ����� �ʵ��� ����
        if bottom > height:
            bottom = height
            top = bottom - short_side

        # �߶� �� 256x256���� ��������
        img_crop = img.crop((left, top, right, bottom))
        img_crop_256 = img_crop.resize((256, 256), Image.LANCZOS)

        # ��� ���� ��� ����
        save_path = os.path.join(output_dir, filename)
        img_crop_256.save(save_path)
        print(f"Processed: {filename} -> {save_path}")

    except Exception as e:
        print(f"���� �߻� - ����: {filename}, ����: {e}")
