# braintumor-DiT

fork 레포지토리
https://github.com/Tencent/HunyuanDiT

20241028 openai/clip-vit-base-patch32 으로 변경

### venv 생성

python -m venv venv

### 모든 gpu 사용하면서 내부 레포지토리 사용하면서 docker image run

### 본인 레포지토리경로 설정 변경

docker run --gpus all -it --shm-size=2g \
 -v /home/juneyonglee/Desktop/braintumor-DiT:/workspace \
 -p 8888:8888 mirrors.tencent.com/neowywang/hunyuan-dit:cuda12

### 데이터셋 전처리 후 데이터셋 폴더 docker 컨테이너에 추가

-v /home/juneyonglee/mydata/BraTS2021_Training_Data:/workspace/mydata/BraTS2021_Training_Data \
