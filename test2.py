import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 간단한 데이터셋 정의
class DummyDataset(Dataset):
    def __init__(self, size=1000, input_dim=10):
        self.data = torch.randn(size, input_dim)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 간단한 모델 정의
class SimpleModel(nn.Module):
    def __init__(self, input_dim=10):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.fc(x)

# GPU 사용 여부 확인
def main():
    if not torch.cuda.is_available():
        print("CUDA가 지원되지 않습니다.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"사용 가능한 GPU 개수: {num_gpus}")

    if num_gpus < 8:
        print("8개 GPU가 필요합니다. 현재 GPU 개수: {num_gpus}")
        return

    # 데이터셋과 DataLoader 생성
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 모델 및 분산 데이터 병렬 처리 설정
    model = SimpleModel()
    model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
    model = model.cuda()

    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 훈련 루프
    for epoch in range(100):  # Epoch 수를 작게 설정
        print(f"Epoch {epoch+1} 시작")
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f"Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    print("훈련 완료!")

if __name__ == "__main__":
    main()
