import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

import os
import pandas as pd
from PIL import Image

# Mixed Precision
from torch.cuda.amp import autocast, GradScaler

# =====================
# DistilBERT (HF)
# =====================
from transformers import DistilBertTokenizer, DistilBertModel

# =====================
# PARAMS
# =====================
MAX_LEN = 128
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
#LEARNING_RATE = 2e-5
NUM_EPOCHS = 12  # 예시
THRESHOLDS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# 1) Dataset
# =====================
class Multimodal(Dataset):
    """
    CSV 컬럼 예시: id, img, label, text
      - id: 샘플 id
      - img: 경로 (e.g. "train/xxxx.png")
      - label: 0 or 1 (Train/Val) / Test에서는 사용 안 할 수도 있음
      - text: "문장"
    """
    def __init__(self, csv_file, img_base_path, is_test=False):
        """
        Args:
            csv_file (str): CSV 파일 경로
            img_base_path (str): 이미지가 위치한 상위 폴더 경로
            is_test (bool): test 셋일 때 True (label 없는 경우도 고려)
        """
        self.data = pd.read_csv(csv_file)
        self.img_base_path = img_base_path
        self.is_test = is_test

        # DistilBERT 토크나이저
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # --- ID
        sample_id = row['id']

        # --- Image
        img_path = os.path.join(self.img_base_path, row['img'])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # --- Text
        text_str = str(row['text'])
        encoding = self.tokenizer(
            text_str,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)         # shape: (MAX_LEN,)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # --- Label (Train/Val 경우)
        if not self.is_test:
            # label이 있는 경우 float으로
            label = torch.tensor(row['label'], dtype=torch.float)
        else:
            # test 셋이면 label이 없거나 dummy일 수 있음
            label = torch.tensor(0.0)

        return {
            'id': sample_id,
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }


# =====================
# 2) Model
# =====================
class MultimodalClassifier(nn.Module):
    """
    이미지 인코더: EfficientNet-B0 (pretrained)
      - 마지막 분류 레이어 제거 -> 1280 차원 특징 추출
    텍스트 인코더: DistilBertModel (pretrained)
      - [CLS] 토큰 임베딩 (768차원)
    결합: (1280 + 768) → Linear 블록 → 출력 1 (이진분류)
    """
    def __init__(self):
        super(MultimodalClassifier, self).__init__()
        # (1) EfficientNet-B0
        self.image_encoder = models.efficientnet_b0(pretrained=True)
        #   - classifier 부분을 날려서 feature만 추출
        self.image_encoder.classifier = nn.Identity()  # (B, 1280)

        # (2) DistilBERT
        self.text_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # (3) Classifier (concatenate -> linear)
        self.classifier = nn.Sequential(
            nn.Linear(1280 + 768, 512),
            nn.GELU(),
            #nn.Dropout(p=0.5)
            nn.Dropout(p=0.8),
            nn.Linear(512, 64),
            nn.GELU(),
            #nn.Dropout(p=0.3)
            nn.Dropout(p=0.6),
            nn.Linear(64, 1)  # 1차원 로짓 (binary)
        )

    def forward(self, images, input_ids, attention_mask):
        # (a) Image features
        img_features = self.image_encoder(images)  # (B,1280)

        # (b) Text features
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # DistilBertModel: last_hidden_state shape (B, seq_len, hidden_dim=768)
        # [CLS] 토큰 = hidden_state[:,0,:]
        text_features = text_outputs.last_hidden_state[:, 0, :]  # (B,768)

        # (c) Concat
        combined = torch.cat((img_features, text_features), dim=1)  # (B, 2048)

        # (d) Classifier
        logits = self.classifier(combined)  # (B,1)
        return logits


# =====================
# 3) Train Function
# =====================
def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )

    scaler = GradScaler()  # Mixed precision
    best_val_acc = 0.0
    best_threshold = 0.5

    for epoch in range(num_epochs):
        # ----------
        # Training
        # ----------
        model.train()
        running_loss = 0.0
        total, correct = 0, 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)  # (B,) float

            optimizer.zero_grad()
            with autocast():
                logits = model(images, input_ids, attention_mask)  # (B,1)
                loss = criterion(logits.squeeze(), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            with torch.no_grad():
                # 기본 threshold=0.5로 train set accuracy 대략 체크
                preds = (torch.sigmoid(logits.squeeze()) >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # ----------
        # Validation
        #    여러 threshold 시도
        # ----------
        model.eval()
        val_loss = 0.0

        # logits/labels 모아서 threshold별 accuracy를 한번에 계산
        val_logits_list = []
        val_labels_list = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                out_logits = model(images, input_ids, attention_mask)  # (B,1)
                loss = criterion(out_logits.squeeze(), labels)
                val_loss += loss.item()

                val_logits_list.append(out_logits.squeeze().cpu())
                val_labels_list.append(labels.cpu())

        val_logits_tensor = torch.cat(val_logits_list, dim=0)   # (N,)
        val_labels_tensor = torch.cat(val_labels_list, dim=0)   # (N,)

        # threshold 별 accuracy 측정
        threshold_accuracies = {}
        for th in THRESHOLDS:
            preds = (torch.sigmoid(val_logits_tensor) >= th).float()
            correct_count = (preds == val_labels_tensor).sum().item()
            acc = 100.0 * correct_count / len(val_labels_tensor)
            threshold_accuracies[th] = acc

        # 가장 높은 accuracy와 threshold 찾기
        best_th, best_th_acc = max(threshold_accuracies.items(), key=lambda x: x[1])
        avg_val_loss = val_loss / len(val_loader)

        # 스케줄러 업데이트 (기준: best_th_acc)
        scheduler.step(best_th_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f" Train Loss: {avg_train_loss:.4f} | Train Acc (0.5 thr): {train_acc:.2f}%")
        print(f" Val   Loss: {avg_val_loss:.4f}")
        for th in sorted(THRESHOLDS):
            print(f"   Thr={th:.2f} | Val Acc={threshold_accuracies[th]:.2f}%")
        print(f" => Best Thr={best_th:.2f}, Best Thr Acc={best_th_acc:.2f}%")

        # Best Model Save
        if best_th_acc > best_val_acc:
            best_val_acc = best_th_acc
            best_threshold = best_th
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'best_threshold': best_threshold
            }
            torch.save(checkpoint, "best_model.pth")
            print(f"*** Best Model Saved! val_acc={best_val_acc:.2f}%, thr={best_threshold:.2f}\n")
        else:
            print()

        torch.cuda.empty_cache()

    print(f"Training Completed! Best Val Acc={best_val_acc:.2f}%, Threshold={best_threshold:.2f}")
    return best_val_acc, best_threshold


# =====================
# 4) Test Inference
# =====================
def predict_testset(model, test_csv, img_base_path, threshold, out_csv="solution.csv"):
    """
    1) Load test dataset
    2) Use the best model & threshold for prediction
    3) Save results as csv (id, label)
       - label은 0 또는 1
    """
    test_dataset = Multimodal(test_csv, img_base_path, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    model.to(device)

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            sample_ids = batch['id']
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits = model(images, input_ids, attention_mask)  # (B,1)
            proba = torch.sigmoid(logits.squeeze())            # (B,)

            preds = (proba >= threshold).long().cpu()          # 0 or 1
            # 저장
            for s_id, pred_label in zip(sample_ids, preds):
                ids_list.append(s_id.item())   # id가 int 형식일 경우
                preds_list.append(pred_label.item())

    # CSV로 저장
    df_out = pd.DataFrame({"id": ids_list, "label": preds_list})
    df_out.to_csv(out_csv, index=False)
    print(f"Test inference completed. Saved results to '{out_csv}'")


# =====================
# 5) main (실행 예시)
# =====================
def main():
    # CSV 경로 예시
    train_csv = "osai-project/train_text_label_1.csv"
    val_csv   = "osai-project/test_text_label_1.csv"
    test_csv  = "osai-project/test_text_label_1.csv"  # 최종 예측용
    img_base_path = "osai-project"                  # 실제로는 "osai-project/train" 등으로 구성 가능

    # (1) 전체 Train+Val 데이터셋 로딩
    #     여기서는 예시로 별도 CSV가 있다면 각각 Dataset을 만듦
    train_dataset = Multimodal(train_csv, img_base_path, is_test=False)
    val_dataset   = Multimodal(val_csv,   img_base_path, is_test=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # (2) 모델 생성
    model = MultimodalClassifier()

    # (3) 학습
    best_val_acc, best_threshold = train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS)

    # (4) Best 모델 & Threshold 불러오기
    checkpoint = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    best_threshold = checkpoint["best_threshold"]

    print(f"\nLoaded Best Model: val_acc={checkpoint['best_val_acc']:.2f}%, thr={best_threshold:.2f}")

    # (5) Test 예측
    predict_testset(model, test_csv, img_base_path, threshold=best_threshold, out_csv="solution.csv")


if __name__ == "__main__":
    main()
