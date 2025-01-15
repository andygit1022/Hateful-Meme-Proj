# OS-AI Project: Hateful Meme Classification

## Overview
This project focuses on the classification of memes into positive or hateful categories. The dataset and challenge are based on the Hateful Memes dataset introduced by Meta AI. This competition is part of the OS-AI course and involves classifying memes using both image and text data.

### Challenge Details
In this competition, participants aim to detect hateful content in memes. Unlike traditional classification tasks, this competition utilizes low-resolution images (original 256x256 resolution downscaled to 64x64) to make the task more challenging. The reduced quality of the images emphasizes the importance of extracting meaningful features from both visual and textual modalities.

The original Hateful Memes dataset by Meta AI can be found here:
[Meta AI Hateful Memes Dataset](https://github.com/facebookresearch/mmf/tree/main/projects/hateful_memes)

### Competition Link
The competition is hosted on Kaggle:
[Kaggle Competition: OS-AI Project](https://www.kaggle.com/competitions/osai-project/submissions)

### Key Results
- **Rank**: 1st Place
- **Accuracy**: 0.723

---

## Dataset Description
The dataset consists of paired image and text data. Each meme contains:
- **Image**: Low-resolution meme image (64x64 pixels).
- **Text**: Caption associated with the meme.
- **Label**: Binary classification (0 for positive/neutral, 1 for hateful).

The unique challenge lies in combining features from both the image and text to make accurate predictions. The reduction in image resolution further tests the robustness of the multimodal approach.

---

## Code Structure
The solution is implemented using PyTorch and Hugging Face Transformers. The architecture combines a pre-trained vision model (EfficientNet-B0) and a text model (DistilBERT). Below is an overview of the project structure:

### 1. Dataset Loader
The custom PyTorch Dataset class handles:
- Reading CSV files containing image paths, captions, and labels.
- Preprocessing images (resize, normalize) and texts (tokenization with DistilBERT tokenizer).

```python
class Multimodal(Dataset):
    # Handles paired image-text data and preprocessing
```

### 2. Model Architecture
A multimodal classifier combining EfficientNet-B0 (for images) and DistilBERT (for text) with a linear head for binary classification:
```python
class MultimodalClassifier(nn.Module):
    # Combines visual and textual features for final classification
```

### 3. Training Pipeline
The training function optimizes the model using:
- Binary Cross-Entropy with Logits Loss
- AdamW optimizer
- Learning rate scheduler (ReduceLROnPlateau)

Mixed precision is used for faster computation:
```python
from torch.cuda.amp import autocast, GradScaler
```

### 4. Validation
During validation, various thresholds are tested to identify the best threshold for classification:
```python
threshold_accuracies = {}
for th in THRESHOLDS:
    # Calculate accuracy for each threshold
```

### 5. Inference
The test inference function generates predictions for unseen data and saves them in CSV format:
```python
def predict_testset(model, test_csv, img_base_path, threshold, out_csv="solution.csv"):
    # Generates predictions and saves to 'solution.csv'
```

---

## How to Run
1. Clone the repository:
```bash
git clone <repository_url>
cd <repository_directory>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the training script:
```bash
python main.py
```

4. The best model will be saved as `best_model.pth`. Use this model for inference:
```bash
python inference.py
```

---

## Insights from Meta's Hateful Memes Dataset
The Hateful Memes dataset by Meta AI was designed to address the challenge of multimodal hate speech detection. The dataset includes over 10,000 examples of memes with varying degrees of offensiveness. One key feature of this dataset is its intentional inclusion of ambiguous and subtle cases, requiring models to rely on both modalities (image and text) to infer the correct label.

For more information, visit the official Hateful Memes dataset page: [Meta AI Hateful Memes Dataset](https://github.com/facebookresearch/mmf/tree/main/projects/hateful_memes)

---

## References
- Meta AI Research: [https://ai.facebook.com/](https://ai.facebook.com/)
- Kaggle OS-AI Competition: [https://www.kaggle.com/competitions/osai-project/submissions](https://www.kaggle.com/competitions/osai-project/submissions)




# OS-AI 프로젝트: Hateful Meme 분류

## 개요
이 프로젝트는 Meme 을 긍정적 또는 부정적인 범주로 분류하는 것을 목표로 합니다. 데이터셋과 챌린지는 Meta AI에서 발표한 Hateful Memes 데이터셋을 기반으로 합니다. 이 대회는 OS-AI 수업의 일환으로 진행되었으며, 이미지와 텍스트 데이터를 모두 사용하여 밈을 분류합니다.

### 대회 세부 내용
이 대회는 밈에서 혐오 콘텐츠를 감지하는 것을 목표로 합니다. 이 대회에서는 저해상도 이미지(원본 256x256 해상도를 64x64로 다운스케일)를 사용하여 문제의 난이도를 높였습니다. 이미지 해상도가 낮아짐에 따라, 시각적 및 텍스트 데이터를 모두 활용하여 의미 있는 특징을 추출하는 것이 중요합니다.

Meta AI에서 제공한 원본 Hateful Memes 데이터셋은 아래 링크에서 확인할 수 있습니다:
[Meta AI Hateful Memes Dataset](https://github.com/facebookresearch/mmf/tree/main/projects/hateful_memes)

### 대회 링크
대회는 Kaggle에서 진행되었습니다:
[Kaggle 대회: OS-AI 프로젝트](https://www.kaggle.com/competitions/osai-project/submissions)

### 주요 결과
- **순위**: 1위
- **정확도**: 0.723

---

## 데이터셋 설명
데이터셋은 이미지와 텍스트 데이터가 쌍을 이루고 있습니다. 각 밈은 다음으로 구성됩니다:
- **이미지**: 저해상도 밈 이미지 (64x64 픽셀).
- **텍스트**: 밈에 포함된 캡션.
- **라벨**: 이진 분류 (0은 긍정/중립, 1은 혐오).

이 대회의 독특한 도전 과제는 이미지와 텍스트의 특징을 결합하여 정확한 예측을 만드는 것입니다. 이미지 해상도 감소는 멀티모달 접근 방식의 견고성을 더욱 테스트합니다.

---

## 코드 구조
해결책은 PyTorch와 Hugging Face Transformers를 사용하여 구현되었습니다. 아키텍처는 사전 학습된 비전 모델(EfficientNet-B0)과 텍스트 모델(DistilBERT)을 결합합니다. 아래는 프로젝트 구조에 대한 개요입니다:

### 1. 데이터셋 로더
커스텀 PyTorch Dataset 클래스는 다음 작업을 처리합니다:
- 이미지 경로, 캡션 및 라벨이 포함된 CSV 파일 읽기.
- 이미지(크기 조정, 정규화) 및 텍스트(DistilBERT 토크나이저로 토큰화) 전처리.

```python
class Multimodal(Dataset):
    # 이미지-텍스트 데이터를 처리 및 전처리
```

### 2. 모델 아키텍처
EfficientNet-B0(이미지)와 DistilBERT(텍스트)를 결합한 멀티모달 분류기:
```python
class MultimodalClassifier(nn.Module):
    # 시각적 및 텍스트 특징을 결합하여 최종 분류 수행
```

### 3. 학습 파이프라인
모델은 다음을 사용하여 최적화됩니다:
- Binary Cross-Entropy with Logits Loss
- AdamW 옵티마이저
- 학습률 스케줄러(ReduceLROnPlateau)

혼합 정밀도(Mixed Precision)를 사용하여 학습 속도를 향상:
```python
from torch.cuda.amp import autocast, GradScaler
```

### 4. 검증
검증 단계에서는 다양한 임계값(threshold)을 테스트하여 최적의 임계값을 식별합니다:
```python
threshold_accuracies = {}
for th in THRESHOLDS:
    # 각 임계값에 대한 정확도 계산
```

### 5. 추론(Inference)
테스트 데이터에 대한 예측을 생성하고 CSV 형식으로 저장합니다:
```python
def predict_testset(model, test_csv, img_base_path, threshold, out_csv="solution.csv"):
    # 예측 생성 후 'solution.csv'에 저장
```

---

## 실행 방법
1. 레포지토리 클론:
```bash
git clone <repository_url>
cd <repository_directory>
```

2. 의존성 설치:
```bash
pip install -r requirements.txt
```

3. 학습 스크립트 실행:
```bash
python main.py
```

4. 최적 모델은 `best_model.pth`로 저장됩니다. 이 모델을 사용하여 추론:
```bash
python inference.py
```

---

## Meta의 Hateful Memes 데이터셋에 대한 통찰
Meta AI의 Hateful Memes 데이터셋은 멀티모달 혐오 발언 탐지 문제를 해결하기 위해 설계되었습니다. 이 데이터셋은 10,000개 이상의 밈 예제를 포함하며, 다양한 수준의 공격성을 가지고 있습니다. 이 데이터셋의 주요 특징 중 하나는 애매하고 미묘한 사례를 의도적으로 포함하여, 모델이 두 가지 모달리티(이미지와 텍스트)를 모두 활용하여 올바른 라벨을 추론해야 한다는 점입니다.

자세한 내용은 공식 Hateful Memes 데이터셋 페이지를 참조하십시오: [Meta AI Hateful Memes Dataset](https://github.com/facebookresearch/mmf/tree/main/projects/hateful_memes)

---

## 참고자료
- Meta AI Research: [https://ai.facebook.com/](https://ai.facebook.com/)
- Kaggle OS-AI 대회: [https://www.kaggle.com/competitions/osai-project/submissions](https://www.kaggle.com/competitions/osai-project/submissions)


