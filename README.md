# Tennessee Eastman Process 결함 탐지 - Temporal Deep Learning 모델

Tennessee Eastman Process (TEP) 데이터셋을 사용한 화학공정 결함 탐지 시스템입니다. CNN1D2D+GAN 하이브리드 모델을 통해 21가지 결함 유형을 분류하고 시계열 데이터를 생성합니다.

## 프로젝트 개요

**핵심 모델: GAN v5**
- CNN1D2D + GAN 결합 아키텍처
- 멀티태스크 학습: 결함 분류 + 데이터 생성 + 실제/가짜 판별
- 시계열 패턴 학습: LSTM Generator + CNN1D2D Discriminator

**주요 특징**
- 52개 센서 데이터 기반 시계열 분석
- 21가지 결함 유형 분류 (정상상태 포함)
- 하이브리드 딥러닝 아키텍처 (CNN1D2D + GAN)
- Multitask Learning 접근법
- 데이터 생성 및 증강 기능

## 프로젝트 구조

```
tennessee_eastman_diploma/
├── src/
│   ├── data/              # 데이터 처리 및 로딩
│   │   ├── dataset.py     # TEP 데이터셋 클래스들
│   │   └── make_dataset.py
│   ├── models/            # 딥러닝 모델 구현
│   │   ├── convolutional_models.py  # TCN, CNN1D2D 모델
│   │   ├── recurrent_models.py      # LSTM 기반 모델
│   │   ├── train_model.py           # 기본 CNN 분류기 훈련
│   │   ├── train_model_1d_composed.py # CNN1D2D 하이브리드 훈련
│   │   ├── train_model_gan_v*.py    # GAN 실험 버전들 (v1~v5)
│   │   └── utils.py                 # 유틸리티 함수
│   ├── features/          # 특성 추출
│   └── visualization/     # 시각화
├── notebooks/            # 탐색적 데이터 분석
│   ├── epoch_1_marks_protype/
│   └── epoch_2_ildar_proto/
│       └── EDA_v1.ipynb  # 데이터 분석 노트북
├── data/                # 데이터셋 저장소
│   └── raw/            # 원본 데이터 파일 위치
│       ├── TEP_FaultFree_Training.RData
│       ├── TEP_Faulty_Training.RData  
│       ├── TEP_FaultFree_Testing.RData
│       └── TEP_Faulty_Testing.RData
├── models/              # 훈련된 모델 저장소
└── reports/             # 결과 리포트
```

## 데이터셋 정보

**Tennessee Eastman Process 데이터**
- 센서 개수: 52개 (22개 공정 측정값, 19개 분석 측정값, 11개 조작 변수)
- 결함 유형: 21가지 (정상상태 포함)
- 샘플링 주기: 3분
- 시뮬레이션 기간: 
  - 훈련: 25시간 (500 샘플)
  - 테스트: 48시간 (960 샘플)

**데이터 구조**
```python
# 훈련 데이터: 21 × 500 × 500 = 5,250,000 샘플
# 테스트 데이터: 21 × 500 × 960 = 10,080,000 샘플
```

**결함 특성**
- 정상 운전: 처음 1시간 (20샘플)
- 결함 발생: 1시간 후부터 (21번째 샘플부터)
- 결함 유형: IDV(1) ~ IDV(20) + 정상상태(0)

## 구현된 모델들

### 메인 모델: GAN v5 (CNN1D2D+GAN 하이브리드)

파일: `src/models/train_model_gan_v5.py`

**구조**: 
- Generator: `LSTMGenerator` (시계열 데이터 생성)
- Discriminator: `CNN1D2DDiscriminatorMultitask` (1D TCN + 2D CNN)

**기능**: 
- 시계열 데이터 생성 (GAN)
- 결함 분류 (21개 클래스)
- 실제/가짜 판별
- 멀티태스크 학습

### 베이스라인 비교 모델들

성능 비교를 위한 단순한 베이스라인 모델들:

**1. 기본 CNN 분류기**
- 파일: `src/models/train_model.py`
- 클래스: `Net`
- 목적: 전통적인 CNN 베이스라인

**2. CNN1D2D 분류기 (GAN 없는 버전)**
- 파일: `src/models/train_model_1d_composed.py`
- 클래스: `CNN1D2D`
- 목적: CNN1D2D 구조만 단독 사용한 베이스라인

**3. GAN 실험 버전들 (v1~v4)**
- 용도: GAN v5 개발 과정의 실험 버전들
- 현재: 더 이상 사용하지 않음

## 딥러닝 구성 요소들

### Temporal Convolutional Network (TCN) 컴포넌트
- 파일: `src/models/convolutional_models.py`
- 클래스: `TemporalConvNet`, `TemporalBlock`
- 특징: Causal dilated convolution
- 구조: 
  - Dilated convolution (dilation: 1, 2, 4, 8, ...)
  - Residual connections
  - Chomp layer (미래 정보 차단)

### LSTM 컴포넌트
- 파일: `src/models/recurrent_models.py`
- 클래스: `LSTMGenerator`, `LSTMDiscriminator`, `TEPRNN`
- 용도: 시계열 생성, 판별, 분류

### 멀티태스크 판별기들
- **CausalConvDiscriminatorMultitask**: TCN 기반 멀티태스크
- **CNN1D2DDiscriminatorMultitask**: 하이브리드 멀티태스크
- 공통 태스크: 
  - 결함 분류 (21개 클래스)
  - 실제/가짜 데이터 판별

## 데이터 처리 클래스들

### 데이터셋 클래스들
- **TEPDataset**: 기본 TEP 데이터셋 (GAN용)
- **TEPDatasetV4**: 전체 시퀀스 처리용 데이터셋
- **TEPRNNGANDataset**: RNN-GAN 훈련용 데이터셋
- **TEPCNNDataset**: CNN 훈련용 데이터셋

### 데이터 전처리
```python
# 정규화 상수 (EDA_v1.ipynb에서 계산됨)
TEP_MEAN = torch.tensor([2.608e-01, 3.664e+03, 4.506e+03, ...])
TEP_STD = torch.tensor([1.461e-01, 4.278e+01, 1.087e+02, ...])

# 정규화 적용
normalized_data = (data - TEP_MEAN) / TEP_STD
```

### 데이터 변환 클래스들
- **ToTensor**: 넘파이 배열을 PyTorch 텐서로 변환
- **Normalize**: TEP 데이터 정규화
- **InverseNormalize**: 정규화 역변환

## 멀티태스크 학습

구현된 멀티태스크 모델들은 두 가지 태스크를 동시에 학습합니다:
- 태스크 1: 결함 분류 (21개 클래스)
- 태스크 2: 정상/비정상 판별 (실제/가짜 데이터 구분)

손실 함수는 가중 결합으로 계산됩니다:
```python
total_loss = fault_type_weight * fault_type_loss + real_fake_weight * real_fake_loss
```

## 사용 방법

### 1. 환경 설정

**CONDA 환경 구성**

```bash
# 1. 새 CONDA 환경 생성
conda create -n tep_project python=3.7 -y
conda activate tep_project

# 2. 기본 과학 계산 패키지 설치
conda install numpy pandas scipy matplotlib scikit-learn -y

# 3. PyTorch 설치 (GPU 버전) - 중요!
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117

# 4. 기타 필수 패키지 설치
pip install pyreadr tensorboardx python-dotenv memory-profiler click pillow opencv-python scikit-image

# 5. 프로젝트 로컬 설치
pip install -e .
```

### 2. 데이터 준비

**필수: CSV 데이터 파일**

GAN v5 모델 훈련에는 다음 CSV 파일들이 필요합니다:

```bash
# 훈련 데이터 (필수)
data/train_faults/
├── train_fault_0.csv  # 정상 운전 (18MB)
├── train_fault_1.csv  # 결함 1 (18MB)
├── train_fault_2.csv  # 결함 2 (18MB)
├── ...
└── train_fault_12.csv # 결함 12 (18MB)

# 테스트 데이터 (평가용)
data/test_faults/
├── test_fault_0.csv   # 정상 운전 (34MB)
├── test_fault_1.csv   # 결함 1 (34MB)
├── test_fault_2.csv   # 결함 2 (34MB)
├── ...
└── test_fault_12.csv  # 결함 12 (34MB)
```

**CSV 파일 형식:**
- 각 파일: 100개 시뮬레이션 런
- 훈련: 런당 500 시점 (25시간)
- 테스트: 런당 960 시점 (48시간) 
- 센서: 52개 (xmeas_1~xmeas_41, xmv_1~xmv_11)

**선택사항: 원본 RData 파일**

기존 R 데이터 파일들도 지원됩니다 (베이스라인 비교용):

- `TEP_FaultFree_Training.RData` (24MB) - 훈련용 정상 데이터
- `TEP_Faulty_Training.RData` (471MB) - 훈련용 결함 데이터  
- `TEP_FaultFree_Testing.RData` (45MB) - 테스트용 정상 데이터
- `TEP_Faulty_Testing.RData` (798MB) - 테스트용 결함 데이터

### 3. 모델 훈련

**메인 모델 훈련 (이것만 하면 됨!)**

```bash
# CNN1D2D+GAN 하이브리드 모델 (메인 모델) - CSV 데이터 사용
python -m src.models.train_model_gan_v5 --cuda 0 --run_tag main_model
```

이 하나의 명령으로 전체 시스템이 완성됩니다:
- **CSV 데이터 자동 로드**: `data/train_faults/train_fault_*.csv` 파일 사용
- 시계열 데이터 생성 (Generator)
- 결함 분류 (13개 클래스: 정상 + 12가지 결함)  
- 정상/비정상 판별
- 멀티태스크 학습

**주요 옵션:**
- `--cuda 0`: GPU 번호 (0번 GPU 사용, 필수 옵션)
- `--run_tag main_model`: 실험 태그 (로그 구분용)

## 핵심 구현 내용

### 1. Temporal Convolutional Network
```python
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)  # 미래 정보 제거
        # ... residual connection
```

### 2. 멀티태스크 판별기
```python
class CausalConvDiscriminatorMultitask(nn.Module):
    def forward(self, x, _, channel_last=True):
        common = self.tcn(x.transpose(1, 2) if channel_last else x).transpose(1, 2)
        
        # 결함 유형 분류 헤드
        type_logits = self.fault_type_head_fc3(
            self.activation(self.fault_type_head_fc2(
                self.activation(self.fault_type_head_fc1(common)))))
        
        # 실제/가짜 판별 헤드
        real_fake_logits = self.real_fake_head_fc2(
            self.activation(self.real_fake_head_fc1(common)))
        
        return type_logits, real_fake_logits
```

### 3. 시계열 시각화 도구
```python
def time_series_to_plot(time_series_batch, dpi=35, titles=None):
    """시계열 배치를 플롯 그리드로 변환"""
    # 각 시계열을 matplotlib 플롯으로 변환
    # 텐서로 결합하여 TensorBoard에서 시각화 가능
```

## GAN v5 모델 구조 및 작동 원리

### 모델 아키텍처

**GAN v5는 전통적인 데이터 증강 방식이 아닌 적대적 학습(Adversarial Training)을 활용한 robust한 결함 분류 모델입니다.**

#### 구성 요소
1. **Generator (LSTMGenerator)**
   - 역할: 가짜 TEP 시계열 데이터 생성
   - 입력: 노이즈(100차원) + 결함 유형 라벨(1차원)
   - 출력: 52개 센서의 시계열 데이터 (500 시간 스텝)

2. **Discriminator (CNN1D2DDiscriminatorMultitask)**
   - 역할: 실제/가짜 구별 + 결함 분류 (멀티태스크)
   - 입력: 시계열 데이터 (실제 또는 가짜)
   - 출력: 
     - `type_logits`: 결함 유형 분류 (13개 클래스: 정상 + 12가지 결함)
     - `real_fake_logits`: 실제/가짜 확률

### 훈련 과정

#### 1. Discriminator 훈련
```python
# 실제 데이터로 훈련
real_inputs, fault_labels = data["shot"], data["label"]  # CSV 파일에서 로드
type_logits, fake_logits = netD(real_inputs, None)
errD_type_real = cross_entropy_criterion(type_logits, fault_labels)  # 결함 분류 학습
errD_real = binary_criterion(fake_logits, REAL_LABEL)  # 실제 데이터 판별

# 가짜 데이터로 훈련
fake_inputs = netG(noise, labels)
type_logits, fake_logits = netD(fake_inputs.detach(), None)
errD_fake = binary_criterion(fake_logits, FAKE_LABEL)  # 가짜 데이터 판별
```

#### 2. Generator 훈련
```python
# Generator가 Discriminator를 속이도록 훈련
type_logits, fake_logits = netD(fake_inputs, None)
errG = binary_criterion(fake_logits, REAL_LABEL)  # 가짜를 진짜로 분류하도록

# 실제 데이터와 유사성 추구
errG_similarity = similarity(generated_data, real_inputs)
```

### 핵심 동작 원리

#### 실제 데이터 vs 가짜 데이터의 역할

**실제 데이터 (TEP CSV 파일)**
- **train_fault_0.csv**: 정상 운전 데이터
- **train_fault_1~12.csv**: 12가지 결함 유형 데이터
- **역할**: 실제 결함 분류 학습에 사용 (의미 있는 학습)
- **라벨**: 정확한 결함 유형 라벨 (0-12)

**가짜 데이터 (Generator 생성)**
- **생성 방식**: 랜덤 노이즈 + 랜덤 결함 라벨
- **역할**: 적대적 학습을 통한 robustness 향상
- **라벨**: 랜덤 라벨 (결함 분류 학습에는 의미 없음)

#### Generator가 필요한 이유

1. **적대적 학습의 핵심 메커니즘**
   ```
   Generator: "진짜같은 가짜 데이터" 생성 → Discriminator 속이기 시도
   Discriminator: 진짜와 가짜 구별 → 더 정교한 판별 능력 획득
   ```

2. **결함 탐지 성능 향상**
   - 일반 분류기: 주어진 실제 데이터만으로 학습
   - GAN 분류기: 가짜 데이터와 비교하면서 **진짜 데이터의 미세한 특징**까지 학습
   - 결과: 더 robust하고 정확한 결함 패턴 인식

3. **Overfitting 방지**
   - Generator가 계속 새로운 패턴의 가짜 데이터 생성
   - Discriminator가 다양한 패턴에 노출되어 일반화 성능 향상

### 학습 데이터 구성

```python
# 손실 함수 가중치
real_fake_w_d = 1.0    # 실제/가짜 구별 가중치
fault_type_w_d = 0.8   # 결함 분류 가중치

# 실제 데이터 손실
errD_complex_real = real_fake_w_d * errD_real + fault_type_w_d * errD_type_real
# → 실제/가짜 구별(1.0) + 의미 있는 결함 분류(0.8)

# 가짜 데이터 손실  
errD_complex_fake = real_fake_w_d * errD_fake + fault_type_w_d * errD_type_fake
# → 실제/가짜 구별(1.0) + 의미 없는 결함 분류(0.8)
```

**결론: 실제 의미 있는 결함 분류 학습은 리얼 데이터(CSV 파일)로만 이루어집니다.**

## 전체 모델 아키텍처 비교

### 사용 가능한 모델들

#### 1. Generator 모델들
```python
# 시계열 데이터 생성 모델들
GENERATORS = {
    'LSTMGenerator': '시계열 생성에 특화된 LSTM 기반 생성기',
    'CausalConvGenerator': '인과적 합성곱 기반 생성기 (미래 정보 차단)'
}
```

#### 2. Discriminator/Classifier 모델들
```python
# 결함 분류 및 판별 모델들
DISCRIMINATORS = {
    'CNN1D2DDiscriminatorMultitask': '1D+2D 하이브리드 멀티태스크 (메인 모델)',
    'CausalConvDiscriminatorMultitask': '인과적 합성곱 멀티태스크',
    'CausalConvDiscriminator': '인과적 합성곱 (단일 태스크)',
    'TEPRNN': '순수 RNN 기반 결함 분류기'
}
```

### 버전별 모델 조합

| 버전 | Generator | Discriminator | 특징 | 사용 목적 |
|------|-----------|---------------|------|----------|
| **GAN v5** | LSTM | CNN1D2D | 하이브리드 멀티태스크 | **메인 모델** (최고 성능) |
| **GAN v4** | LSTM | CausalConv | 멀티태스크 | 성능 비교 |
| **GAN v3** | CausalConv | CausalConv | 순수 CNN GAN | 실험/비교 |
| **GAN v2** | ❌ | TEPRNN | 순수 RNN | 베이스라인 |
| **베이스라인** | ❌ | CausalConv | 단순 분류 | 베이스라인 |

### 모델 아키텍처 세부 사항

#### TEPRNN (순수 RNN 모델)
```python
class TEPRNN(nn.Module):
    def __init__(self, seq_size, features_count, class_count, lstm_size):
        self.lstm = nn.LSTM(features_count, lstm_size, batch_first=True)
        self.dense = nn.Linear(lstm_size, class_count)
    
    def forward(self, x, prev_state):
        output, state = self.lstm(x, prev_state)
        logits = self.dense(output)
        return logits, state
```

**특징:**
- 가장 단순한 아키텍처
- 시계열 데이터의 시간적 의존성 캡처
- GAN 없이 순수 결함 분류만 수행

#### CausalConv 모델들
```python
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)  # 미래 정보 제거 (인과성 보장)
```

**특징:**
- **인과성 보장**: 미래 정보 사용하지 않음 (실시간 추론 가능)
- **팽창 합성곱**: 넓은 수용 영역으로 장기 의존성 캡처
- **잔차 연결**: 그래디언트 소실 문제 해결

#### CNN1D2D 하이브리드 모델
```python
class CNN1D2DDiscriminatorMultitask(nn.Module):
    def __init__(self, input_size, n_layers_1d, n_layers_2d, n_channel, n_channel_2d, class_count):
        # 1D CNN for temporal features
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size)
        # 2D CNN for spatial-temporal features
        self.ccn = TemporalConvNet2D(1, n_channel_2d, kernel_size=5)
```

**특징:**
- **1D CNN**: 시간적 패턴 추출
- **2D CNN**: 센서 간 상관관계 + 시간적 패턴 동시 캡처
- **멀티태스크**: 결함 분류 + 실제/가짜 판별

**모든 모델이 구현되어 있으므로 연구 목적이나 성능 비교를 위해 자유롭게 사용할 수 있습니다!**

## GAN v5 모델 평가


```bash
# 학습 완료된 discriminator 모델 평가
python -m src.models.evaluate_model_csv --model_path models/5_main_model/weights/199_epoch_discriminator.pth --cuda 0
```

#### 파라미터 설명
- `--cuda`: 사용할 GPU 번호 (0, 1, 2, ...)
- `--model_path`: 학습된 discriminator 모델 파일 경로
- `--random_seed`: 랜덤 시드 (옵션)
