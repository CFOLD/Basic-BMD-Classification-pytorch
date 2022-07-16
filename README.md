# Basic-BMD-Classification-pytorch

## 설치
1. python>=3.6, pytorch>=1.10, torchvision>=0.11
2. git clone https://github.com/CFOLD/Basic-BMD-Classification-pytorch bmd
3. cd bmd
4. pip install -r requirements.txt


## 학습 데이터셋 구성
1. 학습 데이터셋은 images 폴더와 label.txt로 구성됩니다.
```
train - images  
        label.txt
```
2. images 폴더에는 학습할 뼈 단위의 이미지 파일이 위치해야 합니다.
3. label.txt는 파일명,클래스 형식으로 작성됩니다. 예를 들어서 파일명이 00001.png, 클래스 0 / 00002.png, 클래스 1이면  
(클래스명은 반드시 정수여야 합니다)
```
   label.txt - 00001.png,0  
               00002.png,1
```
   과 같은 형태로 작성되어야 합니다.


## 테스트 데이터셋 구성
1. 테스트 데이터셋은 Unknown 폴더로 이루어집니다. (폴더명 무관)
```
test - Unknown
```
2. Unknown 폴더에는 테스트할 뼈 단위의 이미지 파일이 위치해야 합니다.


## 학습
```bash
python train.py --batch-size 64 --cos-lr --roc-curve
```

**사용 가능한 인자(arguments)**  
--weights '모델 경로' (기본값 ROOT / best.pth)  
--dataset '데이터셋 경로' (기본값 ROOT / dataset / train)  
--val-ratio 숫자(0-1) : 학습 데이터셋에서 검증 데이터의 분할 비율 (기본값 0.1)  
--epochs 숫자(정수) (기본값 100)  
--imgsz 숫자(정수) : 학습할 이미지 크기 (기본값 224)  
--batch-size 숫자(정수) (기본값 32)  
--lr 숫자 : learning rate (기본값 1e-5)  
--save-checkpoint 숫자(정수) : 모델 저장 주기 (기본값 1)  
--cos-lr : cosine LR scheduler 활성화  
--roc-curve : ROC Curve 저장 활성화


## 테스트
```bash
python test.py
```

**사용 가능한 인자(arguments)**  
--weights '모델 경로' (기본값 ROOT / best.pth)  
--dataset '데이터셋 경로' (기본값 ROOT / dataset / test)  
--imgsz 숫자(정수) : 학습할 이미지 크기 (기본값 224)  
--batch-size 숫자(정수) (기본값 32)
