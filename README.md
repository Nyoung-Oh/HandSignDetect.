## Yolo, MediaPipe, LSTM 모델을 활용한 수어 탐지 웹 서비스 (개인)
<img src="https://github.com/user-attachments/assets/ba3c42a8-c430-46db-bec6-2a348fde3e06"  width="400" height="400">
<img src="https://github.com/user-attachments/assets/d8f96ac2-34e4-4d66-997c-6c5dbcb4b519"  width="400" height="400">

### 🔎 소개 및 목적
- 딥러닝 기술 이용 (LSTM, KNN, Yolo, MediaPipe)
- 웹캠 영상에서 객체 인식 및 수어 인식 구현
### 🔎 개발 기간
2024.07.15 ~ 2024.07.26
### 🔎 사용 언어 및 개발 환경
`Spring Boot Java Web` `JavaScript` `Python` `JSON` `Thyemeleaf` `Servlet` `Eclipse` `PyCharm` `VSCode` `Flask REST Server` `Yolo` `Anaconda` `Jupyter Notebook` `MediaPipe` `Pandas` `KNN` `Numpy` `Tensorflow` `Keras`
### 🔎 구조
<img src="https://github.com/user-attachments/assets/d4585cf5-5a25-4d21-820e-92d2f16f611d"  width="800" height="300">

## 🔎 회고
> - 모델 학습 시 정확도를 높이기 위해Epochs을 100에서 200으로 증가시켰고 학습 이미지 크기를 640에서 1080로 키웠습니다.
> - 변경 후 93%로 정확도가 증가했습니다.
> - 직접 학습 데이터 모델을 만드는 과정에서 다양한 손 각도와 손의 위치 등 여러 상황을 가정해서 학습 모델을 만들어야 인식이 정확하다는 것을 배웠습니다.

## 🔎 주요 기능
### ✔️ 실시간 웹캠 영상 인식
####  ◻   Spring Boot Java Web으로 LSTM 수어 사이트 구현
####  ◻   백엔드 Controller에서 Rest Server Flask로 이미지 전송, 수어를 탐지하고 결과 리턴

### ✔️ 수어 탐지 알파벳 및 정확도 표시 - LSTM, KNN 모델 이용
####   ◻   MediaPipe 이용 손 관절 랜드마크 표시, 특정 좌표를 기준으로 손동작 인식
#####  ◻   LSTM 모델 : 시계열 데이터, 시퀀스 데이터 처리에 적합.
#####  ◻   KNN 알고리즘 : 지도 학습 중 분류 문제에 사용하는 알고리즘. 분류 문제란 새로운 데이터가 들어왔을 때 기존 데이터의 그룹 중 어떤 그룹에 속하는지 분류하는 문제(어느 그룹에 가까운지는 유클리드 거리 계산)
