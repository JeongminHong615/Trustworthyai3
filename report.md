# Assignment3 : Marabou Verification Report

### Structure

- Model : timeout을 방지하기 위해 Pytorch를 사용하여 입력층-은닉층-출력층으로 구성된 단순한 형태의 Fully Connected Network(FCN)를 설계하였다. 학습 속도와 가독성을 위해 100배치 단위로 손실값을 기록하며 1 epoch만 학습시킨 후 ONNX 포맷으로 변환하였다.
- Dataset :
