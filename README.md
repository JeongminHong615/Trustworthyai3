## Assignment 3

본 프로젝트는 STM 기반의 신경망 검증 도구인 Marabou를 사용하여, 외부 모델의 robustness를 형식적으로 검증한다.

### overview

신경망 검증의 목적은 모델의 입력에 작은 변화가 생기더라도 예측 클래스가 변하지 않음을 수학적으로 증명하는 것이다. 본 과제에서는 PyTorch로 학습된 MNIST 분류 모델을 ONNX 포맷으로 변환한 후, $l_{\infty}$-ball 반경 $\epsilon$ 내에서 모델이 항상 올바른 결과를 도출하는지 Marabou를 통해 검증하였다.

### installation

- Marabou 설치  
  git clone https://github.com/NeuralNetworkVerification/Marabou.git  
  cd Marabou
- 재현성 설치  
  pip install -r requirements.txt

### structure

- train_model.py : FCN 모델을 학습하고 ONNX 포맷으로 저장하는 스크립트
- test.py : Marabou를 실행하여 모델의 강건성을 검증하고 결과 출력
- simple_mnist.onnx : 검증 대상인 외부 모델 파일
- result_log.txt : 검증 결과 로그
