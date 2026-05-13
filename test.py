import numpy as np
from maraboupy import Marabou, MarabouCore

def verify_robustness(onnx_path, base_image, target_class, epsilon):
    print(f"목표 클래스 [{target_class}]에 대한 epsilon={epsilon} robustness 검증 시작")

    # 타겟 클래스가 다른 클래스에 대한 반례가 있는지 검사
    for j in range(10):
        if j == target_class:
            continue

        network = Marabou.read_onnx(onnx_path)
        inputVars = [int(v) for v in network.inputVars[0].flatten()]
        outputVars = [int(v) for v in network.outputVars[0].flatten()]

        # 입력 제약 조건
        for i, var in enumerate(inputVars):
            val = float(base_image[i])
            network.setLowerBound(var, max(0.0, val - epsilon))
            network.setUpperBound(var, min(1.0, val + epsilon))

        # 출력 제약 조건
        network.addInequality(
            [outputVars[target_class], outputVars[j]], 
            [1.0, -1.0], 
            0.0
        )

        # Marabou Solve 실행
        options = Marabou.createOptions(verbosity=0) 
        results = network.solve(options=options)
        
        if len(results) == 2:
            vals, stats = results
        elif len(results) == 3:
            exit_code, vals, stats = results
        else:
            vals = results[0]

        if vals and len(vals) > 0:
            print(f"SAT")
            print(f"epsilon {epsilon} 범위 내에서 모델이 숫자 {target_class}를 {j}로 오분류할 수 있다.")
            return

    print(f"UNSAT")
    print(f"epsilon {epsilon} 범위 내에서 입력값이 주어질 때, 모델은 항상 {target_class}로 올바르게 예측")

if __name__ == "__main__":
    dummy_image = np.full((28 * 28,), 0.5, dtype=np.float32)
    
    verify_robustness("simple_mnist.onnx", dummy_image, target_class=3, epsilon=0.1)