import os
import onnxruntime
import torch
import numpy as np
from thop import profile
from emonet.models import EmoNet

if not os.path.isfile("emonet.onnx"):

    state_dict = torch.load('pretrained/emonet_8.pth', map_location='cpu')
    state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
    net = EmoNet(n_expression=8).to('cpu')
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    # 모델 입력 텐서 생성
    dummy_input = torch.randn(1, 3, 256, 256)

    # FLOPs 계산
    flops, params = profile(net, inputs=(dummy_input,))
    print(f"FLOPs: {flops}") # 1.7e+10
    print(f"Parameters: {params}") # 1.39e+7

    # ONNX 변환
    onnx_path = "/Users/dkcns/PycharmProjects/emonet/emonet.onnx"
    torch.onnx.export(net, dummy_input, onnx_path, verbose=True)

    # 모델 변환
    output_names = ['embedding', 'expression', 'valence', 'arousal']
    torch.onnx.export(net,  # 실행될 모델
                      dummy_input,  # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                      "/Users/dkcns/PycharmProjects/emonet/emonet.onnx",  # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                      export_params=True,  # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                      opset_version=11,  # 모델을 변환할 때 사용할 ONNX 버전
                      do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                      input_names=['input'],  # 모델의 입력값을 가리키는 이름
                      output_names=output_names,  # 모델의 출력값을 가리키는 이름
                      verbose=True
    )

# ONNX 모델 로드
onnx_model_path = "emonet.onnx"
session = onnxruntime.InferenceSession(onnx_model_path)

# 입력 텐서 생성
input_name = session.get_inputs()[0].name
input_data = np.zeros((1,3,256,256))
input_data = input_data.astype(np.float32)

# 예측 수행
output_names = [output.name for output in session.get_outputs()]
outputs = session.run(output_names, {input_name: input_data})
