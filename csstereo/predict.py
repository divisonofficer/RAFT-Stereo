import torch

from csstereo.dpn import DPN


def predict(dpnet, input_left, input_right):
    dpnet.eval()
    print(input_left.shape, input_right.shape)

    # 전처리: 0~1로 정규화 후 GPU로 이동
    input_left = torch.clamp(input_left / 255.0, 0.0, 1.0).cuda()
    input_right = torch.clamp(input_right / 255.0, 0.0, 1.0).cuda()

    with torch.no_grad():
        # DPN을 사용한 깊이 추정
        ldisps, rdisps = dpnet(input_left, input_right)

        return ldisps[0] * 1024


def get_csstereo_model():
    # 모델 초기화
    ckpt_path = "csstereo_pretrained.pth"  # 체크포인트 파일 경로 설정

    dpnet = DPN(in_shape=(540, 720))

    checkpoint = torch.load(ckpt_path)
    dpnet.load_state_dict(checkpoint["dpnet"])

    dpnet = dpnet.cuda()

    return lambda x, y: predict(dpnet, x, y)
