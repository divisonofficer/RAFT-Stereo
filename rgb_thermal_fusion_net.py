import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback


class RGBThermalFusionNet(nn.Module):
    def __init__(self, iterations=3, hidden_dim=64):
        super(RGBThermalFusionNet, self).__init__()

        self.iterations = iterations
        self.hidden_dim = hidden_dim

        # 특징 추출기 정의 (채널 수를 hidden_dim으로 설정)
        self.rgb_feature_extractor = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.thermal_feature_extractor = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.output_feature_extractor = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # GRU 기반의 업데이트 모듈 정의
        self.update_gru = ConvGRUCell(
            input_dim=hidden_dim * 3, hidden_dim=hidden_dim, kernel_size=3
        )
        self.output_conv = nn.Conv2d(hidden_dim, 3, kernel_size=3, padding=1)

    def forward(self, inputs):
        # inputs: [batch_size, channels, height, width], channels = 3(RGB) + 1(Thermal) = 4
        rgb = inputs[:, :3, :, :] / 255.0  # 0~1로 정규화
        thermal = inputs[:, 3:, :, :] / 255.0  # 0~1로 정규화

        with torch.cuda.amp.autocast(True):
            current_output = rgb.clone()  # 초기 출력은 RGB로 설정
            batch_size, _, height, width = rgb.size()
            hidden_state = torch.zeros(batch_size, self.hidden_dim, height, width).to(
                rgb.device
            )
            for _ in range(self.iterations):
                # 특징 추출
                rgb_features = self.rgb_feature_extractor(rgb)
                thermal_features = self.thermal_feature_extractor(thermal)
                output_features = self.output_feature_extractor(current_output)

                # 특징 결합
                combined_features = torch.cat(
                    [rgb_features, thermal_features, output_features], dim=1
                )

                # GRU를 통한 hidden state 업데이트
                hidden_state = self.update_gru(combined_features, hidden_state)

                # 업데이트를 통한 출력 생성
                update = torch.sigmoid(self.output_conv(hidden_state))
                current_output = current_output + update  # 또는 current_output = update
            return current_output * 255.0  # 0~255로 스케일링하여 반환


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvGRUCell, self).__init__()

        padding = kernel_size // 2

        self.reset_gate = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding
        )
        self.update_gate = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding
        )
        self.out_gate = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding
        )

    def forward(self, x, h):
        # 입력과 hidden state 결합
        combined = torch.cat([x, h], dim=1)

        # 게이트 계산
        reset = torch.sigmoid(self.reset_gate(combined))
        update = torch.sigmoid(self.update_gate(combined))
        combined_new = torch.cat([x, reset * h], dim=1)
        out = torch.tanh(self.out_gate(combined_new))

        # 새로운 hidden state 계산
        new_h = (1 - update) * h + update * out
        return new_h
