import torch
import torch.nn as nn
import torch.nn.functional as F
from core.extractor import ResidualBlock
from core.fusion import AttentionFeatureFusion


class FusionMultiBasicEncoder(nn.Module):
    def __init__(
        self,
        output_dim=[128],
        norm_fn="batch",
        dropout=0.0,
        downsample=3,
        fusion_module=AttentionFeatureFusion,
        shared_extractor=False,
    ):
        super(FusionMultiBasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample
        self.shared_extractor = shared_extractor

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            self.norm1_2 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(64)
            self.norm1_2 = nn.BatchNorm2d(64)

        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(64)
            self.norm1_2 = nn.InstanceNorm2d(64)

        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm1_2 = nn.Sequential()

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3
        )

        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))

        if not shared_extractor:
            self.conv1_2 = nn.Conv2d(
                3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3
            )
            self.layer1_2 = self._make_layer(64, stride=1)
            self.layer2_2 = self._make_layer(96, stride=1 + (downsample > 1))
            self.layer3_2 = self._make_layer(128, stride=1 + (downsample > 0))

        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)

        self.fusion = fusion_module(128)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1),
            )
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1),
            )
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs32 = nn.ModuleList(output_list)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def freeze_raft(self):
        for layer in [
            self.layer1,
            self.layer2,
            self.layer3,
            self.norm1,
            self.conv1,
            # self.layer4,
            # self.layer5,
            # self.outputs08,
            # self.outputs16,
            # self.outputs32,
        ]:
            for param in layer.parameters():
                param.requires_grad = False

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def modal_forward(self, x, side=0):
        if side == 0:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu1(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        else:
            x = self.conv1_2(x)
            x = self.norm1_2(x)
            x = self.relu1(x)
            x = self.layer1_2(x)
            x = self.layer2_2(x)
            x = self.layer3_2(x)
        return x

    def forward(
        self, x_viz, x_nir, dual_inp=False, num_layers=3, attention_debug=False
    ):
        x_viz = self.modal_forward(x_viz, 0)
        x_nir = self.modal_forward(x_nir, 0 if self.shared_extractor else 1)
        x = self.fusion(x_viz, x_nir)
        if dual_inp:
            v = x
            x = x[: (x.shape[0] // 2)]

        outputs08 = [f(x) for f in self.outputs08]
        output_tupple = (outputs08,)

        if num_layers == 2:
            y = self.layer4(x)
            outputs16 = [f(y) for f in self.outputs16]
            output_tupple += (outputs16,)

        if num_layers == 3:
            z = self.layer5(y)
            outputs32 = [f(z) for f in self.outputs32]
            output_tupple += (outputs32,)

        if dual_inp:
            output_tupple += (v,)

        if attention_debug:
            output_tupple += (
                x_viz,
                x_nir,
            )

        return output_tupple
