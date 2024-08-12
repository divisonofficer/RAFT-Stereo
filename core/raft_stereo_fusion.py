import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock
from core.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from core.extractor_fusion import FusionMultiBasicEncoder
from core.corr import (
    CorrBlock1D,
    PytorchAlternateCorrBlock1D,
    CorrBlockFast1D,
    AlternateCorrBlock,
)
from core.utils.utils import coords_grid, upflow8

autocast = torch.cuda.amp.autocast

from core.fusion import AttentionFeatureFusion, ConcatFusion


class RAFTStereoFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        context_dims = args.hidden_dims

        self.cnet = FusionMultiBasicEncoder(
            output_dim=[args.hidden_dims, context_dims],
            norm_fn=args.context_norm,
            downsample=args.n_downsample,
            fusion_module=self.define_fusion_layer(),
        )
        self.update_block = BasicMultiUpdateBlock(
            self.args, hidden_dims=args.hidden_dims
        )

        self.context_zqr_convs = nn.ModuleList(
            [
                nn.Conv2d(context_dims[i], args.hidden_dims[i] * 3, 3, padding=3 // 2)
                for i in range(self.args.n_gru_layers)
            ]
        )

        self.fusion_fmap = self.define_fusion_layer()(in_channels=256)

        if args.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, "instance", stride=1),
                nn.Conv2d(128, 256, 3, padding=1),
            )
        else:
            self.fnet = BasicEncoder(
                output_dim=256, norm_fn="instance", downsample=args.n_downsample
            )

    def define_fusion_layer(self):
        if self.args.fusion == "AFF":
            return AttentionFeatureFusion
        if self.args.fusion == "ConCat":
            return ConcatFusion

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def rgb2NIR(rgb):
        # Reverse the channels and use torch.maximum
        interm = torch.maximum(rgb, 1 - rgb)[..., ::-1]

        # Compute the weighted sum and apply the power operation
        nir = (
            interm[..., 0] * 0.229 + interm[..., 1] * 0.587 + interm[..., 2] * 0.114
        ) ** (1 / 0.25)

        return nir

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, D, H, W = flow.shape
        factor = 2**self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor * H, factor * W)

    def freeze_raft(self):
        self.freeze_bn()
        self.cnet.freeze_raft()

        if self.args.shared_backbone:
            # self.conv2.eval()
            pass
        else:
            self.fnet.eval()
        # self.context_zqr_convs.eval()
        # self.update_block.eval()

    def extract_feature_map(self, inputs, attention_debug=False):
        image_viz_left, image_viz_right, image_nir_left, image_nir_right = inputs
        if image_nir_left.shape[1] == 1:
            image_nir_left = image_nir_left.repeat(1, 3, 1, 1)
            image_nir_right = image_nir_right.repeat(1, 3, 1, 1)
        if self.args.shared_backbone:
            cnet_output = self.cnet(
                torch.cat((image_viz_left, image_viz_right), dim=0),
                torch.cat((image_nir_left, image_nir_right), dim=0),
                dual_inp=True,
                num_layers=self.args.n_gru_layers,
                attention_debug=attention_debug,
            )
            *cnet_list, x = cnet_output[: (self.args.n_gru_layers + 1)]
            fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0] // 2)

            if attention_debug:
                fmap1, fmap2 = x.split(dim=0, split_size=x.shape[0] // 2)

                fmap1_rgb, fmap2_rgb = cnet_output[-2].split(
                    dim=0, split_size=x.shape[0] // 2
                )
                fmap1_nir, fmap2_nir = cnet_output[-1].split(
                    dim=0, split_size=x.shape[0] // 2
                )
                return (fmap1, fmap2), (fmap1_rgb, fmap2_rgb), (fmap1_nir, fmap2_nir)

        else:
            fmap1, fmap2 = self.fnet([image_viz_left, image_viz_right])
            fmap1_nir, fmap2_nir = self.fnet([image_nir_left, image_nir_right])
            fmap1 = self.fusion_fmap(fmap1, fmap1_nir)
            fmap2 = self.fusion_fmap(fmap2, fmap2_nir)
            cnet_list = self.cnet(image_viz_left, image_nir_left)
        return fmap1, fmap2, cnet_list

    def forward(self, inputs):
        """Estimate optical flow between pair of frames"""

        iters = inputs["iters"]
        flow_init = inputs["flow_init"]
        test_mode = inputs["test_mode"]
        attention_out_mode = (
            inputs["attention_out_mode"] if "attention_out_mode" in inputs else False
        )
        image_viz_left = inputs["image_viz_left"]
        image_viz_right = inputs["image_viz_right"]
        image_viz_left = (2 * (image_viz_left / 255.0) - 1.0).contiguous()
        image_viz_right = (2 * (image_viz_right / 255.0) - 1.0).contiguous()

        heuristic_nir = inputs["heuristic_nir"]
        if heuristic_nir:
            image_nir_left = self.rgb2NIR(image_viz_left)
            image_nir_right = self.rgb2NIR(image_viz_right)
        else:
            image_nir_left = inputs["image_nir_left"]
            image_nir_right = inputs["image_nir_right"]

        image_nir_left = (2 * (image_nir_left / 255.0) - 1.0).contiguous()
        image_nir_right = (2 * (image_nir_right / 255.0) - 1.0).contiguous()

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            if attention_out_mode:
                (fmap1, fmap2), (fmap1_rgb, fmap2_rgb), (fmap1_nir, fmap2_nir) = (
                    self.extract_feature_map(
                        [
                            image_viz_left,
                            image_viz_right,
                            image_nir_left,
                            image_nir_right,
                        ],
                        attention_debug=True,
                    )
                )
                return (fmap1, fmap2), (fmap1_rgb, fmap2_rgb), (fmap1_nir, fmap2_nir)
            fmap1, fmap2, cnet_list = self.extract_feature_map(
                [image_viz_left, image_viz_right, image_nir_left, image_nir_right]
            )

            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]

            # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning
            inp_list = [
                list(conv(i).split(split_size=conv.out_channels // 3, dim=1))
                for i, conv in zip(inp_list, self.context_zqr_convs)
            ]

        if self.args.corr_implementation == "reg":  # Default
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "alt":  # More memory efficient than reg
            corr_block = PytorchAlternateCorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "reg_cuda":  # Faster version of reg
            corr_block = CorrBlockFast1D
        elif self.args.corr_implementation == "alt_cuda":  # Faster version of alt
            corr_block = AlternateCorrBlock

        corr_fn = corr_block(
            fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels
        )

        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume
            flow = coords1 - coords0

            with autocast(enabled=self.args.mixed_precision):
                if (
                    self.args.n_gru_layers == 3 and self.args.slow_fast_gru
                ):  # Update low-res GRU
                    net_list = self.update_block(
                        net_list,
                        inp_list,
                        iter32=True,
                        iter16=False,
                        iter08=False,
                        update=False,
                    )
                if (
                    self.args.n_gru_layers >= 2 and self.args.slow_fast_gru
                ):  # Update low-res GRU and mid-res GRU
                    net_list = self.update_block(
                        net_list,
                        inp_list,
                        iter32=self.args.n_gru_layers == 3,
                        iter16=True,
                        iter08=False,
                        update=False,
                    )
                net_list, up_mask, delta_flow = self.update_block(
                    net_list,
                    inp_list,
                    corr,
                    flow,
                    iter32=self.args.n_gru_layers == 3,
                    iter16=self.args.n_gru_layers >= 2,
                )

            # in stereo mode, project flow onto epipolar
            delta_flow[:, 1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters - 1:
                continue

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:, :1]

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions
