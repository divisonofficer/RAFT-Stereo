from train_fusion.ssim.utils import *


class SelfLoss:
    def __init__(self, w_iters=0.9, w_warp=0.5, w_smooth=1):
        if not hasattr(self, "ssim"):
            self.ssim = SSIM()
        if not hasattr(self, "_rendering"):
            self._rendering = Rendering()
        self.w_iters = w_iters
        self.w_warp = w_warp
        self.w_smooth = w_smooth

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images"""
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def disocc_detection(self, disp_l, img=None):

        if img is not None:
            img_r_warped, valid_mask = self._rendering(img, disp_l, return_mask=True)
            return valid_mask, img_r_warped
        else:
            disp_r_warped, valid_mask = self._rendering(
                disp_l, disp_l, return_mask=True
            )
            return valid_mask, disp_r_warped

    def compute_losses(self, img_left, img_right, disp_preds):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch"""

        total_loss = 0
        img_left /= 255
        img_right /= 255

        for i, disp in enumerate(disp_preds):
            occ_tensor, img_r_warped = self.disocc_detection(disp, img_left)
            reprojection_loss = self.compute_reprojection_loss(img_r_warped, img_right)

            reprojection_loss *= occ_tensor

            reprojection_loss = reprojection_loss.mean()

            loss = reprojection_loss * self.w_warp

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, img_left) * 1e-0

            loss += smooth_loss * self.w_smooth

            total_loss = torch.add(
                total_loss, loss * (self.w_iters ** (len(disp_preds) - i - 1))
            )

        return total_loss, {"warp_loss": reprojection_loss, "smooth": smooth_loss}

    def warp(self, x, disp, padding_mode="border", mode="bilinear"):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W, device=x.device).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device=x.device).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        vgrid = torch.cat((xx, yy), 1).float()

        if padding_mode == "reflection":
            tmp = vgrid[:, :1, :, :] - disp
            tmp[tmp < 0] = vgrid[:, :1, :, :][tmp < 0] * (-1) - disp[tmp < 0]
            vgrid[:, :1, :, :] = tmp
        else:
            # vgrid = Variable(grid)
            vgrid[:, :1, :, :] = vgrid[:, :1, :, :] - disp

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(
            x, vgrid, padding_mode=padding_mode, mode=mode, align_corners=True
        )

        mask = torch.ones_like(x, requires_grad=False)
        mask = F.grid_sample(mask, vgrid, padding_mode="zeros")
        mask = mask >= 0.999

        output = torch.where(mask, output, output.detach())
        return output
