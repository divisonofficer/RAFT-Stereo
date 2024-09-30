import torch


def get_depth_anything_model():
    import sys

    sys.path.append("Depth_Anything")
    import os
    from Depth_Anything.depth_anything.dpt import DepthAnything

    os.chdir("Depth_Anything")

    depth_anything = (
        DepthAnything.from_pretrained("LiheYoung/depth_anything_vits14").cuda().eval()
    )
    os.chdir("..")

    def get_depth_by_anything(image: torch.Tensor) -> torch.Tensor:
        _, _, H, W = image.shape
        patch = 14
        image = torch.nn.functional.pad(
            image, (0, patch - W % patch, 0, patch - H % patch), mode="reflect"
        )
        image = image / 255.0

        with torch.no_grad():
            image = image.cuda()
            depth = depth_anything(image)
        depth = depth[:, :H, :W]
        return depth

    return get_depth_by_anything
