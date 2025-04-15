from typing import Optional
import torch


default_dict = {
    "test_mode": False,
    "flow_init": None,
    "heuristic_nir": False,
    "attention_out_mode": False,
    "spectral_feature": False,
}


class TrainInput:

    def __init__(
        self,
        data_dict={},
    ):
        for key, value in default_dict.items():
            if key not in data_dict:
                data_dict[key] = value
        self.data_dict = data_dict

    @classmethod
    def from_image_tuple(cls, input_tuple):
        return cls(
            {
                "image_viz_left": input_tuple[0],
                "image_viz_right": input_tuple[1],
                "image_nir_left": input_tuple[2],
                "image_nir_right": input_tuple[3],
            }
        )

    @property
    def image_viz_left(self) -> torch.Tensor:
        return self.data_dict["image_viz_left"]

    @image_viz_left.setter
    def image_viz_left(self, value):
        self.data_dict["image_viz_left"] = value

    @property
    def image_viz_right(self) -> torch.Tensor:
        return self.data_dict["image_viz_right"]

    @image_viz_right.setter
    def image_viz_right(self, value):
        self.data_dict["image_viz_right"] = value

    @property
    def image_nir_left(self):
        return self.data_dict["image_nir_left"]

    @image_nir_left.setter
    def image_nir_left(self, value):
        self.data_dict["image_nir_left"] = value

    @property
    def image_nir_right(self):
        return self.data_dict["image_nir_right"]

    @image_nir_right.setter
    def image_nir_right(self, value):
        self.data_dict["image_nir_right"] = value

    @property
    def iters(self):
        return self.data_dict["iters"]

    @iters.setter
    def iters(self, value):
        self.data_dict["iters"] = value

    @property
    def test_mode(self):
        return self.data_dict["test_mode"]

    @test_mode.setter
    def test_mode(self, value):
        self.data_dict["test_mode"] = value

    @property
    def flow_init(self):
        return self.data_dict["flow_init"]

    @flow_init.setter
    def flow_init(self, value):
        self.data_dict["flow_init"] = value

    @property
    def heuristic_nir(self):
        return self.data_dict["heuristic_nir"]

    @heuristic_nir.setter
    def heuristic_nir(self, value):
        self.data_dict["heuristic_nir"] = value

    @property
    def attention_out_mode(self):
        return self.data_dict["attention_out_mode"]

    @attention_out_mode.setter
    def attention_out_mode(self, value):
        self.data_dict["attention_out_mode"] = value
        
    @property
    def spectral_feature(self):
        return self.data_dict["spectral_feature"]
    
    @spectral_feature.setter
    def spectral_feature(self, value: bool):
        self.data_dict["spectral_feature"] = value
