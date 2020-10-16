import torch
from PIL import Image

from .solver import Solver
from .preprocess import PreProcess


class Inference:
    """
    An inference wrapper for makeup transfer.
    It takes two image `source` and `reference` in,
    and transfers the makeup of reference to source.
    """
    def __init__(self, config, device="cpu", model_path="assets/models/G.pth"):
        """
        Args:
            device (str): Device type and index, such as "cpu" or "cuda:2".
            device_id (int): Specefying which devide index
                will be used for inference.
        """
        self.device = device
        self.solver = Solver(config, device, inference=model_path)
        self.preprocess = PreProcess(config, device)

    def transfer(self, source: Image, reference: Image, with_face=False, transfer_lips=True, transfer_skin=True, transfer_eyes=True):
        """
        Args:
            source (Image): The image where makeup will be transfered to.
            reference (Image): Image containing targeted makeup.
        Return:
            Image: Transfered image.
        """
        source_input, face, crop_face = self.preprocess(source)
        reference_input, _, _ = self.preprocess(reference)
        if not (source_input and reference_input):
            if with_face:
                return None, None
            return

        for i in range(len(source_input)):
            source_input[i] = source_input[i].to(self.device)
            # print('source_input', source_input[i].shape)

        for i in range(len(reference_input)):
            reference_input[i] = reference_input[i].to(self.device)
            # print('reference_input', reference_input[i].shape)

        # TODO: Abridge the parameter list.
        result = self.solver.test(*source_input, *reference_input, transfer_lips, transfer_skin, transfer_eyes)
        
        if with_face:
            return result, crop_face, source_input[1]
        return result
