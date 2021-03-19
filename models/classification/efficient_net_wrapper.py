from efficientnet_pytorch import EfficientNet
import utils.logging

"""Wrapper on top of efficientnet model implementation
"""
class EfficientNetWrapper:
    def __init__(self, model_name:str):
        self.model = EfficientNet.from_pretrained(model_name)
    
        