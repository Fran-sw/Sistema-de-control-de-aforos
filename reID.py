import cv2
import sys

# Imports
from torchreid.utils import FeatureExtractor


class reID:
    def __init__(self, name, path, input_device):
        """ Start the corresponding re-identification model given its name,path and the divice to use """
        
        self.extractor = None
        
        # Initialize the feature extractor
        self.extractor = FeatureExtractor(
            model_name=name,
            model_path=path,
            device=input_device
        )
    
    def predict(self,image_list):
        """ Given a set of frames, return the features obtained from comparing them """
        
        # get features of the image list
        features = self.extractor(image_list)
        return features
