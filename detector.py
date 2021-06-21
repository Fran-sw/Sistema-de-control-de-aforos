import sys
import numpy as np
import os, json, cv2, random
import cv2

# Imports for yolo
if True:
    from YOLOv4.tool.utils import *
    from YOLOv4.tool.torch_utils import *
    from YOLOv4.tool.darknet2pytorch import Darknet

# Imports for detectron
if True:
    import detectron2
    from detectron2.utils.logger import setup_logger
    setup_logger()
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog, DatasetCatalog



class detector:
    def __init__(self, detector_name):
        """ Start the corresponding detector given its name """
        
        self.net = None
        self.using = None
        self.use_cuda = True
        
        # Check wich detector to use based on name and start the prediction net
        # YOLOv4
        if detector_name == "YOLOv4":
            print("Starting YOLOv4")
        
            self.net = Darknet('./YOLOv4/cfg/yolov4_only_persons.cfg')
            self.net.load_weights('./YOLOv4/cfg/yolov4.weights')
            self.using = detector_name
            
            if self.use_cuda:
                self.net.cuda()
                
        elif detector_name == "detectron2":
            print("Starting detectron2")
            self.using = detector_name
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
            
            # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            self.net = DefaultPredictor(cfg)
            
        else:
            print("ERROR:No detector known with that name")
            sys.exit()
    
    def predict(self,frame):
        """ Given a frame, detect the bounding boxes of the diferent people in frame, and if the detector supports it, the mask of them"""
        
        # YOLOv4
        if self.using == "YOLOv4":
        
            sized = cv2.resize(frame, (self.net.width, self.net.height))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
            
            width = frame.shape[1]
            height = frame.shape[0]
          
            # Get the boxes
            boxes = do_detect(self.net, sized, 0.4, 0.6, self.use_cuda)
            boxes_aux = []
            for box in boxes[0]:
                box[0] = int(box[0] * width)
                box[1] = int(box[1] * height)
                box[2] = int(box[2] * width)
                box[3] = int(box[3] * height)
                boxes_aux.append(box)

            return boxes_aux,[]
            
        elif self.using == "detectron2":
            # Check if resize
            outputs = self.net(frame)
            
            # Get only the persons
            instances_personas = outputs['instances'][outputs['instances'].pred_classes == 0]

            # Boxes
            boxes= instances_personas.pred_boxes

            # Masks
            masks = instances_personas.pred_masks
            
            return boxes, masks

