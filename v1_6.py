from os.path import isfile, join
import matplotlib.pyplot as plt
from collections import Counter
from copy import deepcopy
from os import listdir
import numpy as np
import argparse
import json
import time
import cv2
import sys
import os

from scipy import spatial
import tensorflow as tf
from detector import *
from reID import *
import torch

from show_gt import *

# Global variables
path_videos = []
annotations = []
frame_names = []
path_cameras = []
num_people = 0
num_frame = 0

iou_threshold = 0.5
cosine_threshold = 0.6

error = 0
evaluate = 0

# PREDICTOR
false_negatives_p = 0
false_positives_p = 0
true_positives_p = 0

# RE-ID
bad_reid = 0
good_reid = 0
bad_matching_total = 0
        
precision_plot = []
recall_plot = []

verbose = 0
show_time = 0
abstract_detector = None
abstract_reID = None

# Red boxes -> Annotations not predicted 
# Blue boxes -> Predictions
# Green boxes -> Annotations that are predicted well

def crop_only_person(im, masker):

    def calculateDistance(x1,y1,x2,y2):  
        dist = math.hypot(x2 - x1, y2 - y1)
        return dist
        
    def get_cropped(rotrect,box,image):
        
        width = int(rotrect[1][0])
        height = int(rotrect[1][1])

        src_pts = box.astype("float32")
        # corrdinate of the points in box points after the rectangle has been
        # straightened
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(image, M, (width, height))
        return warped
        
    mask_out = cv2.bitwise_and(im, im, mask=masker)
    contours, hierarchy = cv2.findContours(masker.copy() ,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[np.argmax([cv2.contourArea(x) for x in contours])]
    rotrect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rotrect)
    box = np.int0(box)
    
    masked = get_cropped(rotrect,box,mask_out)
    return masked

def process_evaluation(frame_annotations, frames, boxes_sets, id_sets, num_camaras, aux_verbose):
    global verbose,num_frame, total_people, verbose, iou_threshold, bad_matching,false_negatives_p,false_positives_p,true_positives_p,bad_reid,good_reid,bad_matching_total

    # Variables to keep track of % of people views matched correctly, and total number of views evaluated
    people_percentage = []
    total_people_views = []
    saved_person = 0
    f = open("./documentacion/evaluacion.txt", "a")
    f.write("*** FRAME " + str(num_frame) + " ***\n")
    
    # PREDICTOR
    false_negatives_p = 0
    false_positives_p = 0
    true_positives_p = 0
    
    # RE-ID
    bad_reid_annotations = 0
    good_reid_annotations = 0
    bad_matching_total_persons = 0
        
    width = frames[0].shape[1]
    height = frames[0].shape[0]
    
    boxes_aux = deepcopy(boxes_sets)
    id_aux = deepcopy(id_sets)
    
    # For each person on annotations we will evaluate the predicted boxes and get the ID's associated for post reid evaluation
    
    for persona in frame_annotations:
    
        id_reid_camera = [-1] * num_camaras
        annotations_camera = [False] * num_camaras
        
        
        id_p = persona["personID"]
        views = persona["views"]
        
        for i in range(num_camaras):
        
            # For each person in the especific frame annotation we take its x,y of the i camera
            values = views[i]
        
            xmax = values["xmax"]
            xmin = values["xmin"]
            ymax = values["ymax"]
            ymin = values["ymin"]
            
            if xmax != -1:
                annotations_camera[i] = True
            
            # Get boxes and ids of camera i
            boxes_i_camera = boxes_aux[i]
            ids_i_camera = id_aux[i]
            num_person = 0
            index_max = 0
            iou_max = 0
            # Check witch iou is the best one
            for box in boxes_i_camera:
                xmin_p = int(box[0])
                ymin_p = int(box[1])
                xmax_p = int(box[2])
                ymax_p = int(box[3])
                
                # Evaluate 
                iou = evaluate_IOU([ymin,xmin,ymax,xmax],[ymin_p,xmin_p,ymax_p,xmax_p])
           
                if iou > iou_max:
                    iou_max = iou
                    index_max = num_person
                    
                num_person = num_person + 1
            
            # Obtain the max iou out of the boxes and if its better than a threshold -> count it as right    
            if iou_max > iou_threshold:
                true_positives_p += 1
                id_reid_camera[i] = ids_i_camera[index_max]
                # Draw annotation near the box if verbose
                if verbose:
                    cv2.rectangle(frames[i], (xmax, ymax), (xmin, ymin),(18, 236, 18), 2)
                    y = ymax - 15 if ymax - 15 > 15 else ymax + 15
                    cv2.putText(frames[i], str(id_p), (xmin+15, ymin+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (18, 236, 18) , 1)
                        
                # Delete the selected box and id
                del ids_i_camera[index_max]
                del boxes_i_camera[index_max]
                
            elif annotations_camera[i]:
                # Should have been detected but nope
                false_negatives_p += 1
                if verbose:
                    cv2.rectangle(frames[i], (xmax, ymax), (xmin, ymin),(18, 18, 236), 2)
            

        # EVALUATE REID
        
        ids_for_person = []
        num_cameras_should_be = 0
        
        
        for i in range(num_camaras):
            if annotations_camera[i] and id_reid_camera[i] > -1:
                # Should be detected and should have an id associated
                ids_for_person.append(id_reid_camera[i])
            elif annotations_camera[i]:
                num_cameras_should_be += 1
        
        total_people_views.append(len(ids_for_person) - 1)
        
        # Select the id given for that person by id (the most repeated one among all of them)
        if len(ids_for_person) > 1:
            occurence_count = Counter(ids_for_person)
            id_given = occurence_count.most_common(1)[0][0]
            pars_ok = 0
            pars_not_ok = 0
            
            # Else, new person, lets check how manytimes we give him other id 
            for element in ids_for_person:
                if element == id_given:
                    pars_ok += 1
                else:
                    pars_not_ok += 1
             
            if pars_ok >= 1:
                good_reid_annotations = good_reid_annotations + (pars_ok - 1)
                people_percentage.append((pars_ok - 1)/(len(ids_for_person) - 1))
            else:
                people_percentage.append("Nan")
            if pars_not_ok >= 1:
                bad_reid_annotations = bad_reid_annotations + (pars_not_ok)

            #print("id_p : ",id_p)
            #print("ids_yolo : ",ids_for_person)
            #print("annotations : ",annotations_camera)
            #print("pars_ok : ",pars_ok)
            #print("pars_not_ok : ",pars_not_ok)
            #print()
         
        else:
            people_percentage.append("Nan")

            # Check if all ids have been matched the same -> diferent Ids means an error in the reid
            if len(set(ids_for_person)) != 1:
                bad_matching_total_persons = bad_matching_total_persons + 1
        
        f.write("Person " + str(saved_person) + " : \n")
        f.write("Total pairs evaluated " + str(total_people_views[saved_person]) + "\n")
        f.write("Percentage of correct " + str(people_percentage[saved_person]) + "\n")
        f.write("---------------------------\n")
        
        saved_person += 1
           
    
    acum = 0
    for box_set in boxes_sets:
        acum += len(box_set) 
    false_positives_p = false_positives_p + (acum - true_positives_p)
    
    bad_reid += bad_reid_annotations
    good_reid += good_reid_annotations
    bad_matching_total += bad_matching_total_persons
    
    
    
    if verbose or aux_verbose:
        print("Evaluation of predictor")
        print("False negatives (boxes not detected and existent on annotations): ", false_negatives_p)
        print("True positives (boxes predicted that match with annotations): ", true_positives_p)
        print("False positives (boxes not on annotations and predcited) : ", false_positives_p)
        print()
        print("Evaluation of reid")
        print("Number of reid pairs views matched correctly : ", good_reid_annotations)
        print("Number of reid pairs views matched incorrectly : ", bad_reid_annotations)
        print("Badly matched persons on annotations totally : ", bad_matching_total_persons)
        print()
    
    f.close()  
    num_frame += 1    
            
            
def process_reID(croped,num_cameras):
    """ Function in charge of applying reIdentification model and post processing to determined the amount of people on frame from all angles
        It takes the diferent sets of cropped boxes from each camera's frame and the amount of cameras as parameters"""
    global abstract_reID
    # Re-Id with the frames and the boxes drawn on them
    # Features of each set of boxes per frame
    features = []
    
    # Lists of id's associated to each box
    id_sets = []
    cos_sets_max= []
    
       
    # For each set of cropped imgs of boxes(1 set per frame) get their features
    i = 0
    for croped_set in croped:
    
        if croped_set != []:
            features.append(abstract_reID.predict(croped_set))
        else:
            features.append([])
            
        ids = [None] * len(croped_set)
        id_sets.append(ids)
        
        cos = [0] * len(croped_set)
        cos_sets_max.append(cos)

        
 
    # Check features to determine how many people are in frame
    # For each set of features boxes of each camera
    num_camera = 0
    num_people = 0
    for feature_set in features:
        ids_camera = id_sets[num_camera]
        
        persona = 0
        # For each person in that camera - compare with the rest of cameras -> if there is any matching delete that person from the freature set of the other camera
        for person_feature in feature_set:
            
            aux_i = num_camera + 1
            
            # Give the new person an id or reID it
            if ids_camera[persona] == None:
                num_people = num_people + 1
                ids_camera[persona] = num_people
            else:
                aux_i = num_cameras
            
            # Person not detected previously, check if you can see it anywhere else
            while aux_i < num_cameras:
                d_max = 0
                i_max = 0
                id_aux = id_sets[aux_i]
                compared_set = features[aux_i]
                cos_set = cos_sets_max[aux_i]
                
                persona_2 = 0
                for compared_person in compared_set:
                    # Calculate cosine distance between tensors if we havent reid this person yet
                    #if id_aux[persona_2] == None:
                    distance = 1 - spatial.distance.cosine(compared_person.cpu().numpy(), person_feature.cpu().numpy())
                    if distance > d_max:
                        d_max = distance
                        i_max = persona_2
                    
                    persona_2 = persona_2 + 1
                
                # if we found someone that looks the same - set his id as the same
                if d_max > cosine_threshold and d_max > cos_set[i_max]:
                    id_aux[i_max] = num_people
                
                aux_i = aux_i + 1
            persona = persona + 1
        num_camera = num_camera + 1
    
    # For each box, give an id
    return num_people, id_sets


def draw_predictions(frame_list, boxes_sets, id_sets):
    """Draws the boundign boxes of the detector and the ids of the reid"""
    width = frame_list[0].shape[1]
    height = frame_list[0].shape[0]
    num_camera = 0
    
    for frame in frame_list:
    
        boxes = boxes_sets[num_camera]
        ids = id_sets[num_camera]
        person = 0
        
        for box in boxes:
        
            xmin_p = max(int(box[0]),1)
            ymin_p = max(int(box[1]),1)
            xmax_p = min(int(box[2]),width)
            ymax_p = min(int(box[3]),height)
            cv2.rectangle(frame, (xmax_p, ymax_p), (xmin_p, ymin_p),
                        (236, 18, 18), 2)
            
            cv2.putText(frame, str(ids[person]), (xmin_p+15, ymin_p+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (18, 236, 18) , 2)
                    
            person = person + 1
        num_camera = num_camera + 1

def process_frame(frame_list, frame_annotations):
    """ Given a set of frame i from the set of cameras, and the predictions of it (and optionally the annotations) draw each person detected and compute error """
    global evaluate, verbose, error, num_people, abstract_detector, abstract_reID, precision_plot,recall_plot

    # Clean CUDA cache
    torch.cuda.empty_cache()
    
    # Variable to keep track of number of camera
    processed_camera = 0
    
    # People detected on this frame
    detected = []
    boxes_sets = []
    total = max(len(frame_annotations),1)
    
    croped = []
    
    # Get Bounding boxes for each person in the diferente frames
    for frame in frame_list:
    
        croped_i = []
        width = frame.shape[1]
        height = frame.shape[0]
        
        # Apply predictor to get boxes
        #start = time.time()
        boxes, masks = abstract_detector.predict(frame)
        #end = time.time()
        #print("PREDICT process time ", "{:.4f}".format(end - start), " seconds")
        
        boxes_aux = []
        i = 0
        # Process predictions - get cropped parts tp reid
        for box in boxes:

            xmin_p = max(int(box[0]),1)
            ymin_p = max(int(box[1]),1)
            xmax_p = min(int(box[2]),width)
            ymax_p = min(int(box[3]),height)
            
            #print(" x_max = ",xmax_p," x_min = ",xmin_p," y_max = ",ymax_p,"y_min = ",ymin_p)

            # Save copped boxes for later reid per frame
            croped_img = frame[ymin_p:ymax_p, xmin_p:xmax_p]
            
            # Check if detected box is big enough
            area = (ymax_p - ymin_p + 1) * (xmax_p - xmin_p + 1)
            if area > 500:
                if masks != []:
                    croped_img = crop_only_person(frame, masks[i].to("cpu").numpy().astype(np.uint8))


                croped_i.append(croped_img)
                boxes_aux.append(box)
                
            i = i + 1
            
        boxes_sets.append(boxes_aux)
        croped.append(croped_i)
        croped_i = []
        
        processed_camera = processed_camera + 1
        
    
    # Re-Id with the frames and the boxes drawn on them
    # Features of each set of boxes per frame
    #start = time.time()
    num_people, id_sets = process_reID(croped,processed_camera)
    #end = time.time()
    #print("REID process time ", "{:.4f}".format(end - start), " seconds")
    
    # Draw predictions 
    draw_predictions(frame_list, boxes_sets, id_sets)
            
    # Update i-frame error if we need to evaluate
    if evaluate:
        process_evaluation(frame_annotations, frame_list, boxes_sets, id_sets, processed_camera,0)
        detected = set(detected)
        
        # Calculate precision and recall
        precision = true_positives_p / (true_positives_p + false_positives_p)
        recall = true_positives_p / (true_positives_p + false_negatives_p)
        
        precision_plot.append(precision)
        recall_plot.append(recall)
        #error = 1 - (len(detected)/total)
        
        # Verbose info    
        if verbose:
            print("People detected with reId: ", num_people)
            print("Total in frame annotation : ",total )
            
            print("Precision : ", precision)
            print("Recall : ", recall)
            print("-----------------------------------------------------")
        
    
    
# IOU (intersection over union) = area of overlap / area of union      
def evaluate_IOU(boxA, boxB):
 
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

def initiate_camera_paths(directory,use_video):
    """ Given a directory where the camera frames are stored in folders, read each forlder name and store it """
    global path_cameras, frame_names, path_videos
    if use_video == 0:
        directory = directory.replace("'", "")
        # Get camera paths to read frame imgs
        aux = 0
        for f in listdir(directory):
            file_path = directory + '/' + f
            path_cameras.append(file_path)
            aux = f
        
        # Get frame names for later use - WE SUPOSE ALL CAMERAS HAVE THE SAME NUMBER OF FRAMES AND WITH THE SAME NAME
        directory = directory + '/' + aux
        for frame_name in listdir(directory):
            frame_names.append(frame_name.replace(".png", ""))
            
    else:
        for f in listdir(directory):
            file_path = directory + '/' + f
            path_videos.append(file_path)

def show_frames(num_exit):
    """ For each frame of the cameras (suposing all cameras have the same frames - lenght of videos is equal)  show results for them """
    global annotations, frame_names, path_cameras,error, evaluate, num_people,verbose, precision_plot,recall_plot
    
    num_camaras = len(path_cameras)
    num_frames_procesed = 0
    
    #def split_list(a_list):
    #   half = len(a_list)//2
    #   return a_list[:half], a_list[half:]

    #_, frame_names = split_list(frame_names)
    
    num_personas_lapse_time = []
    
    for frame_name in frame_names:
        frame_list = []
        
        #if int(frame_name) > 2500 and int(frame_name) < 4000:
        # Get frame i of each camera
        for i in range(num_camaras):
            path_frame = path_cameras[i] + "/" + frame_name + ".png"
            img = cv2.imread(path_frame)
            frame_list.append(img)
            
        # Process the frame i taking into account all camera angles
        if evaluate == 0:
            # We do not have annotations
            process_frame(frame_list, [])
        else:
            # We have annotations
            process_frame(frame_list,  annotations[num_frames_procesed])
           
        # Show results for frame i
            
        for i in range(num_camaras): #!!!!! Remember to change this to show all processed frames
            name = "Camera: " + str(i)
            cv2.putText(frame_list[i], 'NUMBER OF PEOPLE: ' + str(num_people), (50, 60), cv2.FONT_HERSHEY_SIMPLEX , 2, (0, 255, 255), 2, cv2.LINE_4)
            #if not evaluate: 
                #cv2.putText(frame_list[i], 'Error: ' + "Unknown", (50, 80), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 255), 2, cv2.LINE_4)
            #else:
                #cv2.putText(frame_list[i], 'Error: ' + "{:.4f}".format(error), (50, 80), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 255), 2, cv2.LINE_4)
   
            #cv2.imshow(name,frame_list[i])
            cv2.imwrite("results/Camara_" + str(i) +"/" + frame_name + ".png" ,frame_list[i]) # EINA dataset
            #cv2.imwrite("results/C" + str(i) +"/" + frame_name + ".png" ,frame_list[i])         # Wildtrack dataset
        
        #sys.exit()
        print("Number of frame processed: ",frame_name)    
        
        num_frames_procesed = num_frames_procesed + 1
        num_personas_lapse_time.append(num_people)
        num_people = 0
        #cv2.waitKey(10)
        
        if num_frames_procesed > num_exit:
            break
    
    print(num_personas_lapse_time)
    if verbose and evaluate:
        # Plot Precision and Recall
        # x_frames = [i for i in range(num_frames_procesed)]
        # plt.plot(x_frames,precision_plot, label = "precision")
        # plt.plot(x_frames,recall_plot, label = "recall")

        # plt.xlabel('num_frames')
        # plt.ylabel('precision/recall')
        
        # plt.title('Graph precision/recall')
        # plt.legend()
        # plt.show()
        mean_p = 0
        mean_r = 0
        for precision in precision_plot:
            mean_p += precision
            
        for recall in recall_plot:
            mean_r += recall
        print ("mean precision: ", mean_p/len(precision_plot))
        print ("mean recall: ", mean_r/len(recall_plot))
        
        print("Total number of pair reid predictions matched correctly : ", good_reid)
        print("Total number of pair reid predictions matched incorrectly : ", bad_reid)
        print("Badly matched persons on annotations totally : ", bad_matching_total)
    
def show_videos():
    global path_videos, error, num_people, show_time
    
    num_camaras = len(path_videos)
    
    cap = [cv2.VideoCapture(i) for i in path_videos]
   
    frames = [None] * num_camaras
    gray = [None] * num_camaras
    ret = [None] * num_camaras
    
    # Variables for writing videos with the results
    video_writers = []
    
    # For aux img with info
    #out = cv2.VideoWriter('aux_info.avi', cv2.VideoWriter_fourcc(*'MJPG'),30, (720,480))
    
    i = 0
    for video in cap:
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))
        size = (frame_width, frame_height)
        #video_writers.append(cv2.VideoWriter('Camara_' + str(i) + '_results.avi', cv2.VideoWriter_fourcc(*'MJPG'),30, size))
        i += 1
    
    num_frames_procesados = 0
    while num_frames_procesados != 7500: #True:
    
        #start_time = time.time()

        for i,c in enumerate(cap):
            if c is not None:
                ret[i], frames[i] = c.read()
        
        for i,f in enumerate(frames):
            if ret[i] is True:
                gray[i] = f

        if show_time:
            start = time.time()
            process_frame(gray, [])
            end = time.time()
            print("Runtime of the program is ", "{:.4f}".format(end - start), " seconds")
            
        else:
            process_frame(gray, [])
            
        # For aux img with info   
        #blank_image = np.zeros((480,720,3), np.uint8)    
        #cv2.putText(blank_image, 'NUM FRAME: ' + str(num_frames_procesados), (175, 200), cv2.FONT_HERSHEY_SIMPLEX , 0.7, (0, 255, 255), 3, cv2.LINE_AA)
        #cv2.putText(blank_image, 'NUMBER OF PEOPLE: ' + str(num_people), (175, 250), cv2.FONT_HERSHEY_SIMPLEX , 0.7, (0, 255, 255), 3, cv2.LINE_AA)
        #out.write(blank_image)
            
        for i in range(num_camaras):
            cv2.putText(gray[i], 'NUMBER OF PEOPLE: ' + str(num_people), (60, 60), cv2.FONT_HERSHEY_SIMPLEX , 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            #cv2.imshow(path_videos[i], gray[i])
            #video_writers[i].write(gray[i])
        num_frames_procesados += 1
        
        cv2.waitKey(1)
        #print("--- %s seconds ---" % (time.time() - start_time))

    for c in cap:
        if c is not None:
            c.release();

    cv2.destroyAllWindows()
    
def parse_annotations(anno_dir):
    """ Parse the directory in wich the annotation jsons are stored """
    global annotations, frame_names
    anno_dir = anno_dir.replace("'", "")
    # For each annotation the sistem has we are going to parse it
    for f in listdir(anno_dir):
        file_path = anno_dir + '/' + f
        # frame_names.append(f.replace(".json", ""))
        with open(file_path) as file:
            data = json.load(file)
            annotations.append(data)
            
    # In case annotations cease before the videos does    
    dif = len(frame_names) - len (annotations)
    while dif > 0: 
        annotations.append([])
        dif = dif - 1
 
# python v1_6.py --path_videos ../Laboratory_secuence_dataset
# python v1_6.py --path_videos ../EINA_dataset/videos_sincronizados

# python v1_6.py -f ../EINA_dataset/frames_sincronizados -a no
def main(args):
    global path_videos,evaluate, show_time,verbose, iou_threshold, cosine_threshold, abstract_detector, abstract_reID,path_cameras,frame_names,annotations
       
    if args.verbose:
        verbose = 1
        
    if args.iou_threshold != 0.5:
        iou_threshold = args.iou_threshold
        
    if args.show_time:
        show_time = 1
    
    if args.cosine_threshold != 0.6:
        cosine_threshold = args.cosine_threshold
        
    # Decide if we are showing raw videos procesed or frame by frame procesing 
    args.path_videos = args.path_videos.replace("'", "")
    
    if args.path_videos != "":
    
        print("Getting camera video paths...")
        use_videos = 1
        initiate_camera_paths(args.path_videos,use_videos)
        
        print("Starting recogniser predictor config...")
        args.predictor = args.predictor.replace("'", "")
        abstract_detector = detector(args.predictor)
        
        print("Starting recogniser reID model config...")
        args.reID = args.reID.replace("'", "")
        args.reID_Path = args.reID_Path.replace("'", "")
        abstract_reID = reID(args.reID,args.reID_Path,"cuda")
         
        print("Showing videos...")
        show_videos()
         
    else:
    
        print("Getting camera image paths...")
        use_videos = 0
        initiate_camera_paths(args.frames,use_videos)
        
        print("Parsing JSON files...")
        args.annotations = args.annotations.replace("'", "")
        if os.path.exists(args.annotations):
            parse_annotations(args.annotations)
            evaluate = 1
            
        if args.ground_truth == 0:
            
            print("Starting recogniser predictor config...")
            args.predictor = args.predictor.replace("'", "")
            abstract_detector = detector(args.predictor)
            
            print("Starting recogniser reID model config...")
            args.reID = args.reID.replace("'", "")
            args.reID_Path = args.reID_Path.replace("'", "")
            abstract_reID = reID(args.reID,args.reID_Path,"cuda")
            
            print("Showing frames...")
            show_frames(args.num_processed_frames)
        else:
            print("Showing ground truth of the data set...")
            show_gt(path_cameras,frame_names,annotations)
        
    
#   ARGS: -h for help
#       -f  DIR_FRAMES
#       -a  DIR_ANNOTATIONS
#       -v  0|1 
#       ... 
#
# Nomal execution with default parameters and no verbos: python v1_2.py -v 0
# Execute with no annotations and verbose: python v1_2.py -v 0 -a no
# ...
if __name__ == "__main__":

    # get and parse arguments passed to main
    parser = argparse.ArgumentParser()
    
    # Take care, by default you will be using frame imgs,  specify a path to use videos instead
    parser.add_argument("--path_videos", help="Directory from which we will be reading the frames, it should contain the directories of each camera's videos",
                        type=ascii, default="")
    parser.add_argument("-f", "--frames", help="Directory from which we will be reading the frames, it should contain the directories of each camera's frames",
                        type=ascii, default="../Wildtrack_dataset/Image_subsets")
    
    parser.add_argument("-a", "--annotations", help="Directory in wich the annotations are stored, it should contain the directories of each frame's annotations",
                        type=ascii, default="../Wildtrack_dataset/annotations_positions")
                        
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        type=int, default=0)
    parser.add_argument("-t", "--show_time", help="Show the time that takes to process each set of frames",
                        type=int, default=0)
    parser.add_argument("-npf", "--num_processed_frames", help="How many frames to process before stopping",
                        type=float, default=np.inf)
                        
    parser.add_argument("-gt", "--ground_truth", help="Show the ground truth of the data set, only if the annotations given are valid",
                        type=int, default=0)
                        
    parser.add_argument("-iou", "--iou_threshold", help="Set a threshold for the iou evaluation",
                        type=float, default=0.5)
    parser.add_argument("-cos", "--cosine_threshold", help="Set a threshold for the cosine distance evaluation",
                        type=float, default=0.6)
                        
    parser.add_argument("-p", "--predictor", help="Name of the person recogniser predictor. Aviable: YOLOv4",
                        type=ascii, default="YOLOv4")
    parser.add_argument("-r", "--reID", help="Name of the re-idnetification model to use",
                        type=ascii, default="osnet_ain_x1_0")
    parser.add_argument("-r_p", "--reID_Path", help="Path to the re-identification model to use",
                        type=ascii, default="./models/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth")
    args = parser.parse_args()

    main(args)