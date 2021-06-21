import cv2
from v1_6 import process_evaluation

def draw_in_frame(frame_list,  annotations):
    num_people = 0
    processed_camera = 0
    
    # People detected on this set of frames
    num_people = len(annotations)
    
    all_boxs_set =[]
    all_ids_set = []
    
    for frame in frame_list:
        
        box_set = []
        id_set = []
            
        for persona in annotations:
    
            # For each person in the especific frame annotation we take its x,y
            id_p = persona["personID"]
            views = persona["views"]
            values = views[processed_camera]
            
            xmax = values["xmax"]
            xmin = values["xmin"]
            ymax = values["ymax"]
            ymin = values["ymin"]
            
            box_set.append([xmin,ymin,xmax,ymax])
            id_set.append(id_p)
            
            cv2.rectangle(frame, (xmax, ymax), (xmin, ymin),(18, 236, 18), 2)
            y = ymax - 15 if ymax - 15 > 15 else ymax + 15
            cv2.putText(frame, str(id_p), (xmin+15, ymin+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (18, 236, 18) , 1)

        processed_camera =  processed_camera + 1
        all_boxs_set.append(box_set)
        all_ids_set.append(id_set)
    process_evaluation(annotations, frame_list, all_boxs_set, all_ids_set, processed_camera,1)

    return num_people

def show_gt(path_cameras,frame_names,annotations):
    num_camaras = len(path_cameras)
    num_frames_procesed = 0
    imgs_windows = []
    
    for frame_name in frame_names:
        
        frame_list = []
        
        # Get frame i of each camera
        for i in range(num_camaras):
            path_frame = path_cameras[i] + "/" + frame_name + ".png"
            img = cv2.imread(path_frame,0)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR ) 
            frame_list.append(img)
            
        # Process the frame i taking into account all camera angles
        num_people = draw_in_frame(frame_list,  annotations[num_frames_procesed])
        
        for i in range(1): #range(num_camaras): !!!!! Remember to change this to show all processed frames
            name = "Camera: " + str(i)
            cv2.putText(frame_list[i], 'NUMBER OF PEOPLE: ' + str(num_people), (50, 50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 255), 2, cv2.LINE_4)
            cv2.putText(frame_list[i], 'Error: ' + "{:.4f}".format(0.0), (50, 80), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 255), 2, cv2.LINE_4)
   
            cv2.imshow(name,frame_list[i])
            
        num_frames_procesed = num_frames_procesed + 1
        cv2.waitKey(0)
    return 0;