import cv2
import os

# Replace strings with directories of the images, labels and final results
############################################################################    
image_directory = "C:\Users\Owner\Desktop\fyp\Labelling\testimages" # Directory to image
label_directory = "C:\Users\Owner\Desktop\fyp\Labelling\testimages" # Directory to label
results_directory = "C:/Users/Owner/Desktop/fyp/Labelling/camera_2_annotated" # Directory for results
############################################################################

def numeric_part(filename):  # to extract the sequence no of image
    numeric_part = int(filename[7:-4])
    return numeric_part

for image_filename in sorted(os.listdir(image_directory),key = numeric_part): # looping thorugh the directory by the sequence no of image

    # Image
    img_path = os.path.join(image_directory, image_filename) # obtaining path of image
    img = cv2.imread(img_path)
    dh, dw = img.shape # obtaining size

    # Label
    label_filename = image_filename.replace(".png", ".txt") # just replacing .png with .txt, names of labels and image is the same
    label_path = os.path.join(label_directory, label_filename) # obtaining path of labels
    fl = open(label_path, 'r')
    data = fl.readlines() # reading txt file
    fl.close()

    # Drawing Bounding Boxes
    for dt in data:
    # Split string to float from labels
        dt = ' '.join(dt.split())
        print(dt)
        x, y, w, h,conf,seq,led_id = map(float, dt.split(' ')) #
        

    # Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
    # via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380
        l = int((x - w / 2) * dw) # left
        r = int((x + w / 2) * dw) # right
        t = int((y - h / 2) * dh) # top
        b = int((y + h / 2) * dh) # bottom
    
        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1

        cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 1)
        #cv2.putText(img, led_id, (l, b), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1) # input 2nd arg as as string, wip
        #cv2.putText(img, conf, (r, t), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
        
    output_path = os.path.join(results_directory, image_filename)
    cv2.imwrite(output_path, img)