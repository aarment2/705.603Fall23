import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from wandb import Classes
from scipy import ndimage
from skimage.util import random_noise  

class ObjectDetection( ):
    
    def __init__(self):      
        #Load the YOLO model
        net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
        
   
    #Converts an image from BGR to RGB and plots
    def plot_cv_img(input_image):     
        # change color channels order for matplotlib     
        plt.imshow(cv.cvtColor(input_image, cv.COLOR_BGR2RGB))          
        plt.axis('off')  
        plt.savefig("DetectionOutput.jpg")
        plt.show()

    #Resize the image
    #Images to be resized to maximum 416px for either width or height. The other side will be resized proportionally, to avoid distortion
    def resize_image(self, image):
        
        image_height = image.shape[0]
        image_width = image.shape[1]

        if image_width >= image_height:
            new_width = 416
            new_height = int(image_height * (new_width / image_width))
        else:
            new_height = 416
            new_width = int(image_width * (new_height / image_height))

        return cv.resize(image, (new_width, new_height))
        
    #Rotate the image using scipy
    def rotate_image(self, image, angle):
        return ndimage.rotate(image, angle, reshape=False, mode='constant', cval=255)

    #Add speckle noise
    def add_noise(self, image, intensity):
        noise = np.random.normal(0, intensity, image.shape).astype('uint8')
        return cv.add(image, noise)

    #Image processing cycle
    def process_image(self, image, uploadFolder=""):
        rotation_angle=0
        noise_intensity=0
        
        # Resize the image
        image = self.resize_image(image)
        backup = image
        
        # Apply rotation and speckle noise
        for x in range(1,11):
            image = backup
            image = self.rotate_image(image, rotation_angle)
            image = self.add_noise(image, noise_intensity)
            self.detect_objects(image)
            
            newName = str(x)+".jpg"
            newName = os.path.join(uploadFolder, newName)
            save_image = cv.imwrite(newName,image)
            
            rotation_angle += 10
            noise_intensity += 0.25

        return

    #Object Detection
    def detect_objects(self, image):
        # Create a blob from the image
        input_width = 416
        input_height = 416
        height, width, channel = image.shape
        
        net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
        
        classes = []
        with open("coco.names", 'r') as f:
            classes = [line.strip() for line in f.readlines()]
            
        layer_name = net.getLayerNames()
        output_layer = [layer_name[i - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        
        blob = cv.dnn.blobFromImage(image, 1.0/255.0, (input_width, input_height), [0, 0, 0], True, crop=False)

        #Detect objects
        net.setInput(blob)
        outs = net.forward(output_layer)

        #Draw boxes
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.7:
                    # Object detection
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # cv.circle(img, (center_x, center_y), 10, (0, 255, 0), 2 )
                    # Reactangle Cordinate
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
           
        print("Type: ", type(class_ids))

        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)

        font = cv.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])+","+str(round(confidences[i],2))
                print(f'Object: {label} with confidence of {confidences[i]:.2f}')
                color = colors[i]
                cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv.putText(image, label, (x, y + 30), font, 1.5, color, 2)

    def graph_results(self, images, titles):
                for i, (img, title) in enumerate(zip(images, titles)):
                    plt.subplot(1, len(images), i + 1)
                    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
                    plt.title(title)
                    plt.axis('off')
                plt.show()