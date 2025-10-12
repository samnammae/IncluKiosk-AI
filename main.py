import detect
import tflite_runtime.interpreter as tflite
import time

from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np
import time
import os



# .tflite interpreter
interpreter = tflite.Interpreter(
    os.path.join(os.getcwd(), "models", "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"),
    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
    )
interpreter.allocate_tensors()

# Draws the bounding box and label for each object.
def draw_objects(image, objs):
    for obj in objs:
        bbox = obj.bbox
        
        cv2.rectangle(image,(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 255, 0),2)

        bbox_point_w = bbox.xmin + ((bbox.xmax-bbox.xmin) // 2)
        bbox_point_h = bbox.ymin + ((bbox.ymax-bbox.ymin) // 2) 
        
        cv2.circle(image, (bbox_point_w, bbox_point_h), 5, (0,0,255),-1)
        cv2.putText(image, text='%d%%' % (obj.score*100), org=(bbox.xmin, bbox.ymin), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

        

        
def main():
    cap = cv2.VideoCapture(0)    

    while True:
        ret, image = cap.read()

	    # image reshape
        image = cv2.resize(image, dsize=(320, 320), interpolation=cv2.INTER_AREA)
	    # image BGR to RGB
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        tensor = detect.input_tensor(interpreter=interpreter)
        tensor[:, :] = image.copy()
        tensor.fill(0)  # padding        
        interpreter.invoke()  # start
        
        objs = detect.get_output(interpreter, 0.5, (1.0, 1.0))
        
        if len(image):
            draw_objects(image, objs)

        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        cv2.imshow('face detector', image)

        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit # ESC exit
            break


if __name__ == '__main__':
    main()