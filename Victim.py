import numpy as np
import cv2
from ultralytics import YOLO
import torch
from datetime import datetime
device = "0" if torch.cuda.is_available() else "cpu"

if device == "0":
    torch.cuda.set_device(0)

# noinspection PyPep8Naming
class Victim:
    ai = YOLO(r"C:\Users\Robotika\Desktop\Robotika\best.pt")
    classes_number = ai.names
    classes_number[0] = "C"
    classes_number[1] = "F"
    classes_number[3] = "O"
    classes_number[4] = "P"
    
    @staticmethod
    def predictionImg(images, camera):
        results = Victim.ai.predict(images, verbose = False)
        
        results_classes = [int(predicted_class) for predicted_class in results[0].boxes.cls]
        #results_box = [int(predicted_class) for predicted_class in results[0].boxes.xywh]
        victim_found = True
        if len(results_classes) > 0:
            prediction = Victim.classes_number[results_classes[0]]
            area = results[0].boxes.xywh[0][2] * results[0].boxes.xywh[0][3]
            start_point = tuple(int(x) for x in results[0].boxes.xyxy[0][:2])
            end_point = tuple(int(x) for x in results[0].boxes.xyxy[0][2:])
            image_center_x = (start_point[0] + end_point[0]) /2

            #print(f"Detected {prediction} with area {area}")
            if int(area) > 350:
                message = f"The AI recognized the victim, {prediction}"
                now = datetime.now()
                date_time_str = now.strftime("%Y%m%d%H%M%S")
                filename = r"C:\Users\Robotika\Desktop\Robotika\vicc\\" + date_time_str + prediction + r".jpg"
                image = cv2.imread(images[0], cv2.IMREAD_COLOR)
                cv2.imwrite(filename, image)

                if camera == "right":
                    if image_center_x <= 64 /3:
                        grid_position = "front"
                        print(grid_position)
                        
                    elif image_center_x <= 128 /3:
                        grid_position = "middle"
                        print(grid_position)
                        
                    else:
                        grid_position = "back"
                        print(grid_position)
                        
                else:
                    camera == "left"
                    if image_center_x <= 64 /3:
                        grid_position = "back"
                        print(grid_position)
                        
                    elif image_center_x <= 128 /3:
                        grid_position = "middle"
                        print(grid_position)
                        
                    else:
                        grid_position = "front"
                        print(grid_position)
                        



                #print(camera, grid_position)
                return prediction, message, grid_position, victim_found
            
            else:
                return None, None, None, victim_found

        else:
            return None, None, None, victim_found
    @staticmethod
    def detectVictim(img, camera, is_turning):
        return True