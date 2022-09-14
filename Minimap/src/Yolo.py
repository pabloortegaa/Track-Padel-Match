import cv2
import random

class Yolo:
    def __init__(self, model, cap):
        self.model = model
        self.cap = cap
        self.init_attributes()
        
        self.ret, self.frame = cap.read()
        self.coordinates_src = []
        
        self.labels = {}
        self.init_labels()
        
    def get_attributes(self, name):
        self.init_attributes()
        
        self.objects = self.model(self.frame).pandas().xyxy[0]
        
        for i in range(len(self.objects)):
            if self.objects["name"][i] == name:
                x0 = int(self.objects["xmin"][i])
                y0 = int(self.objects["ymin"][i])

                x1 = int(self.objects["xmax"][i])
                y1 = int(self.objects["ymax"][i])

                confidence = round(self.objects["confidence"][0], 2)
                self.bbox.append([x0, y0, x1, y1, confidence])
                
                self.pt_medio.append([int((x0 + x1) / 2), y1])
                                        
        self.get_labels()
                
    def draw_pt_medio(self):
        for i in self.pt_medio:        
            cv2.circle(self.frame, (i[0], i[1]), 5, (0,0,255), -1)

    def draw_bbox(self, name):
        for i in self.bbox:
            cv2.rectangle(self.frame, (i[0], i[1]), (i[2], i[3]), (255,0,0), 2)
            cv2.putText(self.frame, name + " " + str(i[4]), (i[0], i[1]-10), 0, 0.5, (255,0,0))
            
    def draw_labels(self):
        for i in self.labels:
            cv2.putText(self.frame, str(i), (self.labels[i]["point"][0], self.labels[i]["point"][1]+30), 0, .5, (255,255,255))
            
            
    def is_same_label(self, pt_central_actual, pt_central_anterior):
        x_in_range = (pt_central_anterior[0] - 55 < pt_central_actual[0]) and  (pt_central_actual[0] < pt_central_anterior[0] + 55)
        y_in_range = (pt_central_anterior[1] - 55 < pt_central_actual[1]) and  (pt_central_actual[1] < pt_central_anterior[1] + 55)
        
        if x_in_range and y_in_range: return True
        
        return False

    
    def get_labels(self):
        for i in self.pt_medio: 
            for j in self.labels:
                if self.is_same_label(i, self.labels[j]["point"]):
                    self.labels[j]["point"] = i
            
        
    def init_labels(self):
        self.get_attributes("person")
        for i in self.pt_medio:
            b = random.randint(0,255)
            g = random.randint(0,255)
            r = random.randint(0,255)
            self.labels["player_id" + str(len(self.labels))] = {"point":i, "color":(b,g,r)}
                
            
    def init_attributes(self):
        self.bbox = []
        self.pt_medio = []
                
    def generate_next_frame(self):
        self.ret, self.frame = self.cap.read()
            
    def is_finished(self):
        return not self.ret
    
    def get_coordinates(self, event, x, y, flags, param):
        punto = []
        if event == cv2.EVENT_LBUTTONDOWN:
            punto.append(x)
            punto.append(y)
            self.coordinates_src.append(punto)