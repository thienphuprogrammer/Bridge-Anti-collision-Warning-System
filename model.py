# use Yolov8
from ultralytics import YOLO

from module import *


class Model:
    def __init__(self, rand_points: np.ndarray):
        self.yolo = YOLO('yolov8n.pt')
        self.camera = Camera()
        self.rand_points = rand_points
        self.projected_points = self.camera.project_3d_point_to_2d(self.rand_points)
        self.lines = range_image(self.projected_points)
        
    def detect_objects(self, img_path: str):
        # detect objects in the image
        results = self.yolo(img_path)
        return results
    
    
z = [14, 14, 2, 2]
y = [-4, 11, 11, -4]
x = [0, 0, 0, 0]
rand_points = np.vstack((x, y, z))
model = Model(rand_points)
image_1 = 'image/img_3.png'
image_2 = 'image/img_4.png'

results_1 = model.detect_objects(image_1)
results_2 = model.detect_objects(image_2)

results_1[0].show()
results_2[0].show()
