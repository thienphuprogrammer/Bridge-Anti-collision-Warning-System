import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw
from module import *

inch_to_cm = 2.54
ppi = 19812
label_cell_phone = 67
label_mouse = 64
label_cup = 41


def flip_image(image):
    image = image.transpose(method=Image.FLIP_TOP_BOTTOM)
    image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
    return image


def show_objects(image, points: np.ndarray = None, lines: np.ndarray = None, rectangles: np.ndarray = None):
    draw = ImageDraw.Draw(image)
    if points is not None:
        for point in points:
            # convert the point to 2d
            draw.circle(point, 5, fill='red')
    if lines is not None:
        for line in lines:
            draw.line(line, fill='green', width=2)

    if rectangles is not None:
        for rectangle in rectangles:
            draw.rectangle(rectangle, outline='blue', width=2)
    image.show()


def get_objects_by_labels(results, label: int):
    objects = results[0].boxes
    labels = objects.cls
    objects = objects[labels == label]
    if objects.cls is None:
        return None
    return objects


def middle_object(width, height, position) -> Tuple[np.ndarray, np.ndarray]:
    x_min, y_min, x_max, y_max = position
    point_1 = np.array([x_min + ((x_max - x_min) / 2), y_max, 0])
    point_1 = np.array([point_1[0] + cx, point_1[1] + cy, 0])

    point_2 = np.array([x_min + ((x_max - x_min) / 2), y_min, 0])
    point_2 = np.array([point_2[0] + cx, point_2[1] + cy, 0])
    return point_1, point_2


class Model:
    def __init__(self, rand_points_: np.ndarray):
        self.yolo = YOLO('yolov8l.pt')
        self.camera = Camera(
            angles=angles,
            order=order,
            offset=offset,
            f=f,
            s=s,
            a=a,
            cx=cx,
            cy=cy,
            image_size=img_size
        )

        self.rand_points = rand_points_
        self.projected_points = self.camera.project_3d_point_to_2d(self.rand_points)
        self.lines = range_image(self.projected_points)
        self.plane = plane_location(self.rand_points)
        self.pinhole = None

    def detect_objects(self, img_path: str):
        image = Image.open(img_path)
        img = np.array(image)
        # img = flip_image(img)
        return self.yolo(img)

    def measure_object(self, image_size, num_array_1, num_array_2):
        w_1, h_1 = image_size
        w_1 = (w_1 / ppi) * inch_to_cm
        h_1 = (h_1 / ppi) * inch_to_cm

        w_2, h_2 = image_size
        w_2 = (w_2 / ppi) * inch_to_cm
        h_2 = (h_2 / ppi) * inch_to_cm

        # get the objects in the first image
        num_array_1 = (num_array_1 / ppi) * inch_to_cm
        x_1_min, y_1_min, x_1_max, y_1_max = num_array_1

        # get the objects in the second image
        num_array_2 = (num_array_2 / ppi) * inch_to_cm
        x_2_min, y_2_min, x_2_max, y_2_max = num_array_2

        point_1, point_2 = middle_object(w_1, h_1, num_array_1)
        point_3, point_4 = middle_object(w_2, h_2, num_array_2)

        new_line_2 = ray_tracing(np.array([offset, point_2]))
        intersection_2 = intersection_between_line_and_plane(new_line_2, self.plane)

        point_5 = np.array([x_1_min, y_1_max, 0])
        point_5 = np.array([point_5[0] + cx, point_5[1] + cy, 0])

        point_6 = np.array([x_1_max, y_1_max, 0])
        point_6 = np.array([point_6[0] + cx, point_6[1] + cy, 0])

        point_7 = np.array([x_2_min, y_2_max, 0])
        point_7 = np.array([point_7[0] + cx, point_7[1] + cy, 0])

        point_8 = np.array([x_2_max, y_2_max, 0])
        point_8 = np.array([point_8[0] + cx, point_8[1] + cy, 0])

        r = f
        R = abs(offset[2] - intersection_2[2])
        H = offset[1]
        h_s_1 = (distance_between_points(point_1, point_2))
        h_s_2 = (distance_between_points(point_3, point_4))
        d_s_1 = (distance_between_points(point_5, point_6))
        d_s_2 = (distance_between_points(point_7, point_8))

        point_old_1, vec_u_1 = self.lines[0]
        t_1 = (-point_old_1[0] + point_5[0]) / vec_u_1[0]
        point_new_1 = point_old_1 + t_1 * vec_u_1

        point_old_2, vec_u_2 = self.lines[1]
        t_2 = (-point_old_2[0] + point_5[0]) / vec_u_2[0]
        point_new_2 = point_old_2 + t_2 * vec_u_2

        d_1 = (distance_between_points(point_new_1, point_new_2))
        W_c = abs(self.rand_points[1][1] - self.rand_points[0][1])
        self.pinhole = Pinhole(r, R, H, h_s_1, h_s_2, d_s_1, d_s_2, d_1, W_c)
        print(f"r: {r} \nR: {R} \nH: {H} \nh_s_1: {h_s_1} \nh_s_2: {h_s_2} \nd_s_1: {d_s_1} \nd_s_2: {d_s_2} \nd_1: {d_1} \nW_c: {W_c}")
        value = self.pinhole.calculate_height_and_length_of_target()
        L_s, H_s = value[0]
        H_s = H_s
        return L_s, H_s

    def handle_logic_with_camera(self):
        # handle the logic
        cap = cv2.VideoCapture('./image/15724706959')
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 0.5)
        list_points = []
        frame_count = 0
        L_s, H_s = 0, 0
        tep = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if frame_count % frame_interval == 0:
                    results = self.yolo(frame)
                    objects = get_objects_by_labels(results, label_mouse)
                    if objects.xyxy.numel() == 0:
                        continue
                    num_array = objects.xyxy[0].detach().cpu().numpy()
                    x_min, y_min, x_max, y_max = num_array
                    list_points.append(num_array)

                    cv2.rectangle(frame,
                                  (int(x_min), int(y_min)),
                                  (int(x_max), int(y_max)),
                                  (0, 255, 0), 2)

                    if len(list_points) > 0 and y_min > list_points[0][3] + 15:
                        result_1 = list_points[0]
                        result_2 = num_array
                        h, w, _ = frame.shape
                        L_s, H_s = self.measure_object((w, h), result_1, result_2)
                        print(f"L_s: {L_s} \nH_s: {H_s}")
                        # remove the first element
                        list_points.pop(0)

                    # add the text to the image
                cv2.putText(frame, f"L_s: {L_s:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"H_s: {H_s:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Display image
                cv2.imshow('frame', frame)
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break
                frame_count += 1
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

    def handle_logic_with_image(self):
        file_image_1 = 'image/img_7.png'
        file_image_2 = 'image/img_6.png'

        results_1 = model.detect_objects(file_image_1)
        results_2 = model.detect_objects(file_image_2)

        results_1 = get_objects_by_labels(results_1, label_cup)
        results_2 = get_objects_by_labels(results_2, label_cup)
        if results_1.xyxy.detach().cpu().numpy().size == 0 or results_2.xyxy.detach().cpu().numpy().size == 0:
            return

        num_array_1 = results_1.xyxy[0].detach().cpu().numpy()
        num_array_2 = results_2.xyxy[0].detach().cpu().numpy()

        image_1 = Image.open(file_image_1)
        size = (image_1.size[0], image_1.size[1])
        L_s, H_s = self.measure_object(size, num_array_1, num_array_2)
        print(f"L_s: {L_s} \nH_s: {H_s}")


z = [70, 70, 1, 1]
y = [10, 110, 110, 10]
x = [0, 0, 0, 0]
rand_points = np.vstack((x, y, z))
model = Model(rand_points)
# image_1 = 'image/img_5.png'
# image_2 = 'image/img_4.png'
#
# results_1 = model.detect_objects(image_1)
# results_2 = model.detect_objects(image_2)
# results_1[0].show()
# print(results_1[0].boxes)

model.handle_logic_with_image()
# model.handle_logic_with_camera()

# file_image_1 = 'image/img_7.png'
# label = 64  # 64 is the mouse label
# results_1 = model.detect_objects(file_image_1)
# image_1 = Image.open(file_image_1)
# objects_1 = model.get_objects_labels(results_1, label)  # get the mouse objects in the first image
# num_array_1 = objects_1.xyxy[0].detach().cpu().numpy()
# x_min, y_min, x_max, y_max = num_array_1
# draw = ImageDraw.Draw(image_1)
# draw.rectangle(num_array_1, outline='blue', width=2)
# draw.circle((x_min, y_max), 5, fill='red')
# image_1.show()