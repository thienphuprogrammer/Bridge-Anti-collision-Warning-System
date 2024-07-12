import numpy as np
import torch
from PIL import ImageDraw, ImageFont, Image
from torchvision import transforms
from models.utils import *
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = './checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect_image(original_image, min_score, max_overlap, top_k, suppress=None):
    image = normalize(to_tensor(resize(original_image)))
    image = image.to(device)
    predicted_locs, predicted_scores = model(image.unsqueeze(0))
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)
    det_boxes = det_boxes[0].to('cpu')
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    if det_labels == ['background']:
        # Just return original image
        return original_image
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("CalibriL.ttf", 15)
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        bbox = font.getbbox(det_labels[i].upper())
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_size = (text_width, text_height)

        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    return annotated_image

# return list recognized object
def detect_video(original_image, min_score, max_overlap, top_k, suppress=None):
    image = normalize(to_tensor(resize(original_image)))
    image = image.to(device)
    predicted_locs, predicted_scores = model(image.unsqueeze(0))
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)
    det_boxes = det_boxes[0].to('cpu')
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    return det_boxes, det_labels, det_scores

if __name__ == '__main__':
    img_path = 'image/test/70213beecf016d5f341012.jpg'
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    detect_image(original_image, min_score=0.2, max_overlap=0.5, top_k=200).show()

    # Load video
    # cap = cv2.VideoCapture('image/test/3640b947a336feb006efa00de546a881.mp4')
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if ret:
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         frame = Image.fromarray(frame)
    #         det_boxes, det_labels, det_scores = detect_video(frame, min_score=0.2, max_overlap=0.5, top_k=200)
    #
    #         # Draw boxes
    #         draw = ImageDraw.Draw(frame)
    #         font = ImageFont.truetype("CalibriL.ttf", 15)
    #         for i in range(det_boxes.size(0)):
    #             box_location = det_boxes[i].tolist()
    #             draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
    #             draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[det_labels[i]])
    #             # Text
    #             bbox = font.getbbox(det_labels[i].upper())
    #             text_width = bbox[2] - bbox[0]
    #             text_height = bbox[3] - bbox[1]
    #             text_size = (text_width, text_height)
    #             text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
    #             textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4., box_location[1]]
    #             draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
    #             draw.text(xy=text_location, text=det_labels[i].upper(), fill='white', font=font)
    #         del draw
    #         frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    #         cv2.imshow('frame', frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #
    #     else:
    #         break
