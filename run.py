# Using ALDI.mp4,
#   runs pose detection inference
#   using left wrist's prediction, counts its crossings of the basket "threshold"


from ultralytics import YOLO
import cv2
import os
from datetime import datetime

input_dir = "imgs"
out_dir = "imgs_annotated"

# how frequently to analyze ?
frame_sample_rate_seconds = 0.3

# for this video, the boundary of basket is >250px; in reality it
y_threshold = 250


# initial values for thresholding function's results pack
results_pack = {
            "previous_y": 0,
            "previous_time": datetime.now()
        }
texts = []


# Returns id of the biggest bounding box
def __get_biggest_bbox_id(results):
    biggest_bbox_id = -1
    biggest_bbox = 0
    for i in range(len(results.boxes)):
        bbox = results.boxes[i]
        bbox_size = bbox.xywh[0, 2:].prod()
        if bbox_size > biggest_bbox:
            biggest_bbox = bbox_size
            biggest_bbox_id = i
    return biggest_bbox_id

# Finds left wrist's coordinates
def __find_left_wrist_xy(results):
    joint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    left_wrist_id = joint_names.index("left_wrist")
    left_wrist_xy = results.keypoints[person_id, left_wrist_id].xy
    x,y = map(int, left_wrist_xy.detach().cpu().round().tolist())
    return x,y

# Determine if basket threshold crossed
def __was_basket_area_threshold_crossed(left_wrist_y, previous_results_pack):
    previous_time = previous_results_pack["previous_time"]
    previous_y = previous_results_pack["previous_y"]

    now = datetime.now()
    threshold_crossed = False
    # 0.5sec frame sampling
    if (now - previous_time).total_seconds() > frame_sample_rate_seconds:
        # Has the left wrist crossed the threshold from top to bottom
        if previous_y < y_threshold and left_wrist_y >= y_threshold:
            threshold_crossed = True
        previous_results_pack = {
            "previous_y": left_wrist_y,
            "previous_time": datetime.now()
        }
    return threshold_crossed, previous_results_pack

# Draws red bubble in wrist's position
def __draw_red_bubble(image, x,y):
    radius = 5  # bubble size
    color = (0, 0, 255)  # red in BGR
    thickness = -1  # filled circle
    image = cv2.circle(image, (left_wrist_x, left_wrist_y), radius, color, thickness)
    return image

# Expands image to the right with whitespace; adds texts vertically
def __add_texts_to_image(image, texts, extend_x = 100, extend_y = 0):
    white = [255, 255, 255]  # BGR

    orig_height, orig_width = image.shape[:2]

    image = cv2.copyMakeBorder(
        image,
        0, extend_y, 0, extend_x,
        borderType=cv2.BORDER_CONSTANT,
        value=white
    )

    y = 15
    for text in texts:
        cv2.putText(
            image,
            text,
            (orig_width+5, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.3,
            color=(0, 0, 0),  # black
            thickness=1,
            lineType=cv2.LINE_AA
        )
        y += 15
    return image

# Draw a blue threshold between basket area and service area
def __draw_threshold_line(image, y1, y2):
    image=cv2.line(
        image,
        (0, y1),
        (image.shape[1], y2),
        color=(255, 0, 0),  # red (BGR)
        thickness=2
    )

    cv2.putText(
        image,
        "Basket Threshold",
        (image.shape[1] -90, y1-5),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.3,
        color=(255, 0, 0),  # blue
        thickness=1,
        lineType=cv2.LINE_AA
    )
    return image


# Load pretrained multi-person pose model
model = YOLO("yolov8n-pose.pt")  # n, s, m, l available
# Force GPU
model.to("cuda")
print ("Model loaded")

for file_name in os.listdir(input_dir):
    # run inference
    full_file_name = input_dir + "\\" + file_name
    results = model(full_file_name, conf=0.25)

    # first dimension - image index (we only predict on single image here)
    results = results[0]

    # which person?
    #  (cashier is in the foreground, thus biggest; other people in bgrnd may have highest conf scores)
    person_id = __get_biggest_bbox_id(results)

    # if no people found - analyze next image
    if person_id<0:
        continue

    # find left wrist
    left_wrist_x, left_wrist_y = __find_left_wrist_xy(results)

    # Decide if threshold was crossed; if so - add an event
    threshold_crossed, results_pack = __was_basket_area_threshold_crossed(left_wrist_y, results_pack)
    if threshold_crossed:
        now = datetime.now()
        texts.append(now.strftime("%H:%M:%S") + f":{int(now.microsecond / 1000):03d}")

    annotated = cv2.imread(full_file_name)
    # draw a red bubble4 at left wrist
    annotated = __draw_red_bubble(annotated, left_wrist_x, left_wrist_y)
    # add times of crossing threshold
    annotated = __add_texts_to_image(annotated, texts)
    # draw threshold
    annotated = __draw_threshold_line(annotated, y_threshold, y_threshold)
    # save annotated file
    out_file_name = out_dir + "\\" + file_name
    cv2.imwrite(out_file_name, annotated)
