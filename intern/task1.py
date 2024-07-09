import cv2
import numpy as np
import os

# Function to detect lanes in an image
def detect_lanes(image, blur_kernel=9, canny_low=50, canny_high=150):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)
    return edges

# Function to filter lines based on slope and length
def filter_lines(lines, threshold_slope, min_line_length, max_line_gap, roi_y):
    if lines is not None:
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                line_length = np.sqrt((x2 - x1)*2 + (y2 - y1)*2)
                if abs(slope) > threshold_slope and line_length > min_line_length and y1 > roi_y and y2 > roi_y:
                    filtered_lines.append(line)
        return filtered_lines
    else:
        return []

# Function to average and draw lanes
def average_lines(lines, image_shape):
    if lines:
        left_lines, right_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)
            if slope < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)
        
        def average_line(lines):
            x_coords, y_coords = [], []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                x_coords.extend([x1, x2])
                y_coords.extend([y1, y2])
            if x_coords and y_coords:
                coeffs = np.polyfit(x_coords, y_coords, 1)
                slope, intercept = coeffs[0], coeffs[1]
                y1 = int(image_shape[0])
                y2 = int(image_shape[0] * 0.6)  # Adjust the endpoint of the line
                x1 = int((y1 - intercept) / slope)
                x2 = int((y2 - intercept) / slope)
                return [[x1, y1, x2, y2]]
            else:
                return []
        
        averaged_left_line = average_line(left_lines)
        averaged_right_line = average_line(right_lines)
        return averaged_left_line + averaged_right_line
    else:
        return []

# Function to draw lanes on an image
def draw_lines(image, lines, color=(0, 255, 0), thickness=8):  # Increase thickness to make lines more visible
    line_image = np.zeros_like(image)
    if lines:
        for line in lines:
            if isinstance(line, (list, tuple)) and len(line) == 4:
                x1, y1, x2, y2 = line
                cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
            else:
                print(f"Invalid line format: {line}")
    return line_image

# Function to process an image for lane detection
def process_image(image_path, threshold_slope=0.5, min_line_length=50, max_line_gap=100, roi_y=300):
    image = cv2.imread(image_path)
    if image is not None:
        edges = detect_lanes(image)
        lane_lines_image = cv2.HoughLinesP(edges, 1, np.pi/180, 50, np.array([]), minLineLength=50, maxLineGap=100)
        filtered_lane_lines_image = filter_lines(lane_lines_image, threshold_slope, min_line_length, max_line_gap, roi_y)
        averaged_lines_image = average_lines(filtered_lane_lines_image, image.shape)
        lane_image = draw_lines(image, averaged_lines_image)
        result_image = cv2.addWeighted(image, 0.8, lane_image, 1, 0)
        cv2.imshow('Lane Detection Image', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Error loading the image '{image_path}'. Check the file path.")

# Function to process a video for lane detection with temporal consistency check
def process_video(video_path, threshold_slope=0.5, min_line_length=50, max_line_gap=100, roi_y=300):
    cap = cv2.VideoCapture(video_path)
    prev_lines = []  # List to store previous detected lines
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            edges = detect_lanes(frame)
            lane_lines_video = cv2.HoughLinesP(edges, 1, np.pi/180, 50, np.array([]), minLineLength=50, maxLineGap=100)
            filtered_lane_lines_video = filter_lines(lane_lines_video, threshold_slope, min_line_length, max_line_gap, roi_y)
            if filtered_lane_lines_video:
                averaged_lines_video = average_lines(filtered_lane_lines_video, frame.shape)
                # Keep track of dominant lines for each lane
                dominant_lines = []
                for line in averaged_lines_video:
                    if not dominant_lines:
                        dominant_lines.append(line)
                    else:
                        # Check if the line is significantly different from existing dominant lines
                        if not any(np.allclose(line, dom_line, atol=20) for dom_line in dominant_lines):
                            dominant_lines.append(line)
                # Draw the dominant lines on the frame
                lane_image_video = draw_lines(frame, dominant_lines)
                result_video = cv2.addWeighted(frame, 0.8, lane_image_video, 1, 0)
                cv2.imshow('Lane Detection Video', result_video)
            else:
                cv2.imshow('Lane Detection Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()



# Path to the folder containing images
image_folder = 'test_images'
# Get a list of all image files in the folder
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
# Process each image file in the list
for image_file in image_files:
    process_image(image_file)

# Path to the folder containing videos
video_folder = 'test_videos'
# Get a list of all video files in the folder
video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if os.path.isfile(os.path.join(video_folder, f))]
# Process each video file in the list
for video_file in video_files:
    process_video(video_file)
