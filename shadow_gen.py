import cv2
import numpy as np
import gradio as gr

def draw_line(image, start_point, target_point, length, scaling_factor):
    # Calculate the vector from start_point to target_point
    vector = np.array(target_point) - np.array(start_point)
    
    # Normalize the vector
    normalized_vector = vector / np.linalg.norm(vector)
    
    # Calculate the endpoint based on the length and scaling_factor
    endpoint = start_point + length * scaling_factor * normalized_vector
    
    # Convert endpoint to integer coordinates
    endpoint = tuple(map(int, endpoint))
    
    # Draw the line on the image
    color = (0, 255, 0)  # BGR color (green in this case)
    thickness = 2
    cv2.line(image, start_point, endpoint, color, thickness)
    cv2.imshow("line", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return endpoint

def overlay_with_black(input_image):

    background_image = np.zeros_like(input_image)

    # Find the pixels with alpha value 255 and replace them with black                  #This generates shadow by overlaying black pixels on those pixels having alpha channel value 255
    alpha_channel = input_image[:, :, 3]
    background_image[:, :, 0:3][alpha_channel == 255] = [0, 0, 0]
    background_image[:, :, 3] = alpha_channel

    return background_image

def pasteImage(background, foreground):
    # normalize alpha channels from 0-255 to 0-1
    alpha_background = background[:,:,3] / 255.0
    alpha_foreground = foreground[:,:,3] / 255.0                                        

    # set adjusted colors
    for color in range(0, 3):
        background[:,:,color] = alpha_foreground * foreground[:,:,color] + \
            alpha_background * background[:,:,color] * (1 - alpha_foreground)

    # set adjusted alpha and denormalize back to 0-255
    background[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255

    return background

def shadow(image, o_offset, rot, scaling_factor):
    shadow_image = overlay_with_black(image)
    
    alpha_channel_original = shadow_image[:, :, 3]  # Extract the alpha channel

    # Create a mask for the object (considering non-zero alpha values)
    object_mask_original = alpha_channel_original > 0

    # Find coordinates of the object
    coords_original = np.argwhere(object_mask_original)

    # Calculate the bounding box based on coordinates
    topmost_original = tuple(np.min(coords_original, axis=0))  
    bottommost_original = tuple(np.max(coords_original, axis=0))  
    leftmost_original = tuple(np.min(coords_original, axis=0))
    rightmost_original = tuple(np.max(coords_original, axis=0))
    height_original = bottommost_original[0] - topmost_original[0]
    print(height_original)

    start_point_left = (leftmost_original[1], bottommost_original[0])
    target_point_left = (leftmost_original[1], topmost_original[0])
    start_point_right = (rightmost_original[1], bottommost_original[0])
    target_point_right = (rightmost_original[1], topmost_original[0])
    
    endpoint_left = draw_line(image, start_point_left, target_point_left, height_original, scaling_factor)
    endpoint_right = draw_line(image, start_point_right, target_point_right, height_original, scaling_factor)
    print(endpoint_left, endpoint_right)
    perspective_matrix = cv2.getPerspectiveTransform(
    np.float32([(leftmost_original[1], topmost_original[0]), 
                (rightmost_original[1], topmost_original[0]), 
                (leftmost_original[1], bottommost_original[0]), 
                (rightmost_original[1], bottommost_original[0])]),
    np.float32([(endpoint_left[0]+30, endpoint_left[1]), 
                (endpoint_right[0]-30, endpoint_right[1]), 
                (leftmost_original[1], bottommost_original[0]),
                (rightmost_original[1], bottommost_original[0])])
    )
    shadow_image = cv2.warpPerspective(shadow_image, perspective_matrix, (shadow_image.shape[1], shadow_image.shape[0]))

    alpha_channel = shadow_image[:, :, 3]
    object_mask = alpha_channel > 0
    coords = np.argwhere(object_mask)
    topmost = tuple(np.min(coords, axis=0))  
    bottommost = tuple(np.max(coords, axis=0))  
    leftmost = tuple(np.min(coords, axis=0))
    rightmost = tuple(np.max(coords, axis=0))



    #Height and Width Calculation
    height = bottommost[0] - topmost[0]
    width = rightmost[1] - leftmost[1]
    print(height)
    rot_angle = np.deg2rad(rot)
    x1 = int(leftmost[1] + height*np.cos(rot_angle))                       #These are the 2 points in cartesian plane which will depict the transformation of shadow
    y1 = int(bottommost[0] - height*np.sin(rot_angle))                      #Previously Tan function was used for the resizing as tan(theta) respresents slope, but that was wrong, as the individual elevations and deductions 
    x2 = int(rightmost[1] + height*np.cos(rot_angle) + width)              #was not the same for each x and y point, I realised x and y change by different amounts on different elevations, and I now  use sin and cos funtions 
    y2 = int(bottommost[0] - height*np.sin(rot_angle))                      #The changes in each x and y point, *(x1,y1) and (x2,y2) represent the new top right and top left points of the parallelogram respectively*
    print("Old Co-ords ",leftmost[1] + x1,bottommost[0] - y1,rightmost[1] + x2,bottommost[0] - y2)
    perspective_matrix = cv2.getPerspectiveTransform(
    np.float32([(leftmost[1], topmost[0]), 
                (rightmost[1], topmost[0]), 
                (leftmost[1], bottommost[0]),
                (rightmost[1], bottommost[0])]),
    np.float32([(x1, y1), 
                (x2, y2), 
                (leftmost[1], bottommost[0]),
                (rightmost[1], bottommost[0])])
    )

    shadow_image = cv2.warpPerspective(shadow_image, perspective_matrix, (shadow_image.shape[1], shadow_image.shape[0]))            #CV2 Function for warping a plane to a differnt dimension, https://theailearner.com/tag/cv2-warpperspective/

    height, width = shadow_image.shape[:2]

    center = ((rightmost[1]+leftmost[1])//2, bottommost[0])
    dispersed_shadow = shadow_image.copy()
    s_height, s_width = dispersed_shadow.shape[:2]
    for y in range(s_height):
        for x in range(s_width):
            alpha_value = dispersed_shadow[y, x, 3]
            if alpha_value == 0:
                continue
            else:
                # Calculate the distance of each pixel from the center of the shadow
                distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)*o_offset
                # Reduce opacity and apply blur based on distance
                opacity = max(0, 1.1 - distance / max(height, width))
                if opacity> 1:
                    opacity = 1
                dispersed_shadow[y, x, 3] = int(opacity * 255)  # Update the alpha channel
    dispersed_shadow = cv2.GaussianBlur(dispersed_shadow, (21, 21), 0, dst=dispersed_shadow, borderType=cv2.BORDER_DEFAULT)

    composite = pasteImage(dispersed_shadow, image)
    white_background = np.zeros((height, width, 4), dtype=np.uint8)                 #add it to white background for now
    white_background[:, :, 0:3] = (255, 255, 255)                                   #This can easily be changed to custom image by cv2.imread or passing an image to the function itself
    white_background[:, :, 3] = 255
    result = pasteImage(white_background, composite)
    return result


'''
input_image_path = "bottle.png"
input_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
shadow_of_img = shadow(input_image, 1, 60, 0.6)
cv2.imwrite("proper.png", shadow_of_img)'''

input_image_block = gr.Image(type="numpy", image_mode="RGBA")

# Define slider blocks for parameters
rot_slider = gr.Slider(minimum=0, maximum=360, label="Rotation (0-360)")
o_offset_slider = gr.Slider(minimum=0, maximum=2, step=0.1, label="Opacity Offset (0-2)")
scale_slider = gr.Slider(minimum=0, maximum=4, step=0.1, label="Scale Offset (0 to 4)")

# Define an output component for the image as a NumPy array
output_image_block = gr.Image(type="numpy", image_mode="RGBA")

# Create the interface
iface = gr.Interface(
    fn=shadow,  # Your function that processes input
    inputs=[input_image_block,o_offset_slider, rot_slider, scale_slider],
    outputs=output_image_block
)

iface.launch()

