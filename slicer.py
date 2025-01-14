import cv2
import os
import numpy as np

stickerfolder = 'stickerimages'
output_dir = 'cropped_stickers'
os.makedirs(output_dir, exist_ok=True)


images = os.listdir(stickerfolder)
print(images)

images = [item for item in images if item.endswith('png') or item.endswith('jpg')]

for item in images:

    sticker_name = item.split(".")[0]
    output_sticker_dir = os.path.join(output_dir, sticker_name)
    os.makedirs(output_sticker_dir, exist_ok=True)

    image = cv2.imread(os.path.join(stickerfolder, item))
    color_area = np.sum(image, axis=2).astype(np.uint8)

    _, thresh = cv2.threshold(color_area, 29, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sticker_count = 1
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 50: 
            cropped_sticker = image[y:y+h, x:x+w]
 
            background_color = [0,0,0]

            alpha_channel = np.ones((cropped_sticker.shape[0], cropped_sticker.shape[1]), dtype=np.uint8) * 255

            mask = cv2.inRange(cropped_sticker[:, :, :3], np.array(background_color), np.array(background_color))

            alpha_channel[mask > 0] = 0

            image_with_alpha = cv2.merge((cropped_sticker[:, :, 0], cropped_sticker[:, :, 1], cropped_sticker[:, :, 2], alpha_channel))

            cv2.imwrite(f"{output_sticker_dir}/sticker_{sticker_count}.png", image_with_alpha)
            sticker_count += 1

    print(f"Extracted {sticker_count-1} stickers. Saved in: {output_sticker_dir}")
