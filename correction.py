import os

import cv2

if __name__ == '__main__':
    # Get all png files under the input folder
    save_path = "test_data/test/after_HSV_blance/"

    def mkdir(path):
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
            print("--- create new folder...  ---")
        else:
            print("---  There is this folder!  ---")


    mkdir(save_path)
    for i in range(10):
        nir = 'test_data/test/nir/' + str(i + 1) + '.png'
        vis = 'test_data/test/vis_blur/' + str(i + 1) + '.png'

        # Load both images and convert them from BGR to HSV:

        nir = cv2.cvtColor(cv2.imread(nir, cv2.IMREAD_COLOR), cv2.COLOR_BGR2HSV)
        vis = cv2.cvtColor(cv2.imread(vis, cv2.IMREAD_COLOR), cv2.COLOR_BGR2HSV)

        # Copy img1, the one with relevant color and saturation information:

        texture = nir.copy()

        # Merge img1 and img2's value channel:

        s = 0.75
        v = 0.65
        texture[:, :, 1] = s * nir[:, :, 1] + (1.0 - s) * vis[:, :, 1]
        texture[:, :, 2] = v * nir[:, :, 2] + (1.0 - v) * vis[:, :, 2]

        # Convert the image back from HSV to BGR and save it:

        cv2.imwrite(save_path + str(i + 1) + '.png', cv2.cvtColor(texture, cv2.COLOR_HSV2BGR))
        print("The", i, "picture is currently being processed")
