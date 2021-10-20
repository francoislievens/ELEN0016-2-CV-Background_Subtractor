import numpy as np
import cv2
from GaussianSubtractor import GaussianSubtractor
from MorphoOperator import morpho_operator
import os


def main(vid_path='images/CV2021_GROUP02/group2.mp4',
         vid_name='group2',
         buffer_size=100,
         display=True,
         save_frames=True):
    """
    This method will perform the background subtraction on the
    wanted file of the Cytomine previously downloaded dataset.
    :param vid_path: The path of the video file
    :param vid_name: The name of the video who will be used to
    name output frames
    :param buffer_size: The size of the buffer used by our gaussian
    background subtractor
    :param display: Show video frames in a window if True
    :param save_frames: Save each frames in the /output_frames
    folder if true
    """

    # Open the video file
    vid = cv2.VideoCapture(vid_path)

    # Prepare output folder if we want to output each frames
    if not os.path.exists('output_frames') and save_frames:
        os.makedirs('output_frames')

    # Background substraction using MOG
    subtractor = GaussianSubtractor(window_size=buffer_size)

    # Initialize the model with 100 frames
    done = False
    print('Initializing...')
    while not done:
        ret, frame = vid.read()
        # Frame to Grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        done = subtractor.push_init_frame(frame)
    print('   ... Done.')

    # Reset the video reader
    vid.release()
    vid = cv2.VideoCapture(vid_path)

    # Instanciate an erosion filter object
    delayer_A = morpho_operator(kernel_size=3)
    eroder_A = morpho_operator(kernel_size=6)
    delayer_B = morpho_operator(kernel_size=3)

    # Reading Loop
    frame_counter = 0
    while True:
        print('Frame: {}'.format(frame_counter))
        frame_counter += 1
        ret, frame = vid.read()
        if frame is None:
            break

        # Frame to Grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get the foreground mask
        f_mask = subtractor.get_mask(frame)

        # Some delation before flood filling
        f_mask = delayer_A.delay(f_mask)

        # Add two border lines for flood filling
        f_mask_elarg = np.zeros((f_mask.shape[0]+2, f_mask.shape[1]), np.uint8)
        f_mask_elarg[1: -1, :] = f_mask

        # Store the flood mask
        flood_msk = np.zeros((frame.shape[0] + 4, frame.shape[1] + 2), np.uint8)

        # Apply flood filling
        flooded = f_mask_elarg.copy()
        cv2.floodFill(flooded, flood_msk, (0, 0), 255)

        # Take the inverse
        flooded = cv2.bitwise_not(flooded)

        # Combine images
        f_mask = f_mask_elarg | flooded

        # Delete the two lines
        f_mask = f_mask[1:-1, :]

        # Erosion and deletion (opening) to delete the noise
        f_mask = eroder_A.erode(f_mask)
        f_mask = delayer_B.delay(f_mask)

        # Get final masked image
        final_img = cv2.bitwise_and(frame, frame, mask=f_mask)

        # Save it
        if save_frames:
            cv2.imwrite('output_frames/{}_{}.jpg'.format(vid_name, str(frame_counter).zfill(5)), final_img)

        # Display it
        if display:
            cv2.imshow('window', final_img)
            cv2.waitKey(1)
        # Note: to use key press to continue: cv1.waitKey()

    # Close the program
    vid.release()
    if display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

