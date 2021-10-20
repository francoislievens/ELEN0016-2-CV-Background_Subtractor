import numpy as np


class GaussianSubtractor():

    def __init__(self, window_size=100, img_shape=(240, 1600), std_threshold=2.5):

        self.window_size = window_size
        self.img_shape = img_shape
        self.threshold_multiplier = std_threshold  # The number of std to pass to be chosen for the mask

        self.buffer = np.zeros((window_size, img_shape[0], img_shape[1]))

        self.initialized = False

        self.index = 0

    def push_init_frame(self, frame):

        if self.initialized:
            return True

        self.buffer[self.index, :, :] = np.array(frame)
        self.index += 1

        # If buffer is full
        if self.index == self.window_size:
            self.index = 0
            self.initialized = True
            return True
        else:
            return False

    def get_mask(self, frame):

        # Get the frame as np array
        frame = np.array(frame)

        # Compute mean and standard deviation matrix
        mean = np.mean(self.buffer, axis=0)
        std = np.std(self.buffer, axis=0)

        # Get threshold array
        threshold = np.reshape(std * self.threshold_multiplier, (1, self.img_shape[0], self.img_shape[1]))

        # Center and abs value
        centered = np.reshape(np.abs(np.subtract(frame, mean)), (1, self.img_shape[0], self.img_shape[1]))

        # Concatenate both
        concat = np.concatenate((threshold, centered), axis=0)

        mask = np.array(np.argmax(concat, axis=0) * 255, np.uint8)

        # Finally add the new image in the buffer
        self.buffer[self.index, :, :] = frame
        self.index += 1
        if self.index >= self.window_size:
            self.index = 0

        return mask
