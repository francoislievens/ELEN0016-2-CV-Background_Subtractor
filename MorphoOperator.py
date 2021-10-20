import numpy as np

class morpho_operator():

    def __init__(self, kernel_size=3, img_size=(240, 1600)):

        # Build the square kernel
        self.size = kernel_size
        self.kernel = np.ones((self.size, self.size), dtype=np.uint8) * 255

        # Store size of the image
        self.img_size = img_size

        # Compute the padding for each sides
        self.pad = int((self.size - 1) / 2)

        # Vectorize the function:
        self.eroder = np.vectorize(self.sub_erode)
        self.delayer = np.vectorize(self.sub_delay)


        # Enum of all index
        self.idx_enum = range(0, int(img_size[0] * img_size[1]))
        self.idx_enum_coord = []
        for y in range(0, img_size[0]):
            for x in range(0, img_size[1]):
                self.idx_enum_coord.append((x, y))

        self.padded_mat = None

    def erode(self, img):

        # Add padding
        padded = np.zeros((img.shape[0] + 2 * self.pad, img.shape[1] + 2 * self.pad), dtype=np.uint8)
        padded[self.pad:self.pad + img.shape[0], self.pad:self.pad + img.shape[1]] = img
        self.padded_mat = padded

        # Apply the kernel
        opt = np.array(self.eroder(self.idx_enum), dtype=np.uint8).reshape(img.shape)
        return opt


    def sub_erode(self, idx):

        # Get coordinates
        x, y = self.idx_enum_coord[idx]
        min_val = self.padded_mat[y:y+self.kernel.shape[0], x:x+self.kernel.shape[1]].min()

        return min_val

    def delay(self, img):

        # Add padding
        padded = np.zeros((img.shape[0] + 2 * self.pad, img.shape[1] + 2 * self.pad), dtype=np.uint8)
        padded[self.pad:self.pad + img.shape[0], self.pad:self.pad + img.shape[1]] = img
        self.padded_mat = padded

        # Apply the kernel
        opt = np.array(self.delayer(self.idx_enum), dtype=np.uint8).reshape(img.shape)
        return opt

    def sub_delay(self, idx):

        # Get coordinates
        x, y = self.idx_enum_coord[idx]
        max_val = self.padded_mat[y:y+self.kernel.shape[0], x:x+self.kernel.shape[1]].max()

        return max_val