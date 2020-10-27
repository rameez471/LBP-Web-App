import skimage
from os import path
from skimage.feature import local_binary_pattern
from skimage.io import imread

class Face:
    def __init__(self,app):
        self.storage = app.config['storage']
        self.db = app.db
        self.faces = []
        self.know_faces_histogram = []
        self.face_user_keys = {}
        self.load_all()

    def load_user_by_index_key(self, index_key=0):

        key_str = str(index_key)

        if key_str in self.face_user_keys:
            return self.face_user_keys[key_str]
        return None

    def load_train_file_by_name(self,name):
        trained_storage = path.join(self.storage,'trained')
        return path.join(trained_storage, name)

    def load_unknown_file_by_name(self, name):
        unknown_storage = path.join(self.storage,'unknown')
        return path.join(unknown_storage, name)

    def load_all(self):

        results = self.db.select('SELECT faces.id, faces.user_id, faces.filename, faces.created FROM faces')

        for row in results:

            user_id = row[1]
            filename = row[2]

            face = {
                'id': row[0],
                'user_id': user_id,
                'filename': filename,
                'created': row[3]
            }
            self.faces.append(face)
            face_image = imread(self.load_train_file_by_name)
            face_image_encoding = 



def blockshaped(arr, nrows, ncols):

	h, w = arr.shape
	return (arr.reshape(h//nrows, nrows, -1, ncols)
			.swapaxes(1,2)
			.reshape(-1, nrows, ncols))

# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.histogram.html
def getHistogram(imgArray):
	hist, bin_edges = numpy.histogram(imgArray, density=True)
	return hist


# Perform LBP with multiblock
def LBP(img): 
	lbp_value = local_binary_pattern(img, 8, 1)

	# Split img into 10*10 blocks
	shaped = blockshaped(lbp_value, 10, 13)

	# Calculate the histogram for each block
	xBlocks = []
	for s in shaped:
		xBlocks.append(getHistogram(s))

	return numpy.concatenate(xBlocks)