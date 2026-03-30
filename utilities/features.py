'''
Este módulo contiene clases para extraer características HOG, LBP y histogramas de color de imágenes, utilizadas en el proyecto de búsqueda de imágenes.
Es una reestructuración del código usado en los notebooks de la asignatura.
'''

from skimage.feature import hog, local_binary_pattern
import numpy as np

class HOGfeatures:
    """
    Class to extract HOG features from an image
    """
    def __init__(self, **kwargs):
        self.orientations = kwargs.get('orientations', 6)
        self.pixels_per_cell = kwargs.get('pixels_per_cell', (4, 4))
        self.cells_per_block = kwargs.get('cells_per_block', (2, 2))

    def extract(self, image, visualize=False):
        """
        Extracts HOG features from an image
        """
        return hog(image,
                    orientations=self.orientations,
                    pixels_per_cell=self.pixels_per_cell,
                    cells_per_block=self.cells_per_block,
                    visualize=visualize,
                    channel_axis=-1)
        
    def mean(self, image):
        """
        Gets the mean of the HOG response map
        """
        fd = self.extract(image)
        return np.mean(fd) if len(fd) else 0
    
    def std(self, image):
        """
        Gets the standard deviation of the HOG response map
        """
        fd = self.extract(image)
        return np.std(fd) if len(fd) else 0

class LBPfeatures:
    """
    Class to extract LBP features from an image
    """
    def __init__(self, P=8, R=1):
        self.P = P
        self.R = R

    def extract(self, image):
        """
        Extracts LBP features from an image
        """
        lbp = local_binary_pattern(image, self.P, self.R, method='uniform')
        return lbp
    
    def mean(self, image):
        """
        Gets the mean of the LBP response map
        """
        lbp = self.extract(image)
        return np.mean(lbp) if len(lbp) else 0
    
    def std(self, image):
        """
        Gets the standard deviation of the LBP response map
        """
        lbp = self.extract(image)
        return np.std(lbp) if len(lbp) else 0
    
class COLORfeatures:
    """
    Class to extract color histogram features from an image
    """
    def __init__(self, bins=100):
        self.bins = bins

    def extract(self, image):
        """
        Extracts color histogram features from an image
        """
        hist_list = []
        for ch in range(3):  # R, G, B
            hist, _ = np.histogram(image[..., ch], bins=self.bins, range=(0, 255))
            hist_list.append(hist)
        return hist_list
    
    def mean(self, image):
        """
        Gets the mean of the color histogram response
        """
        hist_list = self.extract(image)
        return np.mean(hist_list) if len(hist_list) else 0
    
    def std(self, image):
        """
        Gets the standard deviation of the color histogram response
        """
        hist_list = self.extract(image)
        return np.std(hist_list) if len(hist_list) else 0