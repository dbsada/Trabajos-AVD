'''
Este módulo contiene clases para extraer características HOG, LBP y histogramas de color de imágenes, utilizadas en el proyecto de búsqueda de imágenes.
Es una reestructuración del código usado en los notebooks de la asignatura.
'''

from skimage.feature import hog, local_binary_pattern
import numpy as np
import librosa
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
import hashlib

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

def _extract_audio_features(path: Path) -> dict:
    import warnings
    warnings.filterwarnings("ignore")
    
    features = {}

    # Metadata
    try:
        duration_sec = float(librosa.get_duration(path=str(path)))
    except Exception:
        duration_sec = np.nan

    features["duration_sec"] = duration_sec
    features["duration_min"] = duration_sec / 60 if not np.isnan(duration_sec) else np.nan
    features["sample_rate"] = 44100.0

    # Seed estable (MUY IMPORTANTE)
    seed = int(hashlib.md5(str(path).encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)

    try:
        if not np.isnan(duration_sec) and duration_sec > 30:
            offset = rng.uniform(0, duration_sec - 30)
        else:
            offset = 0.0

        y, sr = librosa.load(path, sr=22050, mono=True, duration=30, offset=offset)
    except Exception:
        return features

    # HPSS
    y_harmonic, y_percussive = None, None
    try:
        y_harmonic, y_percussive = librosa.effects.hpss(y)
    except:
        pass

    # Helper
    def safe_stats(name, arr):
        if arr is None or not isinstance(arr, np.ndarray):
            features[f"{name}_mean"] = np.nan
            features[f"{name}_std"] = np.nan
            return
        try:
            features[f"{name}_mean"] = float(np.mean(arr))
            features[f"{name}_std"] = float(np.std(arr))
        except:
            features[f"{name}_mean"] = np.nan
            features[f"{name}_std"] = np.nan

    # Features
    try:
        safe_stats("rms", librosa.feature.rms(y=y))
        safe_stats("spectral_centroid", librosa.feature.spectral_centroid(y=y, sr=sr))
        safe_stats("spectral_rolloff", librosa.feature.spectral_rolloff(y=y, sr=sr))
        safe_stats("spectral_bandwidth", librosa.feature.spectral_bandwidth(y=y, sr=sr))
        safe_stats("spectral_contrast", librosa.feature.spectral_contrast(y=y, sr=sr))
        safe_stats("spectral_flatness", librosa.feature.spectral_flatness(y=y))
        safe_stats("zcr", librosa.feature.zero_crossing_rate(y))
    except:
        pass

    # MFCC
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            safe_stats(f"mfcc_{i}", mfcc[i])
    except:
        for i in range(13):
            safe_stats(f"mfcc_{i}", None)

    # Harmonic
    if y_harmonic is not None:
        try:
            safe_stats("chroma", librosa.feature.chroma_stft(y=y_harmonic, sr=sr))
            safe_stats("tonnetz", librosa.feature.tonnetz(y=y_harmonic, sr=sr))
        except:
            pass
    else:
        safe_stats("chroma", None)
        safe_stats("tonnetz", None)

    # Rhythm
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features["tempo_bpm"] = float(np.squeeze(tempo))
    except:
        features["tempo_bpm"] = np.nan

    try:
        safe_stats("tempogram", librosa.feature.tempogram(y=y, sr=sr))
    except:
        safe_stats("tempogram", None)

    if y_percussive is not None:
        try:
            safe_stats("onset_strength", librosa.onset.onset_strength(y=y_percussive, sr=sr))
        except:
            safe_stats("onset_strength", None)
    else:
        safe_stats("onset_strength", None)

    return features

def extract_audio_features(paths: list[Path], n_jobs=-1, backend="loky"):
    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_extract_audio_features)(p) for p in tqdm(paths)
    )
    return results

