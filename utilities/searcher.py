"""
Módulo searcher: implementa búsqueda de imágenes similares en base de datos de características.
Soporta múltiples métricas de distancia: chi-squared, euclidean, cosine, manhattan.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from skimage.color import rgb2gray


class Searcher:
    """
    Clase para búsqueda de imágenes similares usando vectores de características.
    Soporta múltiples métricas de distancia.
    """

    def __init__(self, index, metric="chi2"):
        """
        Inicializa el searcher con un índice de características.

        Args:
            index (dict): Diccionario donde las claves son paths y valores son vectores de características.
            metric (str): Métrica de distancia a usar. Opciones: 'chi2', 'euclidean', 'cosine', 'manhattan'.
        """
        self.index = index
        self.metric = metric

        if metric not in ["chi2", "euclidean", "cosine", "manhattan"]:
            raise ValueError(f"Métrica no soportada: {metric}. Opciones: chi2, euclidean, cosine, manhattan")

    def search(self, queryFeatures, top_k=10):
        """
        Busca los top_k imágenes más similares a las características de la query.

        Args:
            queryFeatures (np.ndarray): Vector de características de la imagen query.
            top_k (int): Número de resultados más similares a devolver.

        Returns:
            list: Lista de tuplas (distancia, path) ordenadas ascendentemente por distancia.
        """
        results = {}

        for (path, features) in self.index.items():
            d = self._compute_distance(features, queryFeatures)
            results[path] = d

        # Ordenar resultados por distancia (menor primero)
        ranked = sorted([(v, k) for (k, v) in results.items()])

        return ranked[:top_k]

    def _compute_distance(self, histA, histB, eps=1e-10):
        """
        Calcula la distancia entre dos vectores usando la métrica especificada.

        Args:
            histA (np.ndarray): Vector de características 1.
            histB (np.ndarray): Vector de características 2.
            eps (float): Valor pequeño para evitar divisiones por cero.

        Returns:
            float: Distancia entre los vectores.
        """
        if self.metric == "chi2":
            return self._chi2_distance(histA, histB, eps)
        elif self.metric == "euclidean":
            return self._euclidean_distance(histA, histB)
        elif self.metric == "cosine":
            return self._cosine_distance(histA, histB, eps)
        elif self.metric == "manhattan":
            return self._manhattan_distance(histA, histB)

    @staticmethod
    def _chi2_distance(histA, histB, eps=1e-10):
        """
        Distancia chi-squared (chi-cuadrado).
        Comúnmente usada para comparar histogramas en visión por computadora.
        """
        histA = np.asarray(histA, dtype=np.float32)
        histB = np.asarray(histB, dtype=np.float32)
        num = (histA - histB) ** 2
        den = histA + histB + eps
        return float(0.5 * np.sum(num / den))

    @staticmethod
    def _euclidean_distance(histA, histB):
        """
        Distancia euclidiana.
        """
        return np.sqrt(np.sum((histA - histB) ** 2))

    @staticmethod
    def _cosine_distance(histA, histB, eps=1e-10):
        """
        Distancia coseno (1 - similitud coseno).
        Útil para vectores normalizados.
        """
        norm_a = np.linalg.norm(histA)
        norm_b = np.linalg.norm(histB)

        if norm_a < eps or norm_b < eps:
            return 1.0  # Vectores cero devuelven distancia máxima

        similarity = np.dot(histA, histB) / (norm_a * norm_b)
        # Asegurarse de que similarity está en [-1, 1] (por errores numéricos)
        similarity = np.clip(similarity, -1.0, 1.0)

        return 1.0 - similarity

    @staticmethod
    def _manhattan_distance(histA, histB):
        """
        Distancia Manhattan (L1).
        """
        return np.sum(np.abs(histA - histB))


class ImageSearchPipeline:
    """
    Pipeline completo de búsqueda de imágenes.
    Encapsula extracción de características, construcción de índice y búsqueda.
    """

    def __init__(self, db, hog_extractor, lbp_extractor, color_extractor):
        """
        Inicializa el pipeline.

        Args:
            db (dict): Base de datos cargada con joblib (contiene labels, paths, hog, lbp, color, target_size).
            hog_extractor: Objeto extractor de HOG.
            lbp_extractor: Objeto extractor de LBP.
            color_extractor: Objeto extractor de COLOR.
        """
        self.db = db
        self.hog_extractor = hog_extractor
        self.lbp_extractor = lbp_extractor
        self.color_extractor = color_extractor

    @staticmethod
    def _to_rgb_array(x):
        """Convierte entrada a array RGB numpy."""
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, (str, Path)):
            with Image.open(x) as im:
                return np.asarray(im.convert("RGB"))
        if hasattr(x, "convert"):  # PIL Image
            return np.asarray(x.convert("RGB"))
        return np.asarray(x)

    @staticmethod
    def _l2_normalize(v):
        """Normaliza un vector a norma L2."""
        v = v.astype(np.float32)
        n = np.linalg.norm(v)
        return v / (n + 1e-12)

    def _lbp_histogram(self, image_gray_uint8):
        """Calcula histograma uniforme de LBP."""
        lbp_map = self.lbp_extractor.extract(image_gray_uint8)
        hist, _ = np.histogram(
            lbp_map.ravel(),
            bins=np.arange(0, self.lbp_extractor.P + 3),
            range=(0, self.lbp_extractor.P + 2),
        )
        hist = hist.astype(np.float32)
        return hist / (hist.sum() + 1e-12)

    def _extract_query_vectors(self, query_path):
        """Extrae los tres tipos de vectores (HOG, LBP, COLOR) de una imagen query."""
        arr = self._to_rgb_array(query_path)
        arr = np.asarray(Image.fromarray(arr).resize(self.db["target_size"], Image.Resampling.BILINEAR))
        arr_gray = (rgb2gray(arr) * 255).astype(np.uint8)

        q_hog = self.hog_extractor.extract(arr).astype(np.float32)
        q_lbp = self._lbp_histogram(arr_gray).astype(np.float32)
        q_color = np.concatenate(self.color_extractor.extract(arr)).astype(np.float32)
        q_color = q_color / (q_color.sum() + 1e-12)

        return {
            "hog": q_hog,
            "lbp": q_lbp,
            "color": q_color,
        }

    @staticmethod
    def _validate_feature_blocks(feature_mode):
        """Valida y normaliza la selección de descriptores. Solo acepta lista/tupla."""
        allowed = {"hog", "lbp", "color"}

        if not isinstance(feature_mode, (list, tuple)):
            raise TypeError(
                "feature_mode debe ser una lista o tupla de descriptores, "
                "por ejemplo ['color', 'lbp', 'hog']."
            )

        if len(feature_mode) == 0:
            raise ValueError("feature_mode no puede estar vacío.")

        normalized = [str(x).strip().lower() for x in feature_mode]
        invalid = [x for x in normalized if x not in allowed]
        if invalid:
            raise ValueError(
                f"Descriptores no válidos: {invalid}. Opciones válidas: {sorted(allowed)}"
            )

        return normalized

    def _compose_feature_vector(self, query_vectors, feature_mode):
        """Compone el vector concatenando los descriptores en el orden indicado."""
        blocks_to_use = self._validate_feature_blocks(feature_mode)
        blocks = [self._l2_normalize(query_vectors[p]) for p in blocks_to_use]
        return np.concatenate(blocks)

    def _build_search_index(self, feature_mode):
        """Construye índice normalizado para búsqueda usando los descriptores seleccionados."""
        blocks_to_use = self._validate_feature_blocks(feature_mode)

        index = {}
        for i, path in enumerate(self.db["paths"]):
            parts = [self._l2_normalize(self.db[p][i]) for p in blocks_to_use]
            index[path] = np.concatenate(parts).astype(np.float32)

        return index

    def search(self, query_path, feature_mode=("color", "lbp", "hog"), metric="chi2", top_k=10):
        """
        Realiza búsqueda de imágenes similares a una imagen query.

        Args:
            query_path (str o Path): Ruta a la imagen query.
            feature_mode (list[str] | tuple[str, ...]): Lista/tupla de descriptores en orden.
                Ejemplos: ['hog'], ['color', 'lbp'], ['color', 'lbp', 'hog'].
            metric (str): Métrica de distancia. 
                Opciones: 'chi2', 'euclidean', 'cosine', 'manhattan'.
            top_k (int): Número de resultados a devolver.

        Returns:
            pd.DataFrame: DataFrame con columnas [rank, path, label, distance].
        """
        query_vectors = self._extract_query_vectors(query_path)
        query_features = self._compose_feature_vector(query_vectors, feature_mode)

        index = self._build_search_index(feature_mode)
        searcher = Searcher(index, metric=metric)
        ranked = searcher.search(query_features, top_k=top_k)

        results = []
        path_to_label = {p: l for p, l in zip(self.db["paths"], self.db["labels"])}

        for rank, (dist, path) in enumerate(ranked, start=1):
            results.append({
                "rank": rank,
                "path": path,
                "label": path_to_label[path],
                "distance": dist,
            })

        return pd.DataFrame(results)


class DeepImageSearchPipeline:
    """Pipeline de búsqueda para embeddings de deep learning precomputados."""

    def __init__(self, db, query_feature_extractor, normalize=True):
        """
        Args:
            db (dict): Debe contener keys: 'paths', 'labels', 'features'.
            query_feature_extractor (callable): Función query_path -> vector 1D.
            normalize (bool): Si True aplica normalización L2 a base y query.
        """
        self.db = db
        self.query_feature_extractor = query_feature_extractor
        self.normalize = normalize

        required = {"paths", "labels", "features"}
        missing = sorted(list(required - set(db.keys())))
        if missing:
            raise ValueError(f"Faltan claves en db: {missing}")

    @staticmethod
    def _l2_normalize(v):
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(v)
        return v / (n + 1e-12)

    def _build_search_index(self):
        index = {}
        for path, vec in zip(self.db["paths"], self.db["features"]):
            v = np.asarray(vec, dtype=np.float32).reshape(-1)
            if self.normalize:
                v = self._l2_normalize(v)
            index[path] = v
        return index

    def _extract_query_vector(self, query_path):
        q = self.query_feature_extractor(query_path)
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        if self.normalize:
            q = self._l2_normalize(q)
        return q

    def search(self, query_path, metric="cosine", top_k=10):
        """Busca imágenes similares y devuelve DataFrame [rank, path, label, distance]."""
        query_features = self._extract_query_vector(query_path)
        index = self._build_search_index()

        searcher = Searcher(index, metric=metric)
        ranked = searcher.search(query_features, top_k=top_k)

        path_to_label = {p: l for p, l in zip(self.db["paths"], self.db["labels"])}
        results = []
        for rank, (dist, path) in enumerate(ranked, start=1):
            results.append(
                {
                    "rank": rank,
                    "path": path,
                    "label": path_to_label[path],
                    "distance": dist,
                }
            )

        return pd.DataFrame(results)
