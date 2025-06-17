from typing import Generator, Iterable, List, TypeVar
import numpy as np
from sklearn.cluster import KMeans
import cv2


class Classifier:
    def __init__(self):
        self.team_1_color = None
        self.team_2_color = None

    def fit(self, crops: List[np.ndarray]) -> None:
        pixels = []
        for crop in crops:
            resized_crop = cv2.resize(crop, (50, 50))
            hsv_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2HSV)
            pixels.append(hsv_crop.reshape(-1, 3))

        pixels = np.concatenate(pixels, axis=0)

        kmeans = KMeans(n_clusters=2, random_state=0).fit(pixels)

        dominant_colors = kmeans.cluster_centers_

        dominant_hues = dominant_colors[:, 0]
        sorted_hues = np.argsort(dominant_hues)

        self.team_1_color = dominant_colors[sorted_hues[0]]
        self.team_2_color = dominant_colors[sorted_hues[1]]

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        result = []
        for crop in crops:
            resized_crop = cv2.resize(crop, (50, 50))
            hsv_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2HSV)
            avg_color = np.mean(hsv_crop.reshape(-1, 3), axis=0)

            dist_team_1 = np.linalg.norm(avg_color - self.team_1_color)
            dist_team_2 = np.linalg.norm(avg_color - self.team_2_color)

            if dist_team_1 < dist_team_2:
                result.append(0)
            else:
                result.append(1)

        return np.array(result)