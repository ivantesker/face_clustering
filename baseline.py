import numpy as np
import argparse
import logging
import pickle
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import torch
from PIL import Image
from abc import ABC, abstractmethod
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FaceClusterConfig:
    encodings_path: str = "encodings_face_80_new_preproc.pickle"
    clustering_result_path: str = "clustering_result"
    num_jobs: int = -1
    dataset_path: str = "clusters"
    device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EncodingsLoader:
    def __init__(self, encodings_path: str):
        self._encodings_path = Path(encodings_path)
        self._pca = PCA(n_components=256)

    def _load_encodings(self) -> tuple:
        logger.info("Loading encodings...")
        data = pickle.loads(self._encodings_path.read_bytes())
        data = np.array(data)
        encodings = [d["encoding"] for d in data]
        encodings_pca = self._pca.fit_transform(encodings)
        return data, encodings_pca

    def load_encodings(self) -> tuple:
        return self._load_encodings()


class IClusterService(ABC):
    @abstractmethod
    def perform_clustering(self, encodings, num_jobs: int) -> tuple:
        pass


class DBSCANClusterService(IClusterService):
    def __init__(self, clustering_result_path: str):
        self._clustering_result_path = Path(clustering_result_path)
        self._clustering_result_path.mkdir(exist_ok=True)
        self._label_ids = None

    def perform_clustering(self, encodings, num_jobs: int) -> tuple:
        logger.info("Clustering...")
        clt = DBSCAN(metric="euclidean", n_jobs=num_jobs)
        clt.fit(encodings)
        self._label_ids = np.unique(clt.labels_)

        silhouette = silhouette_score(encodings, clt.labels_)
        logger.info(f"Silhouette Score: {silhouette}")
        return clt, self._label_ids

    def get_label_ids(self):
        return self._label_ids

    def move_image(self, image, image_path, label_id):
        path = self._clustering_result_path / f"label{label_id}"
        path.mkdir(exist_ok=True)
        filename = f"{Path(image_path).stem}.jpg"
        image.save(str(path / filename))


class IFaceEncodingService(ABC):
    @abstractmethod
    def encode_faces(self, dataset_path: str) -> list:
        pass


class FaceNetEncodingService(IFaceEncodingService):
    def __init__(self, encodings_path: str, device: str):
        self._encodings_path = Path(encodings_path)
        self._device = device
        from facenet_pytorch import MTCNN, InceptionResnetV1

        self._mtcnn = MTCNN(
            device=device,
            thresholds=[0.8, 0.8, 0.8],
            image_size=160,
            post_process=True,
            margin=0,
        )
        self._resnet = InceptionResnetV1(
            pretrained="casia-webface", device=device
        ).eval()

    def encode_faces(self, dataset_path: str) -> list:
        if self._encodings_path.exists():
            logger.info("Loading existing encodings from pickle...")
            data = pickle.loads(self._encodings_path.read_bytes())
            return data
        else:
            logger.info("Quantifying faces...")
            image_paths = list(Path(dataset_path).glob("**/*.jpg"))
            data = []
            for i, image_path in enumerate(image_paths):
                logger.info("Processing image {}/{}".format(i + 1, len(image_paths)))

                img = Image.open(str(image_path))

                with torch.inference_mode():
                    img_cropped = self._mtcnn(
                        img, save_path=str(Path("result") / image_path.name)
                    )

                if img_cropped is None:
                    continue

                img_cropped = img_cropped.to(self._device)
                with torch.inference_mode():
                    encoding = self._resnet(img_cropped.unsqueeze(0))
                    d = [
                        {"imagePath": str(image_path), "encoding": encoding.tolist()[0]}
                    ]
                    data.extend(d)

            logger.info("Serializing encodings...")
            self._encodings_path.write_bytes(pickle.dumps(data))

            return data


class DeepFaceEncodingService(IFaceEncodingService):
    def __init__(self, encodings_path: str, device: str):
        self._encodings_path = Path(encodings_path)
        self._device = device

    def encode_faces(self, dataset_path: str) -> list:
        from deepface import DeepFace

        if self._encodings_path.exists():
            logger.info("Loading existing encodings from pickle...")
            data = pickle.loads(self._encodings_path.read_bytes())
            return data
        else:
            logger.info("Quantifying faces...")
            image_paths = list(Path(dataset_path).glob("**/*.jpg"))
            data = []
            for i, image_path in enumerate(image_paths):
                logger.info("Processing image {}/{}".format(i + 1, len(image_paths)))
                try:
                    encoding = DeepFace.represent(img_path=str(image_path))
                except:
                    logger.warn("Image skipped, no face")
                    continue
                d = [{"imagePath": str(image_path), "encoding": encoding}]
                data.extend(d)

            logger.info("Serializing encodings...")
            self._encodings_path.write_bytes(pickle.dumps(data))

            return data


class FaceClustering:
    def __init__(
        self,
        config: FaceClusterConfig,
        clustering_service: IClusterService,
        face_encoding_service: IFaceEncodingService,
    ):
        self._config = config
        self._encodings_loader = EncodingsLoader(config.encodings_path)
        self._cluster_service = clustering_service
        self._clustering_result_path = Path(config.clustering_result_path)
        self._face_encoding_service = face_encoding_service

    def _run_clustering(self):
        data = self._face_encoding_service.encode_faces(
            dataset_path=self._config.dataset_path
        )

        _, encodings = self._encodings_loader.load_encodings()
        _, encodings = self._encodings_loader.load_encodings()
        clt, label_ids = self._cluster_service.perform_clustering(
            encodings, self._config.num_jobs
        )

        num_unique_faces = len(np.where(label_ids > -1)[0])
        logger.info("# unique faces: %s", num_unique_faces)

        for label_id in label_ids:
            idxs = np.where(clt.labels_ == label_id)[0]
            idxs = np.random.choice(idxs, size=min(25, len(idxs)), replace=False)

            for i in idxs:
                image_path = data[i]["imagePath"]
                image = Image.open(image_path)
                self._cluster_service.move_image(image, data[i]["imagePath"], label_id)

    def run_clustering(self):
        self._run_clustering()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--dataset",
        default=FaceClusterConfig.dataset_path,
        help="Path to input directory of faces + images",
    )
    parser.add_argument(
        "-e",
        "--encodings",
        default=FaceClusterConfig.encodings_path,
        help="Path to serialized database of facial encodings",
    )

    args = parser.parse_args()

    config = FaceClusterConfig()
    config.dataset_path = args.dataset
    config.encodings_path = args.encodings

    clustering_service = DBSCANClusterService(
        clustering_result_path=config.clustering_result_path
    )
    face_encoding_service = FaceNetEncodingService(
        encodings_path=config.encodings_path, device=config.device
    )
    face_clustering = FaceClustering(config, clustering_service, face_encoding_service)
    face_clustering.run_clustering()
