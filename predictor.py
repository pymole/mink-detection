import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import pickle
import time
import datetime
import threading
import zipfile
import signal
import shutil
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.resnet50 import ResNet50
import cv2


class ImageLoader(Sequence):
    def __init__(self, image_dir, batch_size, prepare_func):
        self.batch_size = batch_size
        self.image_paths = []
        self.prepare_func = prepare_func

        for filename in os.listdir(image_dir):
            filepath = os.path.join(image_dir, filename)
            if os.path.isfile(filepath):
                self.image_paths.append(filepath)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_paths = self.image_paths[index * self.batch_size: (index + 1) * self.batch_size]

        images = [
            self.prepare_func(cv2.resize(cv2.imread(image_path), (224, 224)))
            for image_path in batch_paths
        ]
        images = np.array(images)

        return images


class Predictor:
    def __init__(self, models_path, input_directory, archive_directory, stats_on_every):
        # loading models
        self.resnet_model = ResNet50(weights='imagenet', include_top=False, classes=False)

        with open(os.path.join(models_path, 'scaler'), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(models_path, 'pca'), 'rb') as f:
            self.pca = pickle.load(f)
        with open(os.path.join(models_path, 'oc_svm'), 'rb') as f:
            self.oc_svm = pickle.load(f)

        self.input_directory = input_directory
        self.archive_directory = archive_directory
        self.stats_on_every = stats_on_every
        self.positive_count = 0
        self.total_count = 0

    def extract_resnet(self, X):
        # TODO fix 1 image bug
        features_array = self.resnet_model.predict(X)
        features_array = np.reshape(features_array,
                                    (-1, features_array.shape[1] * features_array.shape[2] *
                                     features_array.shape[3]))

        return features_array

    def run(self):
        loader = ImageLoader(self.input_directory, 32, tf.keras.applications.resnet50.preprocess_input)

        if len(loader) == 0:
            return

        features = self.extract_resnet(loader)
        features = self.scaler.transform(features)
        features = self.pca.transform(features)

        pred = self.oc_svm.predict(features)

        self.info(loader.image_paths, pred)
        self.move(loader.image_paths)
        self.stats(pred)

    def move(self, file_paths):
        os.makedirs(self.archive_directory, exist_ok=True)

        for src in file_paths:
            dst = os.path.join(self.archive_directory, os.path.basename(src))
            shutil.move(src, dst)

    def info(self, filenames, pred):
        for filename, y in zip(filenames, pred):
            y = 'Норка обнаружена' if y == 1 else 'Норка не обнаружена'
            print(f'{y}: {filename}')

    def stats(self, pred):
        old_total = self.total_count

        self.positive_count += (pred == 1).sum()
        self.total_count += len(pred)

        if old_total // self.stats_on_every < self.total_count // self.stats_on_every:
            print(f'Найдено {self.positive_count} норок среди {self.total_count} обработанных картинок.')


class Archiver:
    def __init__(self, directory, batch_size):
        self.directory = directory
        self.batch_size = batch_size

    def archive(self):
        if not os.path.exists(self.directory):
            return

        # construct paths, get only files, sort by creation time
        filepaths = [
            filepath
            for filename in os.listdir(self.directory)
            if os.path.splitext(filename)[1] != '.zip' and
               os.path.isfile(filepath := os.path.join(self.directory, filename))
        ]
        filepaths.sort(key=os.path.getmtime)

        for batch_index in range(0, len(filepaths) // self.batch_size * self.batch_size, self.batch_size):
            batch = filepaths[batch_index: (batch_index + 1) * self.batch_size]

            archive_name = os.path.join(self.directory,
                                        str(datetime.datetime.fromtimestamp(os.path.getmtime(batch[0]))) + '.zip')
            with zipfile.ZipFile(archive_name, 'w') as z:
                for filepath in batch:
                    z.write(filepath, arcname=os.path.basename(filepath))

            for filepath in batch:
                os.remove(filepath)

            print(f'Создан архив {archive_name}')


class Loop(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thread_exit = False


class PredictorLoop(Loop):
    def __init__(self, models_path, input_directory, archive_directory, timeout, stats_on_every, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor = Predictor(models_path, input_directory, archive_directory, stats_on_every)
        self.timeout = timeout

    def run(self):
        while not self.thread_exit:
            self.predictor.run()
            time.sleep(self.timeout)


class ArchiverLoop(Loop):
    def __init__(self, directory, timeout, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.archiver = Archiver(directory, batch_size)
        self.timeout = timeout

    def run(self):
        last_archive_at = time.time()

        while not self.thread_exit:
            self.archiver.archive()

            last_archive_at += self.timeout
            time.sleep(max(0.0, last_archive_at - time.time()))


def main(args):
    archiver_loop = ArchiverLoop(args.archive_directory, args.archive_period, args.batch_size, daemon=True)
    predictor_loop = PredictorLoop(args.models, args.input, args.archive_directory,
                                   args.prediction_timeout, args.stats_on_every, daemon=True)

    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    archiver_loop.start()
    predictor_loop.start()

    archiver_loop.join()
    predictor_loop.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_timeout', type=float, default=0.1)
    parser.add_argument('--models', type=str)
    parser.add_argument('--input', type=str)
    parser.add_argument('--archive_period', type=float, required=False)
    parser.add_argument('--archive_directory', type=str, required=False)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--stats_on_every', type=int, default=6)

    main(parser.parse_args())
