import os
import argparse
import random
import shutil


def main(args):
    positive = os.listdir(args.positive)
    random.shuffle(positive)

    idx = int(args.positive_train_ratio * len(positive))
    train = positive[:idx]

    positive_test = positive[idx:]
    negative_test = os.listdir(args.negative)

    os.makedirs(args.output, exist_ok=True)

    train_dir = os.path.join(args.output, 'train')
    shutil.rmtree(train_dir, ignore_errors=True)

    train_positive_dir = os.path.join(train_dir, 'positive')
    os.makedirs(train_positive_dir)

    test_dir = os.path.join(args.output, 'test')
    shutil.rmtree(test_dir, ignore_errors=True)

    test_positive_dir = os.path.join(test_dir, 'positive')
    test_negative_dir = os.path.join(test_dir, 'negative')
    os.mkdir(test_dir)
    os.mkdir(test_positive_dir)
    os.mkdir(test_negative_dir)

    for filename in train:
        shutil.copyfile(os.path.join(args.positive, filename), os.path.join(train_positive_dir, filename))

    for filename in positive_test:
        shutil.copyfile(os.path.join(args.positive, filename), os.path.join(test_positive_dir, filename))
    for filename in negative_test:
        shutil.copyfile(os.path.join(args.negative, filename), os.path.join(test_negative_dir, filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default=os.getcwd())
    parser.add_argument('--positive', type=str)
    parser.add_argument('--negative', type=str)
    parser.add_argument('--positive_train_ratio', type=float)

    main(parser.parse_args())
