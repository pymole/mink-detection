import Augmentor
import glob
import os
from PIL import Image
import shutil
import argparse


def convert_to_jpg(src, dest):
    # convert to jpg
    for image_name in os.listdir(src):
        image_path = os.path.join(dest, image_name)
        if os.path.isfile(image_path):
            im = Image.open(image_path)
            rgb_im = im.convert('RGB')
            rgb_im.save(os.path.splitext(image_path)[0] + '.jpg')


def main(args):
    p = Augmentor.Pipeline(args.input, output_directory=args.output)

    p.flip_left_right(probability=1.0)
    p.process()

    # move original images to output directory
    for image_path in glob.glob(os.path.join(args.input, '*.jpg')):
        image_to = os.path.join(args.output, os.path.basename(image_path))
        shutil.copyfile(image_path, image_to)


if __name__ == '__main__':
    # python augment.py --input images/train1/mink --output images/train1_augmented/mink
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)

    main(parser.parse_args())
