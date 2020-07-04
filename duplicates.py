import os
from PIL import Image
from collections import defaultdict
import hashlib
import argparse


def main(args):
    duplicates = defaultdict(list)

    for img_name in os.listdir(args.input):
        img_path = os.path.join(args.input, img_name)

        # open image
        img = Image.open(img_path)

        # resize with constant width
        wpercent = args.base_width / float(img.size[0])
        hsize = int(img.size[1] * wpercent)

        img = img.resize((args.base_width, hsize), Image.ANTIALIAS)

        # convert to greyscale without alpha
        img = img.convert('L')

        # round each pixel to nearest to reduce color number
        img = img.quantize(4)
        img_hash = hashlib.md5(img.tobytes()).hexdigest()

        duplicates[img_hash].append(img_path)

    for images in duplicates.values():
        if len(images) > 1:
            print(images)


if __name__ == '__main__':
    # python duplicates.py --input images/train/mink --output output --base_width 150
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--base_width', type=int, default=150)

    main(parser.parse_args())