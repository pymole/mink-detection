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

        initial_resolution = img.size[0] * img.size[1]

        # resize with constant width and height
        img = img.resize((args.width, args.height), Image.ANTIALIAS)

        # convert to greyscale without alpha
        img = img.convert('L')

        # round each pixel to nearest to reduce color number
        img = img.quantize(4)
        img_hash = hashlib.md5(img.tobytes()).hexdigest()

        duplicates[img_hash].append((img_path, initial_resolution))

    for images in duplicates.values():
        if len(images) > 1:
            print(images)

            # delete all except one with highest resolution
            if args.delete:
                max_res = 0
                best_img_idx = 0
                for idx, (_, res) in enumerate(images):
                    if max_res < res:
                        max_res = res
                        best_img_idx = idx

                images.pop(best_img_idx)
                print('Deleting', images)
                for img_path, _ in images:
                    os.remove(img_path)


if __name__ == '__main__':
    # python duplicates.py --input images/train/mink
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--width', type=int, default=8)
    parser.add_argument('--height', type=int, default=8)
    parser.add_argument('--delete', action='store_true')

    main(parser.parse_args())