import os
import argparse


def main(args):
    for i, filename in enumerate(os.listdir(args.folder)):
        os.rename(os.path.join(args.folder, filename), os.path.join(args.folder, str(i) + '.jpg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str)

    main(parser.parse_args())
