import os
import argparse


def main(args):
    files = os.listdir(args.folder)

    end = args.start + len(files)
    skip = set()

    for idx in range(end - 1, args.start - 1, -1):
        filename, _ = os.path.splitext(os.path.basename(files[idx]))
        if filename.isdigit():
            number = int(filename)
            if args.start <= number < end:
                skip.add(number)
                files.pop(idx)

    for i, filename in zip(set(range(args.start, end)) - skip, files):
        new_filename = str(i) + '.jpg'
        os.rename(os.path.join(args.folder, filename),
                  os.path.join(args.folder, new_filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str)
    parser.add_argument('--start', type=int, default=0)

    main(parser.parse_args())
