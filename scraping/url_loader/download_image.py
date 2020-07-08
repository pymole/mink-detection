import requests
import argparse
import os
import concurrent.futures


def load(url, idx):
    try:
        response = requests.get(url, timeout=3)
    except:
        return None

    return idx, response.content


def main(args):
    with open(args.urls, 'r') as f:
        urls = f.readlines()

    if not os.path.exists('images'):
        os.mkdir('images')

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Start the load operations and mark each future with its URL
        future_to_url = {executor.submit(load, url, i): url for i, url in enumerate(urls)}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                i, data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (url, exc))
            else:
                print('%r page is %d bytes' % (url, len(data)))

            with open(os.path.join('images', str(i) + '.jpg'), 'wb') as f:
                f.write(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--urls', type=str)
    main(parser.parse_args())