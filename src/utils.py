import requests
import os
import zipfile

def fetch_and_unpack_files_from_url(url, source_name=None, base_dir='data/raw'):

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    zip_filename = url.split('/')[-1]

    if source_name is None:
        source_name = zip_filename.split('.')[0]

    extract_to = os.path.join(base_dir, source_name)

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    zip_path = os.path.join(extract_to, zip_filename)
    print(f'Load data from {url} to {zip_path}')

    response = requests.get(url)
    with open(zip_path, 'wb') as f:
        f.write(response.content)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    print(f'Data from {url} has been successfully loaded to {extract_to}')

    return extract_to
