import json
#
'''
Links to download JSONs:
https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.json.tar.gz
https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz
https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz

Links are from here: https://github.com/visipedia/inat_comp/tree/master/2021
'''

path_to_labels = './datasets/inat/train_mini.json'
path_to_bird_images = './datasets/inat/birds_train'
# path_to_labels = '../datasets/inat/val.json'
# path_to_bird_images = '../datasets/inat/birds_val'

import os
if not os.path.exists(path_to_bird_images):
    os.makedirs(path_to_bird_images)


with open(path_to_labels, 'r') as f:
    D = json.load(f)

'''
Note: All JSON files (train.json, train_mini.json, val.json, etc.) should have
the same values for D['categories'], since this is just category metadata and
all of them have the same categories.
'''

'''
It happens to be true that the entries of D['categories'] are sorted, so
that the metadata for category_id j can be found in position j. Not always the
case for COCO-formatted data, but following block demonstrates that it's true
here.
'''
for i in range(len(D['categories'])):
    assert i == D['categories'][i]['id']

'''
It also happens to be true that the entries of D['images'] correspond to the
entries of D['annotations'], but that is also not true in general for COCO
formatted data.
'''
assert len(D['annotations']) == len(D['images'])
for i in range(len(D['annotations'])):
    assert D['annotations'][i]['image_id'] == D['images'][i]['id']

# Grab bird category IDs:
bird_category_id_list = []
for i in range(len(D['categories'])):
    if D['categories'][i]['class'] == 'Aves':
        bird_category_id_list.append(i)
        # We can use i because we know that it's the same as the category ID,
        # see comments above.
print(f'# categories: {len(bird_category_id_list)}')

# Get list of image IDs that correspond to bird categories:
# Note, the image IDs do not correspond to indices - i.e.  D['images'][i]['id'] != i.
bird_image_id_list = []
for i in range(len(D['annotations'])):
    if D['annotations'][i]['category_id'] in bird_category_id_list:
        bird_image_id_list.append(D['annotations'][i]['image_id'])
print(f'# images: {len(bird_image_id_list)}')

print('test')

from tqdm import tqdm

# save all bird images to a path_to_bird_images folder
for bird in tqdm(bird_image_id_list):
    for image in D['images']:
        if image['id'] == bird:
            filename = image['file_name']
            # split filename to get folder
            base_folder = filename.split('/')[0]
            specific_folder = filename.split('/')[1]
            filename = filename.split('/')[2]
            # create folder if it doesn't exist
            if not os.path.exists(os.path.join(path_to_bird_images, specific_folder)):
                os.makedirs(os.path.join(path_to_bird_images, specific_folder))
            # copy file to new folder
            os.system(f'cp ./datasets/inat/train/{base_folder}/{specific_folder}/{filename} {path_to_bird_images}/{specific_folder}/{filename}')

#

path_to_labels = './datasets/inat/val.json'
path_to_bird_images = './datasets/inat/birds_val'

import os
if not os.path.exists(path_to_bird_images):
    os.makedirs(path_to_bird_images)

with open(path_to_labels, 'r') as f:
    D = json.load(f)

for i in range(len(D['categories'])):
    assert i == D['categories'][i]['id']

'''
It also happens to be true that the entries of D['images'] correspond to the
entries of D['annotations'], but that is also not true in general for COCO
formatted data.
'''
assert len(D['annotations']) == len(D['images'])
for i in range(len(D['annotations'])):
    assert D['annotations'][i]['image_id'] == D['images'][i]['id']

# Grab bird category IDs:
bird_category_id_list = []
for i in range(len(D['categories'])):
    if D['categories'][i]['class'] == 'Aves':
        bird_category_id_list.append(i)
        # We can use i because we know that it's the same as the category ID,
        # see comments above.
print(f'# categories: {len(bird_category_id_list)}')

# Get list of image IDs that correspond to bird categories:
# Note, the image IDs do not correspond to indices - i.e.  D['images'][i]['id'] != i.
bird_image_id_list = []
for i in range(len(D['annotations'])):
    if D['annotations'][i]['category_id'] in bird_category_id_list:
        bird_image_id_list.append(D['annotations'][i]['image_id'])
print(f'# images: {len(bird_image_id_list)}')

print('test')

from tqdm import tqdm

# save all bird images to a path_to_bird_images folder
for bird in tqdm(bird_image_id_list):
    for image in D['images']:
        if image['id'] == bird:
            filename = image['file_name']
            # split filename to get folder
            base_folder = filename.split('/')[0]
            specific_folder = filename.split('/')[1]
            filename = filename.split('/')[2]
            # create folder if it doesn't exist
            if not os.path.exists(os.path.join(path_to_bird_images, specific_folder)):
                os.makedirs(os.path.join(path_to_bird_images, specific_folder))
            # copy file to new folder
            os.system(f'cp ./datasets/inat/val/{base_folder}/{specific_folder}/{filename} {path_to_bird_images}/{specific_folder}/{filename}')