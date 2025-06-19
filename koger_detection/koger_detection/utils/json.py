import copy
import json
import os
import random

import numpy as np

def create_empty_annotation_json(json_dict):
    """ Only preserve dataset level info"""
    
    new_dict = copy.deepcopy(json_dict)
    new_dict['images'] = []
    new_dict['annotations'] = []
    return new_dict

def get_annotations_for_id(annotation_dicts, image_id):
    """ Get all annotations that go with a given image id.
    
    Args:
        annotation_dicts: val stored in coco dataset under annotation key
        image_id: image id that you want annotations for
        
    Return annotations
    """
    annotations = []
    for annotation_dict in annotation_dicts:
        if annotation_dict['image_id'] == image_id:
            annotations.append(copy.deepcopy(annotation_dict))
    return(annotations)

def deterministic_in_val(image_num, fraction_val):
    """Determine train val split by putting every nth image in val set.
    
    Args:
        image_num: number of the image currently being added to train/val set.
        fraction_val: fraction of the images to be put in the val set
        
    Return True if image should be in validation set.
    """
    if image_num % int(1/fraction_val) == 0:
        return True
    else:
        return False

def stochastic_in_val(fraction_val):
    """Determine if image should be in the validation set with prob. fraction_val.
    
    Args:
        fraction_val: fraction of the images to be put in the val set
        
    Return True if image should be in validation set.
    """
    if random.random() < fraction_val:
        return True
    else:
        return False

def create_train_val_split(json_file, fraction_val, save_folder=None,
                           train_name="train.json", val_name="val.json",
                           stochastic=False):
    """
        Args:
            json_file: full path to json file for all annotations
            fraction_val: fraction of total dataset should be used for
                testing (.25 -> a quarter of total used for testing)
            train_name: file name for the resulting training set
            val_name: file name for the resulting validation set
            stochastic: if false split is deterministic
    """
    
    with open(json_file, "r") as read_file:
        json_dict = json.load(read_file)
        
    print('There are {} annotated images.'.format(
        len(json_dict['images'])))
    
    # image ids to use
    image_ids = np.arange(len(json_dict['images']))
    
    train_dict = create_empty_annotation_json(json_dict)
    val_dict = create_empty_annotation_json(json_dict)

    images = sorted([an for an in json_dict['images']], key=lambda an: an['id']) 
    images = [images[image_id] for image_id in image_ids]

    images_added = 0

    for image_num, image_dict in enumerate(images):
        image_id = image_dict['id']
        new_annotations = get_annotations_for_id(json_dict['annotations'], image_id)
        if len(new_annotations) != 0:
            if stochastic:
                in_val = stochastic_in_val(fraction_val)
            else:
                in_val = deterministic_in_val(images_added, fraction_val)
            if in_val:
                # validation image
                val_dict['images'].append(image_dict)
                val_dict['annotations'].extend(new_annotations)
            else:
                # training image
                train_dict['images'].append(image_dict)
                train_dict['annotations'].extend(new_annotations)
            images_added += 1

    # correct annotation ids
    for new_id, _ in enumerate(train_dict['annotations']):
        train_dict['annotations'][new_id]['id'] = new_id + 1
    for new_id, _ in enumerate(val_dict['annotations']):
        val_dict['annotations'][new_id]['id'] = new_id + 1

    print('{} training images with {} annotations.'.format(
        len(train_dict['images']),len(train_dict['annotations'])))
    print('{} validation images with {} annotations.'.format(
        len(val_dict['images']),len(val_dict['annotations'])))

    save_folder = os.path.dirname(json_file)

    with open(os.path.join(save_folder, train_name), "w") as write_file:
        json.dump(train_dict, write_file, indent=4, separators=(',', ': '))

    with open(os.path.join(save_folder, val_name), "w") as write_file:
        json.dump(val_dict, write_file, indent=4, separators=(',', ': '))
        
def get_annotations_based_on_id(annotation_dicts, image_id, 
                                new_id, annotation_id):
    annotations = []
    for annotation_dict in annotation_dicts:
        if annotation_dict['image_id'] == image_id:
            annotations.append(copy.deepcopy(annotation_dict))
            annotations[-1]['image_id'] = new_id
            annotations[-1]['id'] = annotation_id 
            annotation_id += 1
    return(annotations, annotation_id)

def combine_jsons(json_files, out_file=None, keep_ids=False):
    """ Combine multiple JSON file into a new single consistent JSON file.
    
    Note: currently just uses the category from the first json file.
    
    Args:
        json_files (list): list of json file strings
        out_file (string): full path of file where we want to save new file
            if None, don't save
        keep_ids: if True use each images existing id, if False, give each image
            a new if based on image number in new combined set of images.
            - This is helpful if image ids are not unique across all jsons being
            combined.
    
    Return combined json
    """
    
    json_dicts = []

    for json_file in json_files:
        with open(json_file, "r") as read_file:
            json_dict = json.load(read_file)
            json_dicts.append(json_dict)
    
    total_images = 0
    total_annotations = 0
    for json_dict in json_dicts:
        total_images += len(json_dict['images'])
        total_annotations += len(json_dict['annotations'])
        
    print(f"There are {total_images} annotated images with {total_annotations} annotations in the JSON files.")
    
    new_dict = create_empty_annotation_json(json_dicts[0])

    images_added = 0
    annotation_id = 0

    for json_dict in json_dicts:

        images = [an for an in json_dict['images']]
        images = sorted(images, key=lambda an: an['id']) 

        for image_num, image_dict in enumerate(images):
            image_id = image_dict['id']
            new_image_id = image_id # image id that will be used in the combined json
            if not keep_ids:
                new_image_id = images_added + 1
            new_annotations, annotation_id = get_annotations_based_on_id(
                json_dict['annotations'], image_id, new_image_id, annotation_id)
            new_dict['images'].append(image_dict)
            new_dict['images'][-1]['id'] = new_image_id
            if len(new_annotations) != 0:
                new_dict['annotations'].extend(new_annotations)
            images_added += 1
                

    print('{} images added to new .json'.format(len(new_dict['images'])))
    print('{} annotations added to new .json'.format(len(new_dict['annotations'])))
    
    if out_file:
        with open(out_file, "w") as write_file:
            json.dump(new_dict, write_file, indent=4, 
                      separators=(',', ': '))
    else:      
        return new_dict

#### FUNCTIONS FOR RENAMING CATEGORIES
    
def create_new_categories(category_mapping):
    """ Creates the 'categories' value for a coco annotation dictionary.
    
    Args: 
        category_mapping: dict where keys are existing category names and 
            values are new category names
    """
    
    new_categories = []
    added_categories = []
    for item_num, (old_cat, new_cat) in enumerate(category_mapping.items()):
        if new_cat not in added_categories:
            category = {'id': len(new_categories) + 1, # id starts at 1
                        'name': new_cat,
                        'supercategory': 'all',
                        'isthing': 1
                       }
            new_categories.append(category)
            added_categories.append(new_cat)
    return new_categories

def get_category_id(category_name, categories):
    """ Returns category id based on category name in coco categories.
    
    Args:
        category_name: name of the category 
        categories: value assosiated with 'categories' key in coco dict
        
    """
    for category in categories:
        if category['name'] == category_name:
            return category['id']
    print(f"{category_name} not found in categories.")
    return None

def map_old_category_ids_to_new(old_cats, new_cats, cat_mapping):
    """ Return a dict where keys are previous category ids and values are new.
    
    Args:
        old_cats: value assosiated with 'categories' key in original coco dict
        new_cats: value assosiated with 'categories' key in new coco dict
        cat_mapping: dict where keys are original category names and 
            values are new category names
   
   Returns dict where keys are previous category ids and values are new category ids
   """
    cat_id_map = {}
    
    for old_cat_name, new_cat_name in cat_mapping.items():
        old_cat_id = get_category_id(old_cat_name, old_cats) 
        new_cat_id = get_category_id(new_cat_name, new_cats) 
        cat_id_map[old_cat_id] = new_cat_id
    return cat_id_map
    
def rename_categories(coco_dict, category_mapping, out_file=None):
    """ Rename category names including combing multiple categories into one.
    
    Args:
        coco_dict: existing coco formated dictionary
        category_mapping: dictionary where keys are old category names and
            values are the new category names
        outfile: Path where new coco dict should be saved (should be a valid
            .json filename)
            
    Saves or returns new coco formated dictionary with the new category names 
        (and updated annotations ids) based on on if outfile is given.
    """
    
    old_categories = coco_dict['categories']

    new_coco = create_empty_annotation_json(coco_dict)
    new_categories = create_new_categories(category_mapping)
    new_coco['categories'] = new_categories

    new_coco['images'] = coco_dict['images']

    old_ids_to_new_ids = map_old_category_ids_to_new(old_categories, 
                                                     new_categories, 
                                                     category_mapping)

    for ann in coco_dict['annotations']:
        ann['category_id'] = old_ids_to_new_ids[ann['category_id']]
        new_coco['annotations'].append(ann)
    
    if out_file:
        with open(out_file, "w") as write_file:
            json.dump(new_coco, write_file, indent=4, 
                      separators=(',', ': '))
    else:      
        return new_coco

def create_json_from_image_names(json_file, image_ids, new_name, save_folder=None):
    """ Create a new coco json file from a subset of existing coco json based on 
    list of image names passed to function.

    Convient way to subset annotations by image dimension, number of annotations
    per image etc. by collecting relevant image names outside of function.
    
        Args:
            json_file: full path to json file for all annotations
            image_ids: list of the image ids to include in the new json
                (assumes these ids are in the json file passed to function)
            new_name: name of the new coco json file
            save_folder: path to folder to save new .json files.
                If None, then save in same file as current json
    """
    
    with open(json_file, "r") as read_file:
        json_dict = json.load(read_file)
    
    new_dict = create_empty_annotation_json(json_dict)

    # The index of an image in json_dict['images'] isn't same as that image's id
    id_to_ind = {}
    for ind, image in enumerate(json_dict['images']):
        id_to_ind[image['id']] = ind

    for image_num, image_id in enumerate(image_ids):
        if image_id not in id_to_ind:
            print(f"warning specified image Id {image_id} is not in passed json")
            print("Skipping...")
            continue
        new_annotations = get_annotations_for_id(json_dict['annotations'], image_id)
        if len(new_annotations) != 0:
            image_dict = json_dict['images'][id_to_ind[image_id]]
            new_dict['images'].append(image_dict)
            new_dict['annotations'].extend(new_annotations)

    # correct annotation ids (count from 1 on up)
    for new_id, _ in enumerate(new_dict['annotations']):
        new_dict['annotations'][new_id]['id'] = new_id + 1


    print(f"New coco json has {len(new_dict['images'])}  images ",
          f"with {len(new_dict['annotations'])} annotations.")

    if save_folder is None:
        save_folder = os.path.dirname(json_file)

    with open(os.path.join(save_folder, new_name), "w") as write_file:
        json.dump(new_dict, write_file, indent=4, separators=(',', ': '))

def alphabetize_categories(coco_json):
    """ Make annotation categories alphabetical.
    
    Labelbox doesn't have deterministic category numbering based on given ontology
    so forcing it to be alphabetical makes category order consistent across projects.

    return json although modifies in place
    """
    categories = coco_json['categories']
    names = sorted([c['name'] for c in categories])
    new_pairing = {}
    for ind, name in enumerate(names):
        new_pairing[name] = ind + 1
    # The index of array is the current class id
    # The value in array will be the new class id
    # (0 index isn't used)
    new_cat = {}
    for cat_ind, category in enumerate(categories):
        new_cat[int(category['id'])] = new_pairing[category['name']]
        category['id'] = new_pairing[category['name']]
    
    categories.sort(key=lambda c: c['id'])
        
    for ann in coco_json['annotations']:
        ann['category_id'] = int(new_cat[int(ann['category_id'])])
    return coco_json

def remove_annotation_category(coco, category_name, remove_image=False):
    """ Remove all annotations with category name from coco annotations.
    
    Can either remove just the annotation or the whole image and corresponding 
    annotations that contain that category.
    
    Args:
        coco: dictionary in coco format
        category_name: should be one of coco['categories'][ind]['name']
            where ind is some valid index
        remove_image: if True remove all images and corresponding annotations
            where category is present. Otherwise just remove the annotations of category.
    
    Return modified coco dict.
    """
    original_categories = coco['categories']
    new_categories = []
    remove_id = None
    for cat in original_categories:
        if category_name == cat['name']:
            remove_id = cat['id']
            continue
        new_categories.append(cat)
    if remove_id is None:
        print(f"Warning category to be removed in not in coco['categories']")
        print("Aborting")
        return coco

    print(f"Starting with {len(coco['annotations'])}")

    # Look through all annotations (by image)
    # This probably gets slow for lots of annotations, but easy while tractable
    retained_images = []
    retained_anns = []
    for image in coco['images']:
        image_id = image['id']
        category_present = False
        raw_im_anns = get_annotations_for_id(coco['annotations'], image_id)
        im_anns = []
        for ann in raw_im_anns:
            if ann['category_id'] == remove_id:
                category_present = True
                continue
            im_anns.append(ann)
        if remove_image:
            if category_present:
                continue
        if im_anns:
            # Make sure image still has annotations
            retained_images.append(image)
            retained_anns.extend(im_anns)

    coco['categories'] = new_categories
    coco['images'] = retained_images
    coco['annotations'] = retained_anns

    # renumber annotations:
    for ann_ind, ann in enumerate(coco['annotations']):
        ann['id'] = ann_ind + 1
        

    print(f"Final coco with {len(coco['annotations'])} annotations.")

    return coco  
                        
