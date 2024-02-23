import os
import json

from labelbox import Project
from labelbox.data.serialization import COCOConverter
import numpy as np
from PIL import Image

def convert_name_to_unicode(name, num_chars_from_end=10):
    """ Convert concatenate unicode values of final characters in string together.
    
    Args:
        name: string
        num_chars_from_end: concatenate all characters in name[-num_chars_from_end:]
        
    return integer
    """
    return int("".join([str(ord(c)) for c in name[-num_chars_from_end:]]))

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
    new_cat = np.arange(len(categories)+1)
    for cat_ind, category in enumerate(categories):
        new_cat[category['id']] = new_pairing[category['name']]
        category['id'] = new_pairing[category['name']]
    
    categories.sort(key=lambda c: c['id'])
        
    for ann in coco_json['annotations']:
        ann['category_id'] = int(new_cat[ann['category_id']])
    return coco_json

    

def download_annotation_project(project, project_name, image_folder, json_file, 
                                save_images=False, verbose=False):
    """ Download dataset from labelbox.
    
    Args:
        project: labelbox project
        project_name: project name (used for image file names)
        image_folder: path where images should be downloaded to
        json_file: full path where coco json should be saved
        save_images: if images should also be saved
        verbose: If True print skipped and unsaved images
    """
    labels = []
    project_labels = project.label_generator()
    # Representation of project name as a number (last 10 characters)
    unicode_name = convert_name_to_unicode(project_name)
    image_num = 0
    for proj_label_num, label in enumerate(project_labels):
        filename = f"{unicode_name}{image_num:05}.jpg"
        path = os.path.join(image_folder, filename)
        if len(label.annotations) == 0:
            if verbose:
                print(f"No annotations in image {label_num}")
            continue
        if save_images:
            im = Image.fromarray(label.data.value)
            im.save(path) 
        else:
            if not os.path.exists(path):
                print(f"Warning image for annotation {label_num} doesn't exist.")
            if verbose:
                print(f"Note: save images is currently set to {save_images}")
            continue
        label.data.file_path = filename
        labels.append(label)
        image_num += 1

    coco = COCOConverter.serialize_instances(
        labels = labels, 
        image_root = image_folder,
        ignore_existing_data=True
    )
    coco.pop('info')

    coco = alphabetize_categories(coco)
    
    with open(json_file, 'w') as file:
        json.dump(coco, file, indent=4)

def download_annotation_projects(annotation_folder, client, 
                                 project_names, download_images=True):
    """ Download one or many labelbox project annotations and images.
    
    Args:
        annotation_folder: Where annotation .jsons will be saved.
            Images will be saved in a folder called images within this folder.
        client: Labelbox client object
        project_names: list of Labelbox project names
        download_images: If True, save images
    """
    
    image_folder = os.path.join(annotation_folder, "images")
    os.makedirs(image_folder, exist_ok=True)

    for name in project_names:
        json_path = os.path.join(annotation_folder, f"{name}.json")
        project = next(client.get_projects(where=(Project.name==name)))
        download_annotation_project(project, name, image_folder, json_path, 
                                    download_images)
        