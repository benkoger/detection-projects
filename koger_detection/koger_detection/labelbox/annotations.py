import os
import json
from datetime import datetime

from labelbox import Project
from labelbox.data.serialization import COCOConverter
import numpy as np
from PIL import Image
import requests

from koger_detection.labelbox.coco import coco_converter
from koger_detection.utils.json import alphabetize_categories

def convert_name_to_unicode(name, num_chars_from_end=10):
    """ Convert concatenate unicode values of final characters in string together.
    
    Args:
        name: string
        num_chars_from_end: concatenate all characters in name[-num_chars_from_end:]
        
    return integer
    """
    return int("".join([str(ord(c)) for c in name[-num_chars_from_end:]]))

def get_time_as_int():
    """ Get current time as 18 digit int for (mostly) unique naming"""
    now = datetime.utcnow()
    return int(now.strftime("%Y%m%d%H%M%S%f")[2:])


def download_annotation_project(project, project_name, image_folder, json_file, 
                                save_images=False, verbose=False,
                               extra_image_info_func=None):
    """ Download dataset from labelbox using old labelbox format.
    
    Args:
        project: labelbox project
        project_name: project name (used for image file names)
        image_folder: path where images should be downloaded to
        json_file: full path where coco json should be saved
        save_images: if images should also be saved
        verbose: If True print skipped and unsaved images
        extra_image_info_func: a function that takes a labelbox label and project
            and returns a dictionary containing info that should be saved to image 
            info in output coco json but isn't normally added (could be something 
            like image classification annoations i.e. nightime)
    """
    coco = coco_converter(project, verbose, extra_image_info_func)
    # coco.pop('info')

    
    for image_info in coco['images']:
        path = os.path.join(image_folder, image_info['file_name'])
        if save_images:
            url = image_info['coco_url']
            im = Image.open(requests.get(url, stream=True).raw)
            im.save(path) 

        else:
            if not os.path.exists(path):
                print(f"Warning image {image_info['file_name']} doesn't exist.")
                if verbose:
                    print(f"Note: save images is currently set to {save_images}")

    coco = alphabetize_categories(coco)
    
    with open(json_file, 'w') as file:
        json.dump(coco, file, indent=4)

    
def download_annotation_project_deprecated(project, project_name, image_folder, json_file, 
                                save_images=False, verbose=False):
    """ Download dataset from labelbox using old labelbox format.
    
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
    image_num = 0
    for proj_label_num, label in enumerate(project_labels):
        current_time = get_time_as_int()
        filename = f"{current_time}.jpg"
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
                                 project_names, download_images=True,
                                 verbose=False, skip_existing=False,
                                 extra_image_info_func=None):
    """ Download one or many labelbox project annotations and images.
    
    Args:
        annotation_folder: Where annotation .jsons will be saved.
            Images will be saved in a folder called images within this folder.
        client: Labelbox client object
        project_names: list of Labelbox project names
        download_images: If True, save images
        verbose: Print updates on export process
        skip_existing: If True and the json file that will be used for a given
            project already exists then skip that project
        extra_image_info_func: a function that takes a labelbox label and project
            and returns a dictionary containing info that should be saved to image 
            info in output coco json but isn't normally added (could be something 
            like image classification annoations i.e. nightime)
    """
    
    image_folder = os.path.join(annotation_folder, "images")
    os.makedirs(image_folder, exist_ok=True)

    for name in project_names:
        json_path = os.path.join(annotation_folder, f"{name}.json")
        if skip_existing:
            if os.path.exists(json_path):
                if verbose:
                    print(f"Skipping {name} since .json already exists")
                continue
        print(f"Downloading project {name}")
        
        project = next(client.get_projects(where=(Project.name==name)))
        download_annotation_project(project, name, image_folder, json_path, 
                                    download_images, verbose=verbose,
                                    extra_image_info_func=extra_image_info_func)
        
def create_dataset(labelbox_client, image_files, dataset_name):
    """First create dataset with images.
    
    Args:
        labelbox_client: labelbox Client object
        image_files: list of files to upload to labelbox dataset
        dataset_name: name of the dataset
    """
    
    storage_dataset = labelbox_client.create_dataset(name=dataset_name)

    # Create data payload
    # External ID is recommended to identify your data
    my_data_rows = []
    for image_file in image_files:
        my_data_rows.append({"row_data": image_file,
                             "external_id": os.path.basename(image_file)
                            }
                           )
    # Bulk add data rows to the dataset
    task = storage_dataset.create_data_rows(my_data_rows)
    task.wait_till_done()
    while task.status != "COMPLETE":
        print(f"Storage dataset upload status: {task.status}")
        time.sleep(3)
    return task.status == "COMPLETE"
        

def create_storage_dataset(labelbox_client, image_files, dataset_name):
    """First create dataset with images including overlay images
    so that they are given an online adress by labelbox. Nessisary
    step to eventually have overlay dataset for focal frames.
    
    Args:
        labelbox_client: labelbox Client object
        image_files: list of files to upload to labelbox dataset
        dataset_name: name of the final overlay dataset to be annotated
            ('-storage' will be appended)
    """
    
    # This dataset is just for uploading all images to the cloud
    # In a second dataset some of these images will become overlay images
    # So the user can use animal movement to help detect crypric individuals
    return create_dataset(labelbox_client, image_files, f"{dataset_name}-storage")
        
        
def add_crops_to_labelbox(labelbox_client, focal_images, dataset_name):
    """ Create labelbox dataset with overlay images around focal image.
    
    Args: 
        labelbox_client: labelbox Client object
        focal_images: list of focal image full paths
            (These are the files that are going to actually be 
            annotated (marked by _f))
        dataset_name: name of the dataset 
            
        """
    
    focal_names = [os.path.basename(f) for f in focal_images]
    
    dataset = labelbox_client.create_dataset(name=dataset_name)
    
    storage_ds = labelbox_client.get_datasets(
        where=labelbox.Dataset.name == f"{dataset_name}-storage").get_one()

    my_data_rows = []

    for focal_name in focal_names:
        prev_name = focal_name.split("_f.jpg")[0] + "_a.jpg"
        next_name = focal_name.split("_f.jpg")[0] + "_b.jpg"
        data_row = storage_ds.data_row_for_external_id(focal_name)
        prev_row = storage_ds.data_row_for_external_id(prev_name)
        next_row = storage_ds.data_row_for_external_id(next_name)

        my_data_rows.append({"row_data": data_row.row_data,
                             "external_id": data_row.external_id,
                             "attachments": [
                                 {
                                     "type": "IMAGE_OVERLAY",
                                     "value": prev_row.row_data
                                 },
                                 {
                                     "type": "IMAGE_OVERLAY",
                                     "value": next_row.row_data
                                 }
                             ]
                            }
                           )

    # Bulk add data rows to the dataset
    task = dataset.create_data_rows(my_data_rows)

    task.wait_till_done()
    while task.status != "COMPLETE":
        print(f"Storage dataset upload status: {task.status}")
        time.sleep(3)
    return f"successful upload: {task.status == 'COMPLETE'}"