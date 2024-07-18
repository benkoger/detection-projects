# Code from https://community.labelbox.com/t/how-to-convert-export-v2-json-to-coco-format/1053

from labelbox import Client
import urllib.request
import argparse
import copy
import json
import datetime
import requests
from PIL import Image 
import numpy as np
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from shapely.geometry import Polygon
import cv2

def index_ontology(ontology_normalized, export_type="index"):
  """ Given an ontology, returns a dictionary where {key=featureSchemaid : values = {"name", "color", "type", "kind", "parent_featureSchemaIds", "encoded_value"} for each feature in the ontology
  Args:
    ontology_normalized   :   Queried from a project using project.ontology().normalized
  Returns:
    Dictionary with key information on each node in an ontology, where {key=featureSchemaid : values = {"name", "color", "type", "kind", "parent_featureSchemaIds", "encoded_value"}
  """
  feature_map = {}
  tools = ontology_normalized["tools"]
  classifications = ontology_normalized["classifications"]
  if tools:
        results = layer_iterator(feature_map=feature_map, node_layer=tools)
        feature_map = results[0]
  if classifications:
        feature_map = layer_iterator(feature_map=feature_map, node_layer=classifications, encoded_value=results[2], parent_featureSchemaIds=[], parent_featureSchemaId = False)[0]
  return feature_map

def layer_iterator(feature_map, node_layer, encoded_value=0, parent_featureSchemaIds=[], parent_featureSchemaId=False):
    """ Receives a normalized ontology layer (list of dictionaries) and for each dictionary (node), pulls key information where they key=featureSchemaid
        Then if a given node has another layer, recursively call this function to loop through all the nested layers of the ontoology node dictionary
    Args:
        feature_map (dict)              :   Building dictinoary where key=featureSchemaid
        node_layer (list)               :   List of ontology node dictionaries to loop through
        encoded_value (int)             :   Each dictionary gets an encoded value, and this increases by one each ontology node dictionary read into the feature_map
        parent_featureSchemaIds (list)  :   For a given ontology node dictionary, a list of parent featureSchemaid strings
        parent_featureSchemaId (str)    :   The immediate parent ontology node dictionary featureSchemaid
    Returns:
        The same input arguments, only with updated values for feature_map and encoded_value
    """
    if parent_featureSchemaId:
        parent_featureSchemaIds.append(parent_featureSchemaId)
    parent_featureSchemaId = ""
    for node in node_layer:
        encoded_value += 1
        color = ""
        if "tool" in node.keys():
            node_type = node["tool"]
            node_kind = "tool"
            node_name = node["name"]
            next_layer = node["classifications"]
            color = node['color']
        elif "instructions" in node.keys():
            node_name = node["instructions"]
            node_kind = "classification"
            node_type = node["type"]
            next_layer = node["options"]
        else:
            node_name = node["label"]
            node_kind = "option"
            if "options" in node.keys():
                next_layer = node["options"]
                node_type = "branch_option"
            else:
                next_layer = []
                node_type = "leaf_option"
        node_dict = { node['featureSchemaId'] : {"name" : node_name, "color" : color, "type" : node_type, "kind" : node_kind, "parent_featureSchemaIds" : parent_featureSchemaIds, "encoded_value" : encoded_value}}
        feature_map.update(node_dict)
        if next_layer:
            feature_map, next_layer, encoded_value, parent_featureSchemaIds, parent_featureSchemaId = layer_iterator(
                feature_map=feature_map, 
                node_layer=next_layer, 
                encoded_value=encoded_value, 
                parent_featureSchemaIds=parent_featureSchemaIds, 
                parent_featureSchemaId=node['featureSchemaId']
                )
        parent_featureSchemaIds = parent_featureSchemaIds[:-1]
    return feature_map, next_layer, encoded_value, parent_featureSchemaIds, parent_featureSchemaId

def coco_bbox_converter(data_row_id, annotation, category_id): # was data_row_idx
    """ Given a label dictionary and a bounding box annotation from said label, will return the coco-converted bounding box annotation dictionary
    Args:
        data_row_id (str)               :     Labelbox Data Row ID for this label
        annotation (dict)               :     Annotation dictionary from label['Label']['objects'], which comes from project.export_labels()
        category_id (str)               :     Desired category_id for the coco_annotation
    Returns:
        An annotation dictionary in the COCO format
    """
    # Changed by koger: save bbox coords as integers instead of strings
    # Also add area
    bbox = annotation['bounding_box']
    area = float(bbox['width']) * float(bbox['height'])
    coco_annotation = {
        "image_id": data_row_id,
        "bbox": [
            int(bbox['left']),
            int(bbox['top']),
            int(bbox['width']),
            int(bbox['height'])
        ],
        "category_id": str(category_id),
        "id": annotation['feature_id'],
        "area": area,
        "segmentation": [
            int(bbox['left']),
            int(bbox['top']),
            int(bbox['left']) + int(bbox['width']),
            int(bbox['top']),
            int(bbox['left']) + int(bbox['width']),
            int(bbox['top']) + int(bbox['height']), 
            int(bbox['left']),
            int(bbox['top']) + int(bbox['height'])
        ],
        "iscrowd": 0
            
    }
    return coco_annotation

def coco_line_converter(data_row_id, annotation, category_id):
    """ Given a label dictionary and a line annotation from said label, will return the coco-converted line annotation dictionary
    Args:
        data_row_id (str)               :     Labelbox Data Row ID for this label
        annotation (dict)               :     Annotation dictionary from label['Label']['objects'], which comes from project.export_labels()
        category_id (str)               :     Desired category_id for the coco_annotation
    Returns:
        An annotation dictionary in the COCO format
    """
    line = annotation['line']
    coco_line = []
    num_line_keypoints = 0
    for coordinates in line:
        coco_line.append(str(coordinates['x']))
        coco_line.append(str(coordinates['y']))
        coco_line.append("2")
        num_line_keypoints += 1
    coco_annotation = {
        "image_id": str(data_row_id),
        "keypoints": coco_line,
        "num_keypoints": str(num_line_keypoints),
        "category_id" : str(category_id),
        "id": str(annotation['feature_id'])
    }
    return coco_annotation, num_line_keypoints

def coco_point_converter(data_row_id, annotation, category_id):
    """ Given a label dictionary and a point annotation from said label, will return the coco-converted point annotation dictionary
    Args:
        data_row_id (str)               :     Labelbox Data Row ID for this label
        annotation (dict)               :     Annotation dictionary from label['Label']['objects'], which comes from project.export_labels()
        category_id (str)               :     Desired category_id for the coco_annotation
    Returns:
        An annotation dictionary in the COCO format
    """
    coco_annotation = {
        "image_id": str(data_row_id),
        "keypoints": [str(annotation['point']['x']), str(annotation['point']['y']), "2"],
        "num_keypoints": str(1),
        "category_id" : str(category_id),
        "id": str(annotation['feature_id'])
    }
    return coco_annotation

def coco_polygon_converter(data_row_id, annotation, category_id):
    """Given a label dictionary and a point annotation from said label, will return the coco-converted polygon annotation dictionary
    Args:
        data_row_id (str)               :     Labelbox Data Row ID for this label
        annotation (dict)               :     Annotation dictionary from label['Label']['objects'], which comes from project.export_labels()
        category_id (str)               :     Desired category_id for the coco_annotation
    Returns:
        An annotation dictionary in the COCO format
    """
    all_points = []
    points_as_coords = []
    for coord in annotation['polygon']:
        points_as_coords.append([coord['x'], coord['y']])
        all_points.append(str(coord['x']))
        all_points.append(str(coord['y']))
    polygon = Polygon(points_as_coords)
    coco_annotation = {
        "image_id" : data_row_id, "segmentation" : all_points,
        "bbox" : [
            str(polygon.bounds[0]), str(polygon.bounds[1]),
            str(polygon.bounds[2]-polygon.bounds[0]),
            str(polygon.bounds[3]-polygon.bounds[1])
        ],
        "area" : str(polygon.area), "id": str(annotation['feature_id']),
        "iscrowd" : "0", "category_id" : str(category_id)
    }
    return coco_annotation

def download_mask(url, headers = None):
    """ Downloads a mask URL
    Args:
        url (dict)       :     URL of a mask
    Returns:
        A 2-D numPy array of said mask
    """
    downloaded = True
    while downloaded:
        # to ensure api limit doesn't throw an error
        requests_per_min = 1500
        interval = 60/ requests_per_min
        time.sleep(interval)
        try:
            payload = requests.get(url, headers = client.headers)
            if payload.status_code == 200:
                pil_image = Image.open(BytesIO(payload.content))

                # Convert the image to grayscale if it's not already
                if pil_image.mode != 'L':
                    pil_image = pil_image.convert('L')

                # Convert the image to a NumPy array
                np_array = np.array(pil_image)
                downloaded = False
        except:
            downloaded = True

    return np_array

def coco_mask_converter(data_row_id, annotation, category_id):
    """Given a label dictionary and a mask annotation from said label, will return the coco-converted segmentation mask annotation dictionary
    Args:
        data_row_id (str)               :     Labelbox Data Row ID for this label
        annotation (dict)               :     Annotation dictionary from label['Label']['objects'], which comes from project.export_labels()
        category_id (str)               :     Desired category_id for the coco_annotation
    Returns:
        An annotation dictionary in the COCO format
    """
    contours, _ = cv2.findContours(download_mask(annotation['mask']['url']), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_points = []
    points_as_coords = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) >= 6:
            for i in range(0, len(contour), 2):
                points_as_coords.append([contour[i], contour[i+1]])
                all_points.append(str(contour[i]))
                all_points.append(str(contour[i+1]))
    polygon = Polygon(points_as_coords)
    coco_annotation = {
        "image_id" : data_row_id, "segmentation" : all_points,
        "bbox" : [
            str(polygon.bounds[0]), str(polygon.bounds[1]),
            str(polygon.bounds[2]-polygon.bounds[0]),
            str(polygon.bounds[3]-polygon.bounds[1])
        ],
        "area" : str(polygon.area), "id": str(annotation['feature_id']),
        "iscrowd" : "0", "category_id" : str(category_id)
    }
    return coco_annotation

def coco_annotation_converter(data_row_id, annotation, ontology_index):
    """ Wrapper to triage and multithread the coco annotation conversion - if nested classes exist, the category_id will be the first radio/checklist classification answer available
    Args:
        data_row_id (str)               :     Labelbox Data Row ID for this label
        annotation (dict)               :     Annotation dictionary from label["projects"][project_id]['labels']['annotations']['objects'], which comes from project.export_labels()
        ontology_index (dict)           :     A dictionary where {key=featureSchemaId : value = {"encoded_value"} which corresponds to category_id
    Returns:
        A dictionary corresponding to the coco annotation syntax - the category ID used will be the top-level tool
    """
    max_line_keypoints = 0
    category_id = ontology_index[annotation['feature_schema_id']]['encoded_value']
    # if "classifications" in annotation.keys():
    #     if annotation['classifications']:
    #         for classification in annotation['classifications']:
    #             if 'answer' in classification.keys():
    #                 if type(classification['answer']) == dict:
    #                     category_id = ontology_index[classification['schemaId']]['encoded_value']
    #                     break
    #             else:
    #                 category_id = ontology_index[classification['answers'][0]['schemaId']]['encoded_value']
    #                 break
    if "bounding_box" in annotation.keys():
        coco_annotation = coco_bbox_converter(data_row_id, annotation, category_id)
    elif "line" in annotation.keys():
        coco_annotation, max_line_keypoints = coco_line_converter(data_row_id, annotation, category_id)
    elif "point" in annotation.keys():
        coco_annotation = coco_point_converter(data_row_id, annotation, category_id)
    elif "polygon" in annotation.keys():
        coco_annotation = coco_polygon_converter(data_row_id, annotation, category_id)
    else:
        coco_annotation = coco_mask_converter(data_row_id, annotation, category_id)
    return coco_annotation, max_line_keypoints

def coco_converter(project, verbose=False, extra_image_info_func=None):
    """ Given a project and a list of labels, will create the COCO export json
    Args:
        project (labelbox.schema.project.Project)   :   Labelbox project object
        verbose: prints info about process if True
        extra_image_info_func: a function that takes a labelbox label and project
            and returns a dictionary containing info that should be saved to image 
            info in output coco json but isn't normally added (could be something 
            like image classification annoations i.e. nightime)
    Returns:
    """

    export_params= {
        "label_details": True,
        "data_row_details": True,
        }

    filters= {
        }

    labels_list = project.export_v2(params=export_params, filters=filters)
    labels_list.wait_till_done()
    labels_list = labels_list.result

    # Info section generated from project information
    info = {
        'description' : project.name,
        'url' : f'https://app.labelbox.com/projects/{project.uid}/overview',
        'version' : "1.0",  'year' : datetime.datetime.now().year,
        'contributor' : project.created_by().email,
        'date_created' : datetime.datetime.now().strftime('%Y/%m/%d'),
    }
    # Licenses section is left empty

    licenses = [ { "url" : "N/A", "id" : 1, "name" : "N/A" } ]

    # Remove any rows without annotations
    # Modified by Koger: check if image has annotations
    cleaned_labels_list = []
    for ann_ind, label in enumerate(labels_list):
        if label["projects"][project.uid]["labels"]:
            cleaned_labels_list.append(label)
    labels_list = cleaned_labels_list

    images = []
    data_row_check = [] # This is a check for projects where one data row has multiple labels (consensus, benchmark)
    # Modified by Koger to add ann_ind so image ids used in coco json are ints and not the labelbox row id
    for ann_ind, label in enumerate(labels_list):
        data_row = label["data_row"]
        if data_row['id'] not in data_row_check:
            data_row_check.append(data_row['id'])
            # Modified by Koger
            filename = f"{data_row['details']['dataset_name']}-{data_row['external_id']}"
            images.append({
                "license" : 1, "file_name" : filename,
                "height" : label["media_attributes"]['height'], 
                "width" : label["media_attributes"]['width'],
                "date_captured" : label["projects"][project.uid]["labels"][0]["label_details"]["created_at"], # data_row.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                "id" : ann_ind, "coco_url": data_row["row_data"]
            })
        if extra_image_info_func is not None:
            # There may be project specific information to add to the image data (classifications)
            # Assumes added_image_info_func is a function that takes a label and returns a dictionary
            extra_info_dict = extra_image_info_func(label, project)
            images[-1].update(extra_info_dict)
    if verbose:
        print(f'\nData Rows Converted into a COCO Dataset.')  

    annotations = []
    if verbose:
        print(f'\nConverting Annotations into the COCO Format...\n')
    ontology_index = index_ontology(project.ontology().normalized) 
    global_max_keypoints = 0
    futures = []
    with ThreadPoolExecutor() as exc:
        # Modified by Koger to add ann_ind so image ids used in coco json are ints and not the labelbox row id
        for ann_ind, label in enumerate(labels_list):
            idx = 0
            for annotation in label["projects"][project.uid]['labels'][idx]['annotations']['objects']:
                futures.append(exc.submit(coco_annotation_converter, ann_ind, annotation, ontology_index))
            idx += 1
        for f in tqdm(as_completed(futures)):
            res = f.result()
            if int(res[1]) > global_max_keypoints:
                global_max_keypoints = int(copy.deepcopy(res[1]))
            annotations.append(res[0])
    if verbose:
        print(f'\nAnnotation Conversion Complete. Converted {len(annotations)} annotations into the COCO Format.') 

    categories = []

    if verbose:
        print(f'\nConverting the Ontology into the COCO Dataset Format...') 
    for featureSchemaId in ontology_index:
        if ontology_index[featureSchemaId]["type"] == "line": 
            keypoints = []
            skeleton = []
            for i in range(0, global_max_keypoints): 
                keypoints.append(str("line_")+str(i+1))
                skeleton.append([str(i), str(i+1)])
            categories.append({
                "supercategory" : ontology_index[featureSchemaId]['name'],
                "id" : str(ontology_index[featureSchemaId]["encoded_value"]),
                "name" : ontology_index[featureSchemaId]['name'],
                "keypoints" : keypoints,
                "skeleton" : skeleton,
            })
        elif ontology_index[featureSchemaId]["type"] == "point": 
            categories.append({
                "supercategory" : ontology_index[featureSchemaId]['name'],
                "id" : str(ontology_index[featureSchemaId]["encoded_value"]),
                "name" : ontology_index[featureSchemaId]['name'],
                "keypoints" : ['point'],
                "skeleton" : ["0", "0"],
            })        
        elif ontology_index[featureSchemaId]['kind'] == 'tool':
            categories.append({
                "supercategory" : ontology_index[featureSchemaId]['name'],
                "id" : str(ontology_index[featureSchemaId]["encoded_value"]),
                "name" : ontology_index[featureSchemaId]['name']
            })     
        elif len(ontology_index[featureSchemaId]['parent_featureSchemaIds']) == 2:
            supercategory = ontology_index[ontology_index[featureSchemaId]['parent_featureSchemaIds'][0]]['name']
            categories.append({
                "supercategory" : supercategory,
                "id" : str(ontology_index[featureSchemaId]["encoded_value"]),
                "name" : ontology_index[featureSchemaId]['name']
            })
    if verbose:
        print(f'\nOntology Conversion Complete')  

    coco_dataset = {
        "info" : info,
        "licenses" : licenses,
        "images" : images,
        "annotations" : annotations,
        "categories" : categories
    }      
    if verbose:
        print(f'\nCOCO Conversion Complete')    
    return coco_dataset