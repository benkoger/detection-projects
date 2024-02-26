# koger_detection

Notebooks and code for efficiently annotating, training, and using object detection models.

## Initial set up

### pip install
The koger_detection package must be pip installed locally for the jupyter notebooks in adjacent folder to see it. After cloning this repository locally, use the terminal to go to the folder in detection-projects called koger_detection (the outer of the two koger_detection folders. The one you want contains a file called pyproject.toml and a second folder called koger_detection.)  and run ```pip install -e .``` The period is part of the command to run indicating it should run in the current directory. The -e means that codes koger_detection won't need to be reinstalled if there are changes to the code. As always, it is recomended to do this in a virtual environment.

### .env file
You must create a plaintext .env file that is saved in the detection-projects folder and defines certain required environment variables listed below. See example_env.txt for an example text file that can be renamed (to .env) and filled in. See https://stackoverflow.com/questions/41546883/what-is-the-use-of-python-dotenv for more info on .env in general.

**ROOT:**
ROOT=*Local project folder* where things like annotatations will be stored in automatically generated subfolders. *Local project folder* is the full path to that folder

**LABELBOX_API_KEY:**
Your labelbox API key see [Downloading Annotations](#Downloading-Annotations) for more info on what this is.

**TORCHVISION_REFERENCES:**
You must have a local copy of the contents of the following folder: https://github.com/pytorch/vision/tree/main/references/detection
The environmental variable TORCHVISION_REFERENCES should be the full path to this folder.

**MODEL_PATH:**
Full path to where trained model weights and corresponding config files etc. will be saved (in created subfolders).
 
## Annotation
These repository is designed to interface with [Labelbox](labelbox.com) for initial and model-assisted annotation. Annotation projects can easily be initiated through their GUI. 

### Downloading Annotations
Annotations and corresponding images are downloaded as COCO format .json files with the notebook called [download_dataset](https://github.com/benkoger/detection-projects/blob/main/example_notebooks/download_dataset.ipynb). The section called "Download projects" downloads the annotations from each project specified in the project_names list into a seperate coco formated .json file in the folder ROOT/annotations. ROOT=*Local project folder* where *Local project folder* is the path to that folder. See example_env.txt for an .env template. The .env file should be saved in the detection-projects folder. Images from all specified projects will be downloaded into the same folder at ROOT/annotations/images. 

Part of this process is done using the Labelbox API. To access data from a specific Labelbox account, the API must be provideded with an access token for that account. To generate this key, follow the instructions at [https://docs.labelbox.com/reference/create-api-key](https://docs.labelbox.com/reference/create-api-key). The code here expects your Labelbox key to be saved in a plain text file called .env where LABELBOX_API_KEY=*Your API key*. See example_env.txt for an .env template. 

## Model Training

For model training use the notebook in example_notebooks called [train_model.ipynb](https://github.com/benkoger/detection-projects/blob/main/example_notebooks/train_model.ipynb). 

## Setup
The training notebook relies on an environmental varaible MODEL_PATH that is defined in your .env file and defined where model weights and config files will be saved.
