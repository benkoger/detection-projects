# koger_detection
 
## Annotation
These repository is designed to interface with [Labelbox](labelbox.com) for initial and model-assisted annotation. Annotation projects can easily be initiated through their GUI. 

### Downloading Annotations
Annotations and corresponding images are downloaded as COCO format .json files with the notebook called [download_dataset](https://github.com/benkoger/detection-projects/blob/main/example_notebooks/download_dataset.ipynb). The section called "Doenload projects" downloads the annotations from each project specified in the project_names list into a seperate coco formated .json file in the folder ROOT/annotations. You can specfiy you projects root path by creating a plain text file called .env and including the line ROOT=<Local project folder> there <Local project folder> is the path to that folder. See example_env.txt for an .env template. The .env file should be saved in the detection-projects folder. Images from all specified projects will be downloaded into the same folder at ROOT/annotations/images. 

Part of this process is done using the Labelbox API. To access data from a specific Labelbox account, the API must be provideded with an access token for that account. To generate this key, follow the instructions at [https://docs.labelbox.com/reference/create-api-key](https://docs.labelbox.com/reference/create-api-key). The code here expects your Labelbox key to be saved in a plain text file called .env where LABELBOX_API_KEY=<Your API key>. See example_env.txt for an .env template. 
