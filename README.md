# CS-433 Machine Learning Project 2: Road Segmentation

## Efe Tarhan, Sabri Yigit Arslan, Eren Akcanal

To reproduce the results presented in both the report and the notebooks in the repository, one can simply follow the instructions listed here onward.

## Folder Structure
The folder structure of the project is as follows:

```bash
project/
    |__ figures/
            |__ pspnet_figures/
            |__ segformer_figures/
            |__ unet_figures/
    |__ model_dicts/
    |__ notebooks/
            |__ pspnet_notebook.ipynb
            |__ segformer_notebook.ipynb
            |__ unet_notebook.ipynb
    |__ src/
    |__ submission/
    |__ test/
            |__ test1.png
            |__ test2.png
            |__ test3.png
            |__ .
            |__ .
            |__ test50.png
    |__ train/
            |__ images/
                    |__ satImage_001.png
                    |__ satImage_002.png
                    |__ satImage_003.png
                    |__ .
                    |__ .
                    |__ satImage_100.png
            |__ groundtruth/
                    |__ satImage_001.png
                    |__ satImage_002.png
                    |__ satImage_003.png
                    |__ .
                    |__ .
                    |__ satImage_100.png
    |__ requirements.txt
    |__ run.py
```

In other words, after cloning the project repository, one needs to create two subfolders with names *train* and *test* inside the project folder. Then, as illustrated above, one needs to download and copy the training images, training groundtruths, and test images of the road segmentation dataset by strictly following the above hierarchy.

## Creating the Virtual Environment

In order to execute the script *run.py*, one needs to install the required dependencies for the project. The simplest way to do this is to create a Python virtual environment and run the project within that virtual environment after installing the required dependencies.

Note that these instructions assume that enough disk space is present on one's computer, and also the necessary graphics card drivers are already installed and functional.

### 1) Create the virtual environment
Execute the following command in shell to create a new Python virtual environment:

```bash
python -m venv road_seg
```
### 2) Activate the virtual environment
If on Windows, execute the following command in shell to activate the virtual environment:
```bash
road_seg\Scripts\activate
```
If on Linux/Mac, execute the following command instead:
```bash
source road_seg/bin/activate
```

### 3) Install PyTorch with CUDA
Our training and testing scripts were tested in PyTorch version 1.13.0 along with CUDA support with version 11.6. In order to use CUDA for efficiently reproducing the training results, execute the following command within the virtual environment:

```bash
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

### 4) Install the remaining dependencies via pip
A *requirements.txt* file can be found inside the project folder, which can be used to install the required dependencies to the virtual environment. Simply execute the following command on the shell after modifying the path to file *requirements.txt*:

```bash
pip install -r path/to/requirements.txt
``` 
Now, a compatible Python virtual environment having CUDA support with all the required dependencies should be all set.


## Description of the Files and Folders

### figures/
As can be understood from its name, this folder contains several plots and figures for three different models that we experimented with for the road segmentation task at hand. The three different models refer to PSPNet, SegFormer, and U-Net, respectively.

### notebooks/
This folder contains the notebooks highlighting the experiments that had been done with three different models. The results obtained in these notebooks were used in the report for the comparisons between different models and varying setups.

### src/
In order to present a simple *run.py* file which would generate the relevant predictions to be submitted to the AICrowd system, we had to break the notebook for our best-performing model, SegFormer, into several easy-to-read scripts containing various utility classes and functions. These scripts are placed in the *src* folder.

### submission/
This folder contains the predictions made by our models during the inference phase on the test set. It is also the default folder in which new *submission.csv* files are stored after running the script *run.py*.

### run.py
This is the script one needs to run with the appropriate constants to generate our best predictions submitted to the online competition platform.

## Reproducing the Training Results and Acquiring the Predictions

In order to generate the predictions of our best-performing model on the test set, namely the SegFormer, one needs to first update several variables inside the script *run.py*.

To start with, update the lines 18-19 of the script *run.py* by typing the appropriate absolute paths for both the train and test data, respectively:

```python
# Load the dataset and preprocess
train_path = "/path/to/project/train/"
test_path = "/path/to/project/test"
```

Then, update the lines 70-71 of the script to indicate the location for dumping the generated predictions:

```python
# Make predictions
output_dir = "/path/to/project/submission/predicted/segformer_predicted"
submission_file = "/path/to/project/submission/submission.csv"
```

Now, after running the script, the predictions of the newly trained model will be recorded under directory *submission*, and the submission file *submission.csv* will also be put here.
