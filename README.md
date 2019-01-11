# Overview
This project was created for the AUVSI SUAS 2016-17 competition. The objective is to autonomously classify ground targets, as per competition specifications. More information regarding the competition may be found at the following link: http://www.auvsi-suas.org/static/competitions/2017/auvsi_suas-2017-rules.pdf

# Setup
Most of the requirements are included in requirements.txt. The listed packages can be installed by running:
```
pip install -r requirements.txt
```
Python-OpenCV is also used for image read/write, processing, and visualization. At the time of writing, OpenCV needed to be installed separately. However, there now appears to be [unofficial packages](https://pypi.org/project/opencv-python/) available on PyPI, which could greatly ease the OpenCV setup process. The base package may be installed using:
```
pip install opencv-python
```
See the [PyPI page](https://pypi.org/project/opencv-python/) for more details and package options

# Usage
## Quick Start
The main entry point is `classify_images.py`. To get information about its options, run
```
# Script located in root of repository
python classify_images.py -h
```
A common example usage is
```
# Script located in root of repository
python classify_images.py --dir /path/to/directory/with/images --checkpoint_dir /path/to/MODEL_CHECKPOINT.ckpt
```
The script will loop continuously, processing all images of the specified format (e.g. \*.jpg) in the specified directory (including images added while it's running). Terminate it at any time using `Ctrl+C`. Processed images are moved to subdirectories of as they are processed, e.g. `/home/user/images/foo.jpg` would be moved to `/home/user/images/processed_00/foo.jpg`. Note that the script attempts to resolve naming conflicts by incrementing a counter, e.g. if it processes some image `/home/user/images/bar.jpg` and `/home/user/images/processed_00/bar.jpg` already exists, then it will attempt to move the processed image to `/home/user/images/processed_01/bar.jpg`, up to `.../processed_99/bar.jpg`.

## ROS Integration
The 2016-17 codebase used ROS to connect its various components. The image processing stack integrates into the ROS framework via `classify_ros.py`, which additionally depends on `rospy`, which is installed with ROS. ROS installation instructions can be found [here](http://wiki.ros.org/kinetic/Installation/Ubuntu).

Running the code should be fairly straightforward:
```
# Script located in root of repository
python classify_ros.py -c /path/to/SOME_MODEL_CHECKPOINT.ckpt
```
The script starts a ROS node that listens for ROIs and automatically classifies and conditionally transmits the classification results using the code in `classify_images.py`

## Training
The main (WideResNet) model is defined in `wideresnet_model.py`, while training and evaluation are defined in `wideresnet_train.py` and `wideresnet_eval.py` respectively. The training and evaluation scripts may be run directly, and can be inspected for options by running
```
# Scripts located in convnets/wideresnet
python wideresnet_train.py -h
python wideresnet_eval.py -h
```
However, convenience scripts for training/evaluating models for each of the various SUAS target characteristics are available, which can be run without needing to provide any arguments. For instance:
```
# Scripts located in convnets/wideresnet
python suas_shapes_train.py
python suas_shapes_eval.py
```
Note that you can still customize your run with various flags. Some frequent configurations I used looked like:
```
# Scripts located in convnets/wideresnet
python suas_shapes_train.py --batch_size 64 --train_dir ~/aza_cv/shapes/train
python suas_shapes_eval.py --run_once --eval_dir ~/aza_cv/shapes/eval --checkpoint_dir ~/aza_cv/shapes/train
```
Training can be resumed from a checkpoint (e.g. in the event of an interruption) using:
```
# Scripts located in convnets/wideresnet
python suas_alphanum_train.py --checkpoint_path ~/aza_cv/alphanum/train --train_dir ~/aza_cv/alphanum/training_resume
```
**WARNING!** Be *very* careful with your train_dir, as the code might\* wipe out the directory on startup or overwrite files. Either way, if you care about the checkpoint, I'd recommend backing it up (copy-pasting it to another dir is enough) before starting any other train or evaluation operation.  
*\*I recall losing checkpoints because of the code clearing directories at one point, but don't remember which part exactly did this and/or if the code still does that*
### Mean-StdDev Normalization
The model uses mean-standard normalization (mean subtraction followed by standard deviation normalization). This is a somewhat dated technique (and could likely be replaced entirely by batchnorming), but if you'd like to retrain the model as is you will need to manually update the mean-stddev values. `meanstd.py` is a helper script which will calculate the mean and standard deviation of a dataset and print the results to the console:
```
# Script located in root of this repository
python meanstd.py --dir /path/to/image/directory
```
Use the printed mean and standard deviation values to update the mean and stddev definitions in `convnets/wideresnet/suas_*_data.py` in **[B, G, R] format** before training or evaluating on your data.

# Future Work
Much of this repository is out of date as technology continues to evolve. Some recommended changes include:
- Freeze models instead of loading from checkpoint during deployment. See https://www.tensorflow.org/guide/extend/model_files
- Retrain model with more, better data. Last time the classifier was used, the results were very poor due to poor training (lack of good data, large class imbalance, etc)
- Change to a newer, more state-of-the-art model
- Change tabs to spaces. I used to think tabs were better. I was wrong.

# Author
David Hung (davidhung@email.arizona.edu)  
Arizona Autonomous Vehicles Club  
University of Arizona  
