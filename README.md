<!-- TITLE -->
<br />
<p align="center">
  <h3 align="center">Comparison of deep learning architectures for colon cancer mutation detection</h3>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Presentation of the Project](#presentation-of-the-project)
* [Prerequisite](#prerequisite)
* [Workflow](#workflow)
  * [Create the Dataset](#create-the-dataset)
  * [Train](#train)
  * [Evaluate](#evaluate)
  * [Explainability](#explainability)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)


<!-- PRESENTATION OF THE PROJECT -->
## Presentation Of The Project

This research is carried out as part of the project [AiCOLO](https://wemmertc.github.io/aicolo/) and has been been published in the conference [IEEE 36th International Symposium on Computer Based Medical Systems (CBMS) 2023](https://2023.cbms-conference.org/). The paper is available [here](https://hal.science/hal-04168891/document). This work is a deep learning project applied to medical images. This project introduces a methodology to train deep neural networks to classify genetic mutations directly from histopathological images.

![workflow](https://github.com/RobinHCK/Comparison-of-deep-learning-architectures-for-colon-cancer-mutation-detection/blob/main/img/workflow.png)


<!-- GETTING STARTED -->
## Prerequisite

- Before executing the scripts, make sure you have correctly edited the configuration file: **config.cfg**
- Medical images with annotations of tumor areas and genetic mutations (The medical images used in this project come from the AiCOLO private dataset).
- Tensorflow-Keras / Python


<!-- WORKFLOW -->
## Workflow

### Create The Dataset

- The dataset should be organized as follows:

![dataset](https://github.com/RobinHCK/Comparison-of-deep-learning-architectures-for-colon-cancer-mutation-detection/blob/main/img/dataset_orga.png)

- If you use QuPath to annotate tumor areas in your medical images, you can use these files to cut patches from the annotated areas:
  - python 1_move_patches_from_QuPath.py
  - python 2_create_patches_from_QuPath_patches.py
- Apply the staining normalization:
  - python 3_normalize_patches.py
- Organize patches into 5 folds for cross-validation:
  - python 4_organize_split.py

### Train

- Train one network per fold:
  - python 5_experiment_train.py

### Evaluate

- Create and save network training and evaluation graphs:
  - python 6_create_graphs.py
- Draw the predictions for one WSI:
  - python 7_draw_preds_on_WSI.py

### Explainability

- Run CAM or LIME to understand network predictions
  - python 8_run_explainability.py

![expl](https://github.com/RobinHCK/Comparison-of-deep-learning-architectures-for-colon-cancer-mutation-detection/blob/main/img/expl.png)


<!-- CONTACT -->
## Contact

Robin Heckenauer - robin.heckenauer@gmail.com

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

This work was supported by the AiCOLO project funded by INSERM/Plan Cancer. 
The computational resources were provided by the Mesocentre of the University of Strasbourg
