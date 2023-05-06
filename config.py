# QuPath infos
QuPath_annotations_path = "/" # Path to QuPath annotations folder

# Dataset infos
# The list of WSI to use for each class
WT = ["20BM02832_1","20BM03021_2","20BM03096_2","20BM03113_1","20BM03733_1","A17005311","A18003401","A18004240","A18004574","A18005087","A18005507","A19004917","A19005113"]
G12C = ["18BM03667_2","18BM03673_2","18BM03749","19BM00116_b","19BM00370_1","19BM00503_1","19BM01220_2","19BM01626_2","19BM01734","19BM01964_2","19BM02461_1","19BM03278_2","21BM02226_2"]
others = ["20BM02181_2","20BM02184_2","20BM02270","20BM02396_1","20BM02594_1","20BM02700_2","20BM02984_2","20BM03163_1","20BM03476_2","20BM03564","20BM03574_1","A17004822","A18000005"]
# The list of WSI to use for each fold
Folds = [
["A18004240","A18005087","18BM03673_2","19BM01626_2","20BM02594_1","20BM03163_1","20BM02396_1"],
["A18004574","20BM03113_1","20BM03021_2","19BM00370_1","19BM01734","A17004822","20BM03476_2","20BM03564"],
["A18005507","A18003401","20BM03733_1","19BM00503_1","19BM03278_2","18BM03749","20BM03574_1","20BM02984_2"],
["A17005311","A19005113","A19004917","19BM02461_1","19BM01220_2","19BM00116_b","20BM02181_2","20BM02184_2"],
["20BM02832_1","20BM03096_2","18BM03667_2","19BM01964_2","21BM02226_2","A18000005","20BM02270","20BM02700_2"]]

dataset_path = "../dataset/"
magnification_level_WSI = 20 # The magnification level of the WSI in the dataset
magnification_level_patches = 20 # The magnification level desired for the patches
patch_height = 224
patch_width = 224
threshold_tumor_area = 1 # Is the percentage of the area of a patch that must belong to the tumor class
stain_estimation_method = "macenko"

# Network params
num_epochs = 30
learning_rate = 0.01
min_learning_rate = 0.001
batch_size = 32
loss = "categorical_crossentropy"
metrics = ["accuracy"]
dropout_rate = 0.1
earlystopping_patience = 10
reducelr_patience = 5
reducelr_factor = 0.1
l2_regularization = None

# Transfer Learning
transfer_learning = False
weights = None
nbr_layers_to_retrain = 1

# Data Augmentation
rescale = 1./255
horizontal_flip = True
vertical_flip = True
rotation_range = 45
shear_range = 0.2
zoom_range = 0.4

# Explainability
explainers = ["LIME","CAM"]
mutation_to_analyze = ["G12C","others","WT"]
wsi_to_explain = WT+G12C+others
number_of_patches_per_wsi_to_explain = 15



# Random seed
seed = 31415