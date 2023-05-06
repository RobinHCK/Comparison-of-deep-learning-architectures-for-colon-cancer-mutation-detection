from config import *
import os
import sys
import numpy as np
import tensorflow
#gpus = tensorflow.config.experimental.list_physical_devices('GPU')
#tensorflow.config.experimental.set_memory_growth(gpus[0], True)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt





def get_and_filter_arguments():
    """Filter and return script arguments
   
    Args:
        None.
    Returns:
        class_to_show (array): The mutation id and the mutation name to study with the explainer (id_mutation, mutation) ex: 0, G12C; 1, others; 2, WT
        wsi (str): The WSI to explain
        number_of_patches_per_wsi_to_explain (int): The number of patches to explain per wsi
       
    Raises:
        Exception: Number of arguments should be 3
        Exception: Mutation given in argument is not recognised
    """
    if len(sys.argv) != 4: # 3 args expected + 1 (filename)
        raise Exception('Number of arguments should be 3')

    mutation = sys.argv[1]
    wsi = sys.argv[2]
    number_of_patches_per_wsi_to_explain = int(sys.argv[3])
    list_mutation = ["G12C","others","WT"]

    if mutation not in list_mutation:
        raise Exception('Mutation given in argument is not recognised')

    id_mutation = list_mutation.index(mutation)
    class_to_show = (id_mutation, mutation) # 0, G12C; 1, others; 2, WT

    return class_to_show, wsi, number_of_patches_per_wsi_to_explain





def create_results_directory(WSI_to_explain, class_to_show, number_of_patches_per_wsi_to_explain):
    """Create the directory to store the results.
   
    Create the WSI_to_explain/ folder.
    Then create a subfolder for each kind of explanation.
    Args:
        WSI_to_explain (str): The WSI to explain.
        class_to_show (str): The class to generate the LIME heatmap
        number_of_patches_per_wsi_to_explain (int): The number of patches to explain per wsi
    Returns:
        None.
       
    Raises:
        OSError: Error while creating the folder.
    """
    try:  
        if not os.path.exists("results/" + WSI_to_explain + "/"):
            os.mkdir("results/" + WSI_to_explain + "/")

        for type_explanation in ["LIME_top_class_only_#"+str(number_of_patches_per_wsi_to_explain)+"_"+class_to_show, "LIME_top_class_#"+str(number_of_patches_per_wsi_to_explain)+"_"+class_to_show, "LIME_pros_and_cons_#"+str(number_of_patches_per_wsi_to_explain)+"_"+class_to_show, "LIME_heatmap_#"+str(number_of_patches_per_wsi_to_explain)+"_"+class_to_show]:
            if not os.path.exists("results/" + WSI_to_explain + "/" + type_explanation + "/"):
                os.mkdir("results/" + WSI_to_explain + "/" + type_explanation + "/")
       
    except OSError:  
        print ('Creation of the directory failed')





def explain_patches_from_a_WSI(WSI_to_explain, class_to_show, number_of_patches_per_wsi_to_explain):
    """ Apply Lime explainer for each patch of a WSI
   
    Create & save explanation for each patch.
    The following methods are applied:
        - top_class_only
        - top_class
        - pros_and_cons
        - heatmap
    Source: https://github.com/marcotcr/lime
    Args:
        WSI_to_explain (str): The WSI to explain.
        class_to_show (array): The class id and name used to generate the LIME heatmap (patch_height,patch_width)
        number_of_patches_per_wsi_to_explain (int): The number of patches to explain per wsi
    Returns:
        None.
       
    Raises:
        OSError: Error while creating the folder.
    """

    # Find the fold where to WSI is in the Test split
    fold_id = [fold_id+1 for fold_id in range(5) if WSI_to_explain in Folds[fold_id]][0]
    base_path = "../dataset/patches_organized_per_split/fold_" + str(fold_id) + "/Test/WT_G12C_others/"

    # Get the patch paths from the WSI
    patch_paths = [path for path in os.listdir(base_path) if WSI_to_explain in path]

    # Load model
    model = load_model(("results/model_K" + str(fold_id) + ".h5"))

    # Initialize Lime explainer
    explainer = lime_image.LimeImageExplainer()

    # Explain a limited number of patches
    count = number_of_patches_per_wsi_to_explain - 1
    
    # Apply Lime to each patch   
    for patch_path in patch_paths:
        # Load a patch 
        img = image.load_img(base_path + patch_path)
        x = image.img_to_array(img)
        x *= rescale
        x = np.expand_dims(x, axis=0)

        # Create visualization
        explanation = explainer.explain_instance(np.squeeze(x, axis=0).astype('double'), model.predict, top_labels=5, hide_color=0, num_samples=1000)
        # top_class_only
        temp, mask = explanation.get_image_and_mask(class_to_show[0], positive_only=True, num_features=5, hide_rest=True)
        plt.imsave("results/" + WSI_to_explain + "/LIME_top_class_only_#"+str(number_of_patches_per_wsi_to_explain)+"_"+class_to_show[1]+"/" + patch_path, mark_boundaries(temp, mask))
        # top_class
        temp, mask = explanation.get_image_and_mask(class_to_show[0], positive_only=True, num_features=5, hide_rest=False)
        plt.imsave("results/" + WSI_to_explain + "/LIME_top_class_#"+str(number_of_patches_per_wsi_to_explain)+"_"+class_to_show[1]+"/" + patch_path, mark_boundaries(temp, mask))
        # pros_and_cons
        temp, mask = explanation.get_image_and_mask(class_to_show[0], positive_only=False, num_features=10, hide_rest=False)
        plt.imsave("results/" + WSI_to_explain + "/LIME_pros_and_cons_#"+str(number_of_patches_per_wsi_to_explain)+"_"+class_to_show[1]+"/" + patch_path, mark_boundaries(temp, mask))
        # heatmap
        # class_to_show[0] is the class predicted by the network ex: 0
        # explanation.top_labels is the ranking array of the predictions for each class ex: [0,2,1]
        dict_heatmap = dict(explanation.local_exp[class_to_show[0]])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
        plt.imsave("results/" + WSI_to_explain + "/LIME_heatmap_#"+str(number_of_patches_per_wsi_to_explain)+"_"+class_to_show[1]+"/" + patch_path, heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())

        if count <= 0:
            break
        else:
            count -= 1





if __name__ == "__main__":

    # Get script arguments
    class_to_show, WSI_to_explain, number_of_patches_per_wsi_to_explain = get_and_filter_arguments()

    # Create hierarchy
    create_results_directory(WSI_to_explain, class_to_show[1], number_of_patches_per_wsi_to_explain)
       
    # Perform Lime on patches from a WSI to explain the results of the model
    explain_patches_from_a_WSI(WSI_to_explain, class_to_show, number_of_patches_per_wsi_to_explain)
