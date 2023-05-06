from config import *
import os
import sys
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib.cm as cm





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
    print(sys.argv)
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
        class_to_show (str): The class to use to draw the heatmap.
        number_of_patches_per_wsi_to_explain (int): The number of patches to explain per wsi
    Returns:
        None.
       
    Raises:
        OSError: Error while creating the folder.
    """
    try:  
        if not os.path.exists("results/" + WSI_to_explain + "/"):
            os.mkdir("results/" + WSI_to_explain + "/")

        for type_explanation in ["CAM_heatmap_#"+str(number_of_patches_per_wsi_to_explain)+"_"+class_to_show,"CAM_heatmap_on_patch_#"+str(number_of_patches_per_wsi_to_explain)+"_"+class_to_show]:
            if not os.path.exists("results/" + WSI_to_explain + "/" + type_explanation + "/"):
                os.mkdir("results/" + WSI_to_explain + "/" + type_explanation + "/")
       
    except OSError:  
        print ('Creation of the directory failed')





def get_img_array(img_path, size):
    """ Load and transform an image to an array
    Source: fchollet
      - https://keras.io/examples/vision/grad_cam/
      - https://github.com/keras-team/keras-io
    Args:
        img_path (str): The image path
        size (array): The image size (patch_height,patch_width)
    Returns:
        array (array): The image array
    """
    # `img` is a PIL image 
    img = tensorflow.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array
    array = tensorflow.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    array = np.expand_dims(array, axis=0)

    return array





def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """ Create CAM heatmap from prediction
    Source: fchollet
      - https://keras.io/examples/vision/grad_cam/
      - https://github.com/keras-team/keras-io
    Args:
        img_array (array): The image array
        model (model): The model to use
        last_conv_layer_name (str): The name of the last conv. layer in the network
        pred_index (int): The class index to use to generate the heatmap
    Returns:
        _ (array): The heatmap
    """
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tensorflow.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tensorflow.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tensorflow.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tensorflow.newaxis]
    heatmap = tensorflow.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tensorflow.maximum(heatmap, 0) / tensorflow.math.reduce_max(heatmap)

    return heatmap.numpy()





def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """ Create and merge CAM heatmap on the original patch, then the image result is saved
    Source: fchollet
      - https://keras.io/examples/vision/grad_cam/
      - https://github.com/keras-team/keras-io
    Args:
        img_path (str): The image path
        heatmap (array): The heatmap
        cam_path (str): The path used to save the image result
        alpha (float): The intensity of the heatmap
    Returns:
        None
    """
    # Load the original image
    img = tensorflow.keras.preprocessing.image.load_img(img_path)
    img = tensorflow.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    heatmap[heatmap < 25] = 0
    heatmap[heatmap >= 25] = 255

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tensorflow.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tensorflow.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tensorflow.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)





def explain_patches_from_a_WSI(WSI_to_explain, class_to_show, number_of_patches_per_wsi_to_explain):
    """ Apply CAM for each patch of a WSI
   
    Create & save explanation for each patch.
    Source: https://github.com/zhoubolei/CAM
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

    # Explain a limited number of patches
    count = number_of_patches_per_wsi_to_explain - 1
    
    # Apply CAM to each patch   
    for patch_path in patch_paths:
        # Load a patch 
        img_array = tensorflow.keras.applications.mobilenet_v2.preprocess_input(get_img_array(base_path+patch_path, size=(patch_height,patch_width)))

        # Generate class activation heatmap
        heatmap = make_gradcam_heatmap(img_array, model, "Conv_1_bn", pred_index=class_to_show[0])
        plt.imsave("results/" + WSI_to_explain + "/CAM_heatmap_#"+str(number_of_patches_per_wsi_to_explain)+"_"+class_to_show[1]+"/" + patch_path, heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())

        save_and_display_gradcam(base_path+patch_path, heatmap, cam_path="results/" + WSI_to_explain + "/CAM_heatmap_on_patch_#"+str(number_of_patches_per_wsi_to_explain)+"_"+class_to_show[1]+"/" + patch_path)

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
