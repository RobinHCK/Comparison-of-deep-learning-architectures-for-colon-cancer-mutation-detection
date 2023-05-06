from config import *
import numpy as np
import random
random.seed(seed)
import os
import sys
import tensorflow
import lime
from lime import lime_image
from tensorflow.keras.models import load_model
from skimage.io import imread
from skimage.color import gray2rgb
from skimage.segmentation import mark_boundaries
import gc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 20000000000
import imageio





def get_and_filter_arguments():
    """Filter and return script arguments
   
    Args:
        None.
    Returns:
        explainer (str): The explainability method to use (CAM or LIME)
        class_to_show (array): The mutation id and the mutation name to study with the explainer (id_mutation, mutation) ex: 0, G12C; 1, others; 2, WT
        wsi (str): The WSI to explain
        layers_option (str): The layers to explain (last or all)
       
    Raises:
        Exception: Number of arguments should be 3 or 4
        Exception: Explainer should be CAM or LIME
        Exception: Mutation given in argument is not recognised
        Exception: Layers to explain given in argument should be last or all
    """
    if len(sys.argv) != 4 and len(sys.argv) != 5: # 3 or 4 args expected + 1 (filename)
        raise Exception('Number of arguments should be 3 or 4')

    explainer = sys.argv[1]
    mutation = sys.argv[2]
    wsi = sys.argv[3]
    layers_option = None
    list_mutation = ["G12C","others","WT"]

    if explainer != "LIME" and explainer != "CAM":
        raise Exception('Explainer should be CAM or LIME')

    if mutation not in list_mutation:
        raise Exception('Mutation given in argument is not recognised')

    if len(sys.argv) == 4:
        layers_option = "last"
    elif len(sys.argv) == 5:
        layers_option = sys.argv[4]

    if layers_option != "last" and layers_option != "all":
        raise Exception('Layers to explain given in argument should be last or all')

    if explainer == "LIME" and len(sys.argv) == 5:
        raise Exception('LIME can only be used on the whole model, so it is not possible to create the animation')

    id_mutation = list_mutation.index(mutation)
    class_to_show = (id_mutation, mutation) # 0, G12C; 1, others; 2, WT

    return explainer, class_to_show, wsi, layers_option





def create_results_directory(explainer, class_to_show, WSI_to_explain):
    """Create the directory to store the results.
   
    Create the WSI_to_explain/ folder.
    Then create a subfolder for each kind of explaination.
    Args:
        explainer (str): The explainability method to use (CAM or LIME)
        class_to_show (str): The class to use to draw the heatmap.
        WSI_to_explain (str): The WSI to explain.
    Returns:
        None.
       
    Raises:
        OSError: Error while creating the folder.
    """
    try:  
        if not os.path.exists("results/" + WSI_to_explain + "/"):
            os.mkdir("results/" + WSI_to_explain + "/")

        if not os.path.exists("results/" + WSI_to_explain + "/" + explainer + "_QuPathPatches_" + class_to_show + "/"):
            os.mkdir("results/" + WSI_to_explain + "/" + explainer + "_QuPathPatches_" + class_to_show + "/")
       
    except OSError:  
        print ('Creation of the directory failed')





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





def merge_gradcam_heatmap_with_patch(patch, heatmap, alpha=0.4):
    """ Create and merge CAM heatmap on the original patch, then the image result is saved
    Source: fchollet
      - https://keras.io/examples/vision/grad_cam/
      - https://github.com/keras-team/keras-io
    Args:
        patch (array): The original patch
        heatmap (array): The heatmap of the original patch explaination
        alpha (float): The intensity of the heatmap
    Returns:
        superimposed_img (array): The original patch merge with the heatmap
    """
    # Transform the original patch
    img = tensorflow.keras.preprocessing.image.img_to_array(patch)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    heatmap[heatmap < 25] = 0
    heatmap[heatmap >= 25] = 255

    # Use a colormap to colorize heatmap
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

    return superimposed_img





def run_explainer_on_patch(explainer, model, class_to_show, patch, layer_to_explain):
    """Run the desired explainer on the given patch
    
    Transform the patch to fit the explaination requirement
    Then, generate a heatmap thanks to an explainer
    The heatmap is merged with the original patch 
   
    Source: fchollet
      - https://keras.io/examples/vision/grad_cam/
      - https://github.com/keras-team/keras-io
    Args:
        explainer (str): The explainer to use
        model (model): The model to use
        class_to_show (array): The mutation to analyze
        patch (array): The patch to explain
        layer_to_explain (str): The name of the layer to analyze
    Returns:
        patch_explained (array): The explained patch
    """
    patch_explained = patch

    if explainer == "CAM":
        # Transform patch
        patch_array = tensorflow.keras.preprocessing.image.img_to_array(patch)
        patch_array = np.expand_dims(patch_array, axis=0)
        patch_array = tensorflow.keras.applications.mobilenet_v2.preprocess_input(patch_array)

        # Generate class activation heatmap
        heatmap = make_gradcam_heatmap(patch_array, model, layer_to_explain, pred_index=class_to_show[0])
        patch_explained = merge_gradcam_heatmap_with_patch(patch, heatmap)

    elif explainer == "LIME":
        # Initialize Lime explainer
        LIME_explainer = lime_image.LimeImageExplainer()
        # Transform patch
        patch_array = tensorflow.keras.preprocessing.image.img_to_array(patch)
        patch_array *= rescale
        patch_array = np.expand_dims(patch_array, axis=0)

        # Create visualization
        explanation = LIME_explainer.explain_instance(np.squeeze(patch_array, axis=0).astype('double'), model.predict, top_labels=5, hide_color=0, num_samples=1000)
        # pros_and_cons
        ind = class_to_show[0]
        temp, mask = explanation.get_image_and_mask(ind, positive_only=False, num_features=10, hide_rest=False)
        patch_explained = mark_boundaries(temp, mask)
        patch_explained = np.uint8(patch_explained*255)
        patch_explained = PIL.Image.fromarray(patch_explained)

    return patch_explained





def apply_explainer_on_QuPath_patches(explainer, class_to_show, WSI_to_explain, layers_option):
    """Draw explaination on QuPath patches.
   
    Open the images & associated masks in patches_QuPath/ for the WSI_to_explain.
    A scale_factor (defined as magnification_level_WSI / magnification_level_patches),
    is used to create patches at the right magnification level.
    The tumor patches are created by a sliding window which browse the mask,
    and generate a patch (patch_width x patch_height) if the percentage of tumor area is greater than threshold_tumor_area.
    Then, the explainer is used on every single patch from the QuPatch patches.
    The QuPatch patches result are saved as .png file.

    Args:
        explainer (str): The explainability method to use (CAM or LIME)
        class_to_show (array): The class id and name to use to draw the heatmap.
        WSI_to_explain (str): The WSI to explain.
        layers_option (str): The option given by the used
    Returns:
        None
    Raises:
        ValueError: The image path or the mask path given does not exist.

    """
    scale_factor = int(magnification_level_WSI / magnification_level_patches)

    # Find the fold where to WSI is in the Test split
    fold_id = [fold_id+1 for fold_id in range(5) if WSI_to_explain in Folds[fold_id]][0]

    # Load model
    model = load_model(("results/model_K" + str(fold_id) + ".h5"))

    # Find the layers to explain
    layers_to_explain = [layer.name for layer in model.layers if isinstance(layer, tensorflow.keras.layers.Conv2D)]

    if layers_option == "last":
        layers_to_explain = [layers_to_explain[-1]]


    id_layer_to_explain = 1

    for layer_to_explain in layers_to_explain:

        for patch_QuPath in os.listdir(dataset_path + 'patches_QuPath/'):

            if 'image' in patch_QuPath and WSI_to_explain in patch_QuPath:
                path_image = dataset_path + 'patches_QuPath/' + patch_QuPath
                path_mask = dataset_path + 'patches_QuPath/' + patch_QuPath.replace('image','mask')
                if not os.path.isfile(path_image) :
                    raise ValueError('The image path given does not exist: ', path_image)
                if not os.path.isfile(path_mask) :
                    raise ValueError('The groundtruth image path given does not exist: ', path_mask)

                try:
                    # Open image & mask
                    image_result = PIL.Image.fromarray(imread(path_image))
                    image = imread(path_image)
                    mask = imread(path_mask)

                    # Sliding window
                    # print("#cells in grid:", mask.shape[1]/(patch_width*scale_factor), mask.shape[0]/(patch_height*scale_factor))
                    for i in range(0, mask.shape[1], patch_width*scale_factor):
                        for j in range(0, mask.shape[0], patch_height*scale_factor):
                            if i + patch_width*scale_factor < mask.shape[1] and j + patch_height*scale_factor < mask.shape[0]: # Patches are not created outside the image.
                                cropped_mask = mask[j:j+patch_height*scale_factor,i:i+patch_width*scale_factor]
                                ratio_tumor_area = np.count_nonzero(cropped_mask==255)/(np.count_nonzero(cropped_mask==0)+np.count_nonzero(cropped_mask==255))

                                # Create patch if the percentage of tumor area in cropped_mask is greater than threshold_tumor_area
                                if ratio_tumor_area >= threshold_tumor_area:
                                    patch = image[j:j+patch_height*scale_factor,i:i+patch_width*scale_factor]
                                    patch = PIL.Image.fromarray(patch)
                                   
                                    # Resize the crop to match the patch size & magnification level desired.
                                    patch = patch.resize((patch_width, patch_height), PIL.Image.NEAREST)
                                   
                                    # Run explainer on patch
                                    patch_explained = run_explainer_on_patch(explainer, model, class_to_show, patch, layer_to_explain)
                                    
                                    # Update image_result with explainer patch heatmap
                                    image_result.paste(patch_explained, (i, j))

                    # Save the QuPath patches with explaination
                    image_result.save("results/" + WSI_to_explain + "/" + explainer + "_QuPathPatches_" + class_to_show[1] + "/" + patch_QuPath.split("-")[0] + "_" + str(id_layer_to_explain) + ".png")
                    print("Successfully ran " + explainer + " on " + WSI_to_explain + " for " + class_to_show[1] + " mutation (#layer " + str(id_layer_to_explain) + ")")

                    gc.collect()
                except Exception as e:
                    print('Impossible to create patches from', path_image, 'error :', e)

        id_layer_to_explain += 1              





def create_animation(explainer, class_to_show, WSI_to_explain, layers_option):
    """Create an animated file (.gif) showing a QuPath patches explained by all the conv. layers
    
    Args:
        explainer (str): The explainability method to use (CAM or LIME)
        class_to_show (array): The class id and name to use to draw the heatmap.
        WSI_to_explain (str): The WSI to explain.
        layers_option (str): The option given by the used
    Returns:
        None
    Raises:
        ValueError: The image path or the mask path given does not exist.

    """
    if explainer == "CAM" and layers_option == "all":

        # Find unique QuPathPach
        QuPathPatches = ['_'.join(QuPathPatch.split("_")[:-1]) for QuPathPatch in os.listdir("results/" + WSI_to_explain + "/" + explainer + "_QuPathPatches_" + class_to_show[1] + "/")]
        QuPathPatches_unique = list(dict.fromkeys(QuPathPatches))

        # 
        path = "results/" + WSI_to_explain + "/" + explainer + "_QuPathPatches_" + class_to_show[1] + "/"
        for QuPatch in QuPathPatches_unique:
            sprites = [imageio.imread(path + QuPathPatch) for QuPathPatch in os.listdir("results/" + WSI_to_explain + "/" + explainer + "_QuPathPatches_" + class_to_show[1] + "/") if QuPatch in QuPathPatch]
            imageio.mimsave(path + "/" + QuPatch + ".gif", sprites)






if __name__ == "__main__":

    # Get script arguments
    explainer, class_to_show, WSI_to_explain, layers_option = get_and_filter_arguments()

    # Create hierarchy
    create_results_directory(explainer, class_to_show[1], WSI_to_explain)

    # Draw explaination on QuPath patches
    apply_explainer_on_QuPath_patches(explainer, class_to_show, WSI_to_explain, layers_option)

    # Create animation
    create_animation(explainer, class_to_show, WSI_to_explain, layers_option)
