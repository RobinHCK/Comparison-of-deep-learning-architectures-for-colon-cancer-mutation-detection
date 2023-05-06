from config import *
import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.io import imread
from skimage.color import gray2rgb
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 20000000000
import gc





def draw_preds_on_WSI(wsi_id):
    """None.

    Args:
        None.

    Returns:
        None.
        
    """

    # Get all patches_QuPath belonging to wsi_id
    patches_QuPath_to_predict = []

    for patch_QuPath in os.listdir(dataset_path + 'patches_QuPath/'):
        if 'image' in patch_QuPath and wsi_id in patch_QuPath:
            patches_QuPath_to_predict.append(patch_QuPath)

    # Select the model to apply
    network_id = None

    for i in range(5):
        if wsi_id in Folds[i]:
            network_id = i+1

    if not network_id:
        raise ValueError('The chosen WSI is not in any fold:', wsi_id)

    model = load_model(("results/model_K" + str(network_id) + ".h5"))

    # Create/Test patches on the flow & Draw preds on QuPath patches
    scale_factor = int(magnification_level_WSI / magnification_level_patches)

    for patch_QuPath in patches_QuPath_to_predict:

        path_image = dataset_path + 'patches_QuPath/' + patch_QuPath
        path_mask = dataset_path + 'patches_QuPath/' + patch_QuPath.replace('image','mask')
        if not os.path.isfile(path_image) :
            raise ValueError('The image path given does not exist: ', path_image)
        if not os.path.isfile(path_mask) :
            raise ValueError('The groundtruth image path given does not exist: ', path_mask)

        try:
            # Open image & mask
            image_result = imread(path_image)
            image = imread(path_image)
            mask = imread(path_mask)

            # Sliding window
            for i in range(0, mask.shape[1], patch_width*scale_factor):
                for j in range(0, mask.shape[0], patch_height*scale_factor):
                    if i + patch_width*scale_factor < mask.shape[1] and j + patch_height*scale_factor < mask.shape[0]: # Patches are not created outside the image.
                        cropped_mask = mask[j:j+patch_height*scale_factor,i:i+patch_width*scale_factor]
                        ratio_tumor_area = np.count_nonzero(cropped_mask==255)/(np.count_nonzero(cropped_mask==0)+np.count_nonzero(cropped_mask==255))

                        # Create patch if the percentage of tumor area in cropped_mask is greater than threshold_tumor_area
                        if ratio_tumor_area >= threshold_tumor_area:
                            cropped_patch = image[j:j+patch_height*scale_factor,i:i+patch_width*scale_factor]
                            cropped_patch = PIL.Image.fromarray(cropped_patch)
                            
                            # Resize the crop to match the patch size & magnification level desired.
                            cropped_patch = cropped_patch.resize((patch_width, patch_height), PIL.Image.NEAREST)
                            cropped_patch = img_to_array(cropped_patch)
                            cropped_patch = cropped_patch.reshape((1,) + cropped_patch.shape)

                            # Test model on cropp
                            test_datagen = ImageDataGenerator(rescale = rescale)
                            test_generator = test_datagen.flow(x=cropped_patch, batch_size=1)

                            y_pred = model.predict_generator(test_generator)
                            y_class = y_pred.argmax(axis=-1)[0]

                            # Draw pred
                            color_WT = [122,166,194] #7aa6c2
                            color_G12C = [88,160,102] #58a066
                            color_others = [230,186,149] #e6ba95
                            class_color = None

                            if y_class == 0:
                                class_color = color_G12C
                            elif y_class == 1:
                                class_color = color_others
                            elif y_class == 2:
                                class_color = color_WT
                                    
                            image_result[j:j+patch_height*scale_factor,i:i+patch_width*scale_factor] = class_color
            
            # Save the concatenated images as one image which shows the original image, the mask and the patches selected by the sliding windows. 
            PIL.Image.fromarray(np.hstack((image,image_result,gray2rgb(mask)))).save('results/draw_predictions_' + patch_QuPath[:-4] + '_model_' + str(network_id) + '.png')    
            print('Draw predictions made by model',network_id ,'on QuPath patch', patch_QuPath, 'from WSI', wsi_id)

            gc.collect()
        except Exception as e:
            print('Impossible to create patches from', path_image, 'error :', e)





if __name__ == "__main__":
    # Draw the predictions for one WSI
    draw_preds_on_WSI(wsi_id="WSI_to_draw")
