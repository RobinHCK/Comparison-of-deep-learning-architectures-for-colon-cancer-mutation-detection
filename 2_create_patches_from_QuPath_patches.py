from config import *
import numpy as np
import random
random.seed(seed)
import os
import sys
from skimage.io import imread
from skimage.color import gray2rgb
import gc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 20000000000





def create_dataset_hierarchy(dataset_path, chosen_wsi):
    """Create the dataset hierarchy.
    
    Create the patches_organized_per_wsi/ folder and one subfolder for each WSI listed in chosen_wsi.

    Args:
        dataset_path (str): The dataset path to use.
        chosen_wsi (array str): The WSI names to use to create patches.

    Returns:
        None.
        
    Raises:
        OSError: Error while creating the folders.

    """
    try:  
        if not os.path.exists(dataset_path + "patches_organized_per_wsi/"):
            os.mkdir(dataset_path + "patches_organized_per_wsi/")
        for image in chosen_wsi:
            if not os.path.exists(dataset_path + "patches_organized_per_wsi/" + image + "/"):
                os.mkdir(dataset_path + "patches_organized_per_wsi/" + image + "/")
        
    except OSError:  
        print ('Creation of the directory failed')



def create_patches_from_QuPath_patches():
    """Create tumor patches from images and masks.
    
    Open the image & associated mask for each patch in patches_QuPath/.
    A scale_factor (defined as magnification_level_WSI / magnification_level_patches),
    is used to create patches at the right magnification level.
    The tumor patches are created by a sliding window which browse the mask,
    and generate a patch (patch_width x patch_height) if the percentage of tumor area is greater than threshold_tumor_area.
    The patch result is saved as .png file in patches_organized_per_wsi/wsi_id/ folder.

    Args:
        None.

    Returns:
        None.
        
    Raises:
        ValueError: The image path or the mask path given does not exist.

    """
    patch_id = 0
    scale_factor = int(magnification_level_WSI / magnification_level_patches)
    
    for patch_QuPath in os.listdir(dataset_path + 'patches_QuPath/'):

        if 'image' in patch_QuPath:
            path_image = dataset_path + 'patches_QuPath/' + patch_QuPath
            path_mask = dataset_path + 'patches_QuPath/' + patch_QuPath.replace('image','mask')
            if not os.path.isfile(path_image) :
                raise ValueError('The image path given does not exist: ', path_image)
            if not os.path.isfile(path_mask) :
                raise ValueError('The groundtruth image path given does not exist: ', path_mask)

            try:
                # Open image & mask
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
                                cropped_image = image[j:j+patch_height*scale_factor,i:i+patch_width*scale_factor]
                                cropped_image = PIL.Image.fromarray(cropped_image)
                                
                                # Resize the crop to match the patch size & magnification level desired.
                                cropped_image = cropped_image.resize((patch_width, patch_height), PIL.Image.NEAREST)
                                
                                cropped_image.save(dataset_path + 'patches_organized_per_wsi/' + ('_'.join(patch_QuPath[:-4].split('-')[0].split('_')[:-1]))[:-1] + '/' + patch_QuPath[:-4] + '_' + str(patch_id) + '.png')
                                
                                print('Patch created: n°', patch_id, ' ', patch_QuPath[:-4] + '_' + str(patch_id), 'at', str(magnification_level_patches), 'X -', str(patch_height), 'x', str(patch_width), 'px')
                                patch_id += 1

                gc.collect()
            except Exception as e:
                print('Impossible to create patches from', path_image, 'error :', e)



def remove_patches_from_the_same_WSI(nbr_patches_to_keep):
    """Removes patches from the same WSI to balanced folds and classes
    
    If a WSI has too much patches (i.e > nbr_patches_to_keep/(number_of_folds x number_of_classes)), a percentage of the patches from each WSI are removed.

    Args:
        nbr_patches_to_keep (int): The number of patches to keep.

    Returns:
        None.
        
    Raises:
        ValueError: The patch path to remove does not exist.
        
    """  
    nbr_patches_to_keep_per_fold = nbr_patches_to_keep / 5

    nbr_wsi_above_limit = 0
    nbr_patches_removed = 0

    nbr_patches_per_wsi_WT = sum([len(os.listdir(dataset_path + 'patches_organized_per_wsi/' + wsi_id)) for wsi_id in WT])
    nbr_patches_per_wsi_G12C = sum([len(os.listdir(dataset_path + 'patches_organized_per_wsi/' + wsi_id)) for wsi_id in G12C])
    nbr_patches_per_wsi_others = sum([len(os.listdir(dataset_path + 'patches_organized_per_wsi/' + wsi_id)) for wsi_id in others])
    
    for fold_id in range(1,len(Folds)+1):
        # WT
        nbr_patches_per_wsi_in_fold_WT = [[len(os.listdir(dataset_path + 'patches_organized_per_wsi/' + wsi_id)) for wsi_id in WT][WT.index(wsi_id)] for wsi_id in Folds[fold_id-1] if wsi_id in WT]
        wsi_id_in_fold_WT = [wsi_id for wsi_id in Folds[fold_id-1] if wsi_id in WT]
        nbr_patches_in_fold_WT = sum(nbr_patches_per_wsi_in_fold_WT)
        nbr_patches_per_wsi_in_fold_WT_to_keep = [int((nbr_patches/nbr_patches_in_fold_WT)*(nbr_patches_to_keep_per_fold/3)) for nbr_patches in nbr_patches_per_wsi_in_fold_WT]

        if sum(nbr_patches_per_wsi_in_fold_WT_to_keep) < nbr_patches_to_keep_per_fold//3:
            nbr_patches_per_wsi_in_fold_WT_to_keep[nbr_patches_per_wsi_in_fold_WT_to_keep.index(max(nbr_patches_per_wsi_in_fold_WT_to_keep))] += nbr_patches_to_keep_per_fold//3 - sum(nbr_patches_per_wsi_in_fold_WT_to_keep)
        elif sum(nbr_patches_per_wsi_in_fold_WT_to_keep) > nbr_patches_to_keep_per_fold//3:
            nbr_patches_per_wsi_in_fold_WT_to_keep[nbr_patches_per_wsi_in_fold_WT_to_keep.index(max(nbr_patches_per_wsi_in_fold_WT_to_keep))] -= sum(nbr_patches_per_wsi_in_fold_WT_to_keep) - nbr_patches_to_keep_per_fold//3

        for wsi_id, nbr_patches_to_keep in zip(wsi_id_in_fold_WT, nbr_patches_per_wsi_in_fold_WT_to_keep):
            patches = os.listdir(dataset_path + 'patches_organized_per_wsi/' + wsi_id)
            random.shuffle(patches)
            patches_to_remove = patches[int(nbr_patches_to_keep):]

            for patch_to_remove in patches_to_remove:
                path_patch_to_remove = dataset_path + 'patches_organized_per_wsi/' + wsi_id + '/' + patch_to_remove

                if os.path.exists(path_patch_to_remove):
                    os.remove(path_patch_to_remove)
                else:
                    raise ValueError('The patch path to remove does not exist: ', path_patch_to_remove)

            nbr_wsi_above_limit += 1
            nbr_patches_removed += len(patches_to_remove)
        
        # G12C    
        nbr_patches_per_wsi_in_fold_G12C = [[len(os.listdir(dataset_path + 'patches_organized_per_wsi/' + wsi_id)) for wsi_id in G12C][G12C.index(wsi_id)] for wsi_id in Folds[fold_id-1] if wsi_id in G12C]
        wsi_id_in_fold_G12C = [wsi_id for wsi_id in Folds[fold_id-1] if wsi_id in G12C]
        nbr_patches_in_fold_G12C = sum(nbr_patches_per_wsi_in_fold_G12C)
        nbr_patches_per_wsi_in_fold_G12C_to_keep = [int((nbr_patches/nbr_patches_in_fold_G12C)*(nbr_patches_to_keep_per_fold/3)) for nbr_patches in nbr_patches_per_wsi_in_fold_G12C]

        if sum(nbr_patches_per_wsi_in_fold_G12C_to_keep) < nbr_patches_to_keep_per_fold//3:
            nbr_patches_per_wsi_in_fold_G12C_to_keep[nbr_patches_per_wsi_in_fold_G12C_to_keep.index(max(nbr_patches_per_wsi_in_fold_G12C_to_keep))] += nbr_patches_to_keep_per_fold//3 - sum(nbr_patches_per_wsi_in_fold_G12C_to_keep)
        elif sum(nbr_patches_per_wsi_in_fold_G12C_to_keep) > nbr_patches_to_keep_per_fold//3:
            nbr_patches_per_wsi_in_fold_G12C_to_keep[nbr_patches_per_wsi_in_fold_G12C_to_keep.index(max(nbr_patches_per_wsi_in_fold_G12C_to_keep))] -= sum(nbr_patches_per_wsi_in_fold_G12C_to_keep) - nbr_patches_to_keep_per_fold//3

        for wsi_id, nbr_patches_to_keep in zip(wsi_id_in_fold_G12C, nbr_patches_per_wsi_in_fold_G12C_to_keep):
            patches = os.listdir(dataset_path + 'patches_organized_per_wsi/' + wsi_id)
            random.shuffle(patches)
            patches_to_remove = patches[int(nbr_patches_to_keep):]
            
            for patch_to_remove in patches_to_remove:
                path_patch_to_remove = dataset_path + 'patches_organized_per_wsi/' + wsi_id + '/' + patch_to_remove
            
                if os.path.exists(path_patch_to_remove):
                    os.remove(path_patch_to_remove)
                else:
                    raise ValueError('The patch path to remove does not exist: ', path_patch_to_remove)

            nbr_wsi_above_limit += 1
            nbr_patches_removed += len(patches_to_remove)

        # others    
        nbr_patches_per_wsi_in_fold_others = [[len(os.listdir(dataset_path + 'patches_organized_per_wsi/' + wsi_id)) for wsi_id in others][others.index(wsi_id)] for wsi_id in Folds[fold_id-1] if wsi_id in others]
        wsi_id_in_fold_others = [wsi_id for wsi_id in Folds[fold_id-1] if wsi_id in others]
        nbr_patches_in_fold_others = sum(nbr_patches_per_wsi_in_fold_others)
        nbr_patches_per_wsi_in_fold_others_to_keep = [int((nbr_patches/nbr_patches_in_fold_others)*(nbr_patches_to_keep_per_fold/3)) for nbr_patches in nbr_patches_per_wsi_in_fold_others]

        if sum(nbr_patches_per_wsi_in_fold_others_to_keep) < nbr_patches_to_keep_per_fold//3:
            nbr_patches_per_wsi_in_fold_others_to_keep[nbr_patches_per_wsi_in_fold_others_to_keep.index(max(nbr_patches_per_wsi_in_fold_others_to_keep))] += nbr_patches_to_keep_per_fold//3 - sum(nbr_patches_per_wsi_in_fold_others_to_keep)
        elif sum(nbr_patches_per_wsi_in_fold_others_to_keep) > nbr_patches_to_keep_per_fold//3:
            nbr_patches_per_wsi_in_fold_others_to_keep[nbr_patches_per_wsi_in_fold_others_to_keep.index(max(nbr_patches_per_wsi_in_fold_others_to_keep))] -= sum(nbr_patches_per_wsi_in_fold_others_to_keep) - nbr_patches_to_keep_per_fold//3

        for wsi_id, nbr_patches_to_keep in zip(wsi_id_in_fold_others, nbr_patches_per_wsi_in_fold_others_to_keep):
            patches = os.listdir(dataset_path + 'patches_organized_per_wsi/' + wsi_id)
            random.shuffle(patches)
            patches_to_remove = patches[int(nbr_patches_to_keep):]
            
            for patch_to_remove in patches_to_remove:
                path_patch_to_remove = dataset_path + 'patches_organized_per_wsi/' + wsi_id + '/' + patch_to_remove
            
                if os.path.exists(path_patch_to_remove):
                    os.remove(path_patch_to_remove)
                else:
                    raise ValueError('The patch path to remove does not exist: ', path_patch_to_remove)

            nbr_wsi_above_limit += 1
            nbr_patches_removed += len(patches_to_remove)
        
    print('Removed', nbr_patches_removed, 'patches from', nbr_wsi_above_limit, 'WSI from classes WT/G12C/others.')
               
    

def draw_sliding_window_on_QuPath_patch(patch_QuPath):
    """Draw the sliding window on QuPath patch.
    
    Draw the patches selected by the sliding windows on one image.    
    Three images (original image and mask + image with drawing) are concatenated in one before to be saved.

    Args:
        patch_QuPath (str): The QuPath patch to use.

    Returns:
        None.
        
    Raises:
        ValueError: The image path or the mask path given does not exist.
        Exception: Impossible to create patches.
        
    """
    scale_factor = int(magnification_level_WSI / magnification_level_patches)

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

                    # Draw patch in green
                    if ratio_tumor_area >= threshold_tumor_area:
                        #image_result[j-5:j+patch_height*scale_factor-5,i+5:i+patch_width*scale_factor-5] = [0,255,0]
                        image_result[j:j+10,i:i+patch_width*scale_factor,:] = [0,255,0]
                        image_result[j:j+patch_height*scale_factor,i:i+10,:] = [0,255,0]
                        image_result[j:j+patch_height*scale_factor,i+patch_width*scale_factor-10:i+patch_width*scale_factor,:] = [0,255,0]
                        image_result[j+patch_height*scale_factor-10:j+patch_height*scale_factor,i:i+patch_width*scale_factor,:] = [0,255,0]

        
        # Save the concatenated images as one image which shows the original image, the mask and the patches selected by the sliding windows. 
        PIL.Image.fromarray(np.hstack((image,image_result,gray2rgb(mask)))).save('results/sliding_window_' + patch_QuPath[:-4] + '_threshold_' + str(threshold_tumor_area) + '.png')
    
    except Exception as e:
        print('Impossible to create patches from', path_image, 'error :', e)
                
                
                
def create_graphs_dataset_statistics():
    """Create & save graphs which show the dataset distribution
    
    Create 1 graph (bar chart) which shows the number of patches per WSI, sorted by class (WT, G12C, others).
    Create 1 graph (pie chart) which shows the percentage of patches per class (WT, G12C, others).
    Create 1 graph (bar chart) which shows the number of patches per fold, sorted by class (WT, G12C, others).
    Create 5 graphs (pie charts) which show the percentage of patches per WSI and per class (WT, G12C, others) for each fold.
    Create 3 graphs (plot charts) which show the color distribution for each channel (R, G and B).

    Args:
        None.

    Returns:
        None.
    """
    create_graph_bar_number_of_patches_per_wsi()
    
    create_graph_pie_percentage_of_patches_per_class()
    
    create_graph_bar_number_of_patches_per_fold()
    
    create_graphs_pie_percentage_of_patches_per_wsi_per_class()
    
    #create_RGB_histograms()
          


def create_graph_bar_number_of_patches_per_wsi():
    """Create 1 graph (bar chart) which shows the number of patches per WSI, sorted by class (WT, G12C, others).
    
    Each class has a color code : WT '#7aa6c2' ; G12C '#58a066' ; others '#e6ba95'
    The graph is saved as .pdf file to enhance his quality.

    Args:
        None.

    Returns:
        None.
    """
    nbr_patches_per_wsi_WT = []
    nbr_patches_per_wsi_G12C = []
    nbr_patches_per_wsi_others = []
    
    for wsi_id in WT:
        nbr_patches_per_wsi_WT.append(len(os.listdir(dataset_path + 'patches_organized_per_wsi/' + wsi_id)))
    for wsi_id in G12C:
        nbr_patches_per_wsi_G12C.append(len(os.listdir(dataset_path + 'patches_organized_per_wsi/' + wsi_id)))
    for wsi_id in others:
        nbr_patches_per_wsi_others.append(len(os.listdir(dataset_path + 'patches_organized_per_wsi/' + wsi_id)))
    
    nbr_patches_per_wsi_WT_sorted, WT_sorted = zip(*sorted(zip(nbr_patches_per_wsi_WT, WT), reverse=True))
    nbr_patches_per_wsi_G12C_sorted, G12C_sorted = zip(*sorted(zip(nbr_patches_per_wsi_G12C, G12C), reverse=True))
    nbr_patches_per_wsi_others_sorted, others_sorted = zip(*sorted(zip(nbr_patches_per_wsi_others, others), reverse=True))
    
    x = np.arange(len(WT_sorted) + len(G12C_sorted) + len(others_sorted))
    width = 0.75
    
    fig, ax = plt.subplots()
    rects = ax.bar(x, nbr_patches_per_wsi_WT_sorted + nbr_patches_per_wsi_G12C_sorted + nbr_patches_per_wsi_others_sorted, width)
    [rects[i].set_color('#7aa6c2') for i in range(len(WT_sorted))]
    [rects[i].set_color('#58a066') for i in range(len(WT_sorted),len(WT_sorted)+len(G12C_sorted))]
    [rects[i].set_color('#e6ba95') for i in range(len(WT_sorted)+len(G12C_sorted), len(WT_sorted)+len(G12C_sorted)+len(others_sorted))]
    
    ax.set_xlabel('WSI')
    ax.set_ylabel('Number of patches')
    ax.set_title('Number of patches per WSI')
    ax.set_xticks(x)
    ax.set_xticklabels(WT_sorted + G12C_sorted + others_sorted, rotation=90)
    legend_elements = [Line2D([0], [0], color='#7aa6c2', lw=4, label='WT'), Line2D([0], [0], color='#58a066', lw=4, label='G12C'), Line2D([0], [0], color='#e6ba95', lw=4, label='others')]
    ax.legend(handles=legend_elements)
    plt.grid(axis='y')  
    fig.tight_layout()
    
    plt.savefig('results/number_of_patches_per_wsi.pdf')
    plt.clf()
        
    
    
def create_graph_pie_percentage_of_patches_per_class():
    """ Create 1 graph (pie chart) which shows the percentage of patches per class (WT, G12C, others).
    
    Each class has a color code : WT '#7aa6c2' ; G12C '#58a066' ; others '#e6ba95'
    The graph is saved as .pdf file to enhance his quality.

    Args:
        None.

    Returns:
        None.
    """
    nbr_patches_per_wsi_WT = []
    nbr_patches_per_wsi_G12C = []
    nbr_patches_per_wsi_others = []
    
    for wsi_id in WT:
        nbr_patches_per_wsi_WT.append(len(os.listdir(dataset_path + 'patches_organized_per_wsi/' + wsi_id)))
    for wsi_id in G12C:
        nbr_patches_per_wsi_G12C.append(len(os.listdir(dataset_path + 'patches_organized_per_wsi/' + wsi_id)))
    for wsi_id in others:
        nbr_patches_per_wsi_others.append(len(os.listdir(dataset_path + 'patches_organized_per_wsi/' + wsi_id)))
        
    labels = 'WT ('+str(sum(nbr_patches_per_wsi_WT))+')','G12C ('+str(sum(nbr_patches_per_wsi_G12C))+')','others ('+str(sum(nbr_patches_per_wsi_others))+')'
    sizes = [sum(nbr_patches_per_wsi_WT),sum(nbr_patches_per_wsi_G12C),sum(nbr_patches_per_wsi_others)]
    
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#7aa6c2','#58a066','#e6ba95'])
    ax.set_title('Percentage of patches per class')
    ax.axis('equal')
    legend_elements = [Line2D([0], [0], color='#7aa6c2', lw=4, label='WT'), Line2D([0], [0], color='#58a066', lw=4, label='G12C'), Line2D([0], [0], color='#e6ba95', lw=4, label='others')]
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.9, 0.9))
    
    plt.savefig('results/percentage_of_patches_per_class.pdf')
    plt.clf()
    
    
    
def create_graph_bar_number_of_patches_per_fold():
    """ Create 1 graph (bar chart) which shows the number of patches per fold, sorted by class (WT, G12C, others).
    
    Each class has a color code : WT '#7aa6c2' ; G12C '#58a066' ; others '#e6ba95'
    The graph is saved as .pdf file to enhance his quality.

    Args:
        None.

    Returns:
        None.
    """
    nbr_patches_per_fold_WT = []
    nbr_patches_per_fold_G12C = []
    nbr_patches_per_fold_others = []

    for fold in Folds:        
        nbr_patches_per_fold_WT.append(sum([len(os.listdir(dataset_path + 'patches_organized_per_wsi/' + wsi_id)) for wsi_id in fold if wsi_id in WT]))
        nbr_patches_per_fold_G12C.append(sum([len(os.listdir(dataset_path + 'patches_organized_per_wsi/' + wsi_id)) for wsi_id in fold if wsi_id in G12C]))
        nbr_patches_per_fold_others.append(sum([len(os.listdir(dataset_path + 'patches_organized_per_wsi/' + wsi_id)) for wsi_id in fold if wsi_id in others]))

    x = np.arange(len(nbr_patches_per_fold_WT))
    width = 0.2
    
    fig, ax = plt.subplots()
    rects_WT = ax.bar(x-width, nbr_patches_per_fold_WT, width, tick_label=nbr_patches_per_fold_WT)
    [rects_WT[i].set_color('#7aa6c2') for i in range(len(nbr_patches_per_fold_WT))]
    rects_G12C = ax.bar(x, nbr_patches_per_fold_G12C, width)
    [rects_G12C[i].set_color('#58a066') for i in range(len(nbr_patches_per_fold_G12C))]
    rects_others = ax.bar(x+width, nbr_patches_per_fold_others, width)
    [rects_others[i].set_color('#e6ba95') for i in range(len(nbr_patches_per_fold_others))]
    
    ax.set_xlabel('Fold')
    ax.set_ylabel('Number of patches')
    ax.set_title('Number of patches per fold')
    ax.set_xticks(x)
    ###
    # Matplotlib < 3.4
    ax.set_xticklabels([str(e[0]//1000) + 'K ' + str(e[1]//1000) + 'K ' + str(e[2]//1000) + 'K\n' + str(i+1) for i,e in enumerate(np.vstack((nbr_patches_per_fold_WT,nbr_patches_per_fold_G12C,nbr_patches_per_fold_others)).T)])
    # Matplotlib >= 3.4
    #ax.set_xticklabels([1,2,3,4,5])
    #ax.bar_label(rects_WT, padding=3)
    #ax.bar_label(rects_G12C, padding=3)
    #ax.bar_label(rects_others, padding=3)
    ###
    legend_elements = [Line2D([0], [0], color='#7aa6c2', lw=4, label='WT'), Line2D([0], [0], color='#58a066', lw=4, label='G12C'), Line2D([0], [0], color='#e6ba95', lw=4, label='others')]
    ax.legend(handles=legend_elements)
    plt.grid(axis='y')
    fig.tight_layout()

    plt.savefig('results/number_of_patches_per_fold.pdf')
    plt.clf()
    
    
    
def create_graphs_pie_percentage_of_patches_per_wsi_per_class():
    """Create 5 graphs (pie charts) which show the percentage of patches per WSI and per class (WT, G12C, others) for each fold.
    
    Each class has a color code : WT '#7aa6c2' ; G12C '#58a066' ; others '#e6ba95'
    The graph is saved as .pdf file to enhance his quality.

    Args:
        None.

    Returns:
        None.
    """
    id_fold = 0
    for fold in Folds:  
        id_fold += 1
        number_of_WT_patches_in_fold = [len(os.listdir(dataset_path + 'patches_organized_per_wsi/' + wsi_id)) for wsi_id in fold if wsi_id in WT]
        number_of_G12C_patches_in_fold = [len(os.listdir(dataset_path + 'patches_organized_per_wsi/' + wsi_id)) for wsi_id in fold if wsi_id in G12C]
        number_of_others_patches_in_fold = [len(os.listdir(dataset_path + 'patches_organized_per_wsi/' + wsi_id)) for wsi_id in fold if wsi_id in others]
    
        fig, ax = plt.subplots()
        size = 0.3
        outer_colors = ['#7aa6c2','#58a066','#e6ba95']
        inner_colors = ['#7aa6c2']*len(number_of_WT_patches_in_fold) + ['#58a066']*len(number_of_G12C_patches_in_fold) + ['#e6ba95']*len(number_of_others_patches_in_fold)
        ax.pie([sum(number_of_WT_patches_in_fold), sum(number_of_G12C_patches_in_fold),  sum(number_of_others_patches_in_fold)], radius=1, pctdistance=0.85, colors=outer_colors, labels=['WT','G12C','others'], autopct='%1.1f%%', wedgeprops=dict(width=size, edgecolor='w'))
        ax.pie(number_of_WT_patches_in_fold + number_of_G12C_patches_in_fold + number_of_others_patches_in_fold, radius=1-size, pctdistance=0.75, colors=inner_colors, autopct='%1.1f%%', wedgeprops=dict(width=size, edgecolor='w'))
        ax.set(aspect="equal", title='Percentage of patches per wsi per class for fold ' + str(id_fold))

        label_WT = ''.join([line[0] + '     [' + line[1] + ' patches - ' + str(round(int(line[1])*100/(sum(number_of_WT_patches_in_fold)+sum(number_of_G12C_patches_in_fold)+sum(number_of_others_patches_in_fold)),1)) + '%]\n' for line in np.vstack(([wsi_id for wsi_id in fold if wsi_id in WT], [str(nbr_patches) for nbr_patches in number_of_WT_patches_in_fold])).T])
        label_G12C = ''.join([line[0] + '     [' + line[1] + ' patches - ' + str(round(int(line[1])*100/(sum(number_of_WT_patches_in_fold)+sum(number_of_G12C_patches_in_fold)+sum(number_of_others_patches_in_fold)),1)) + '%]\n' for line in np.vstack(([wsi_id for wsi_id in fold if wsi_id in G12C], [str(nbr_patches) for nbr_patches in number_of_G12C_patches_in_fold])).T])
        label_others = ''.join([line[0] + '     [' + line[1] + ' patches - ' + str(round(int(line[1])*100/(sum(number_of_WT_patches_in_fold)+sum(number_of_G12C_patches_in_fold)+sum(number_of_others_patches_in_fold)),1)) + '%]\n' for line in np.vstack(([wsi_id for wsi_id in fold if wsi_id in others], [str(nbr_patches) for nbr_patches in number_of_others_patches_in_fold])).T])
        legend_elements = [Line2D([0], [0], color='#7aa6c2', lw=4, label=label_WT), Line2D([0], [0], color='#58a066', lw=4, label=label_G12C), Line2D([0], [0], color='#e6ba95', lw=4, label=label_others)]
        ax.legend(handles=legend_elements, bbox_to_anchor=(0.9, 0.9))
        
        plt.savefig('results/percentage_of_patches_per_wsi_per_class_fold_' + str(id_fold) + '.pdf', bbox_inches="tight")
        plt.clf()
        
        

def create_RGB_histograms():
    """Create 3 graphs (plot charts) which show the intensity distribution for each WSI, for each channel (R, G and B).

    Each class has a color code : WT '#7aa6c2' ; G12C '#58a066' ; others '#e6ba95'
    The graph is saved as .pdf file to enhance his quality.

    Args:
        None.

    Returns:
        None.
    """
    histograms_R = []
    histograms_G = []
    histograms_B = []
    
    for wsi_id in WT+G12C+others:
        patch_histogram_R = np.zeros(256)
        patch_histogram_G = np.zeros(256)
        patch_histogram_B = np.zeros(256)
        
        patch_paths = os.listdir(dataset_path + 'patches_organized_per_wsi/' + wsi_id)
        nbr_patches_in_wsi = len(patch_paths)
        
        for patch_path in patch_paths:
            patch_img = np.asarray(PIL.Image.open(dataset_path + 'patches_organized_per_wsi/' + wsi_id + "/" + patch_path))
            patch_histogram_R = patch_histogram_R + np.histogram(patch_img[:,:,2], bins=np.arange(257))[0]
            patch_histogram_G = patch_histogram_G + np.histogram(patch_img[:,:,1], bins=np.arange(257))[0]
            patch_histogram_B = patch_histogram_B + np.histogram(patch_img[:,:,0], bins=np.arange(257))[0]
            
        histograms_R.append(patch_histogram_R / nbr_patches_in_wsi)
        histograms_G.append(patch_histogram_G / nbr_patches_in_wsi)
        histograms_B.append(patch_histogram_B / nbr_patches_in_wsi)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,4))
    fig.suptitle('Histograms RGB')

    for i in range(len(WT+G12C+others)):
        color = None

        if i < len(WT):
            color = "#7aa6c2"
        elif i >= len(WT) and i < len(WT)+len(G12C):
            color = "#58a066"
        elif i >= len(WT)+len(G12C):
            color = "#e6ba95"

        ax1.plot(histograms_R[i], color=color, linewidth="0.5")
        ax2.plot(histograms_G[i], color=color, linewidth="0.5")
        ax3.plot(histograms_B[i], color=color, linewidth="0.5")

    ax1.set(xlabel='', ylabel='Number of pixels',title='R')
    ax1.grid()

    ax2.set(xlabel='Intensity', ylabel='Number of pixels',title='G')
    ax2.grid()
    ax2.label_outer()

    ax3.set(xlabel='', ylabel='Number of pixels',title='B')
    ax3.grid()
    ax3.label_outer()

    ylim = [0, np.amax([histograms_R, histograms_G, histograms_B])]
    ax1.set_ylim(ylim)
    ax2.set_ylim(ylim)
    ax3.set_ylim(ylim)

    legend_elements = [Line2D([0], [0], color='#7aa6c2', lw=4, label="WT"), Line2D([0], [0], color='#58a066', lw=4, label="G12C"), Line2D([0], [0], color='#e6ba95', lw=4, label="others")]
    ax3.legend(handles=legend_elements, bbox_to_anchor=(0.9, 0.9))
    fig.tight_layout()
    plt.savefig('results/histogram_RGB.pdf', bbox_inches="tight")
    plt.clf()
        

    
    
    
if __name__ == "__main__":

    # Create hierarchy
    create_dataset_hierarchy(dataset_path, WT + G12C + others)

    # Create patches from QuPath patches
    create_patches_from_QuPath_patches()
    
    # Removes patches to balanced the dataset
    remove_patches_from_the_same_WSI(nbr_patches_to_keep=76000)

    # Draw & save graphs & save infos about dataset statistics
    create_graphs_dataset_statistics()
    
    # Draw the sliding window on a specific QuPath patch
    #draw_sliding_window_on_QuPath_patch(patch_QuPath='image_to_draw.png')
    #draw_sliding_window_on_QuPath_patch(patch_QuPath='image_to_draw.png')
    
