from config import *
from pathml.preprocessing import StainNormalizationHE
from PIL import Image
import numpy as np
import os



def overwrite_patch(wsi, path_patch, normalize_patch):
    """Overwrite a patch with the normalized result

    Args:
        wsi (string): The wsi name.
        path_patch (string): The patch path.
        normalize_patch (array Image): The normalized patch.

    """
    normalize_patch.save(dataset_path + "patches_organized_per_wsi/" + wsi + "/" + path_patch)



def normalize_patch(patch, normalizer):
    """Normalize a patch 
    
    Apply color normalization on a patch with Macenko method.
    For more information, see: https://pathml.readthedocs.io/en/latest/examples/link_stain_normalization.html

    Args:
        patch (array Image): The patch to normalize.
        normalizer (StainNormalizationHE): The normalizer to use.

    Returns:
        _ (array Image): The normalized patch.

    """
    return Image.fromarray(normalizer.F(patch))
        

        
def normalize_patches_from_WSI(chosen_WSI, normalizer):
    """Normalize every patches of each WSI
    
    Each patch will be normalized and then overwrite.

    Args:
        chosen_WSI (array string): The chosen WSI to use.
        normalizer (StainNormalizationHE): The normalizer to use.

    """
    for wsi in chosen_WSI:
        for patch_name in os.listdir(dataset_path + "patches_organized_per_wsi/" + wsi):
            patch = np.array(Image.open(dataset_path + "patches_organized_per_wsi/" + wsi + "/" + patch_name))
            
            try:
                normalized_patch = normalize_patch(patch, normalizer)
                
                overwrite_patch(wsi, patch_name, normalized_patch)
                
                print('Normalize patch', patch_name, 'from', wsi)
            
            except:
                print('Fail to normalize patch', patch_name, 'from', wsi)
    



    
if __name__ == "__main__":
    
    # Create normalizer https://pathml.readthedocs.io/en/latest/examples/link_stain_normalization.html
    normalizer = StainNormalizationHE(target="normalize", stain_estimation_method=stain_estimation_method)
    
    # Normalize every patches of each WSI
    normalize_patches_from_WSI(WT+G12C+others, normalizer)
    
