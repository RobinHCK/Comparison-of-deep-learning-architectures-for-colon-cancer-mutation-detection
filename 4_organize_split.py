from config import *
import os
import shutil





def create_dataset_hierarchy(dataset_path):
    """Create the dataset hierarchy.
    
    The organization followed the standard to use flow_from_directory()
    from keras https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
    This function will allow the network to load patches on the flow (imperative in the case of a large dataset).
    unfortunately, flow_from_directory() does not support cross-validation, 
    hence, we create one organization per fold. 
    
    Create hierarchy :
    |
    | --- patches_organized_per_split/
          | --- fold_1
          |     |
          |     | --- Train
          |     |     | --- WT
          |     |           |--- patch_name...
          |     |     | --- G12C
          |     |           |--- patch_name...
          |     |     | --- others
          |     |           |--- patch_name...
          |     | --- Val
          |     |     | --- WT
          |     |           |--- patch_name...
          |     |     | --- G12C
          |     |           |--- patch_name...
          |     |     | --- others
          |     |           |--- patch_name...
          |     | --- Test
          |           | --- WT_G12C_others
          |                 |--- patch_name...
          | --- fold_2 ...
          ...

    Args:
        dataset_path (str): The dataset path to use.

    Returns:
        None.
        
    Raises:
        OSError: Error while creating the folders.

    """
    try:
        if not os.path.exists(dataset_path + "patches_organized_per_split/"):
            os.mkdir(dataset_path + "patches_organized_per_split/")
            
        for fold_id in range(1,len(Folds)+1):
            if not os.path.exists(dataset_path + "patches_organized_per_split/fold_" + str(fold_id) + "/"):
                os.mkdir(dataset_path + "patches_organized_per_split/fold_" + str(fold_id) + "/")
                   
            for split in ['Train','Val','Test']:
                if not os.path.exists(dataset_path + "patches_organized_per_split/fold_" + str(fold_id) + "/" + split + "/"):
                    os.mkdir(dataset_path + "patches_organized_per_split/fold_" + str(fold_id) + "/" + split + "/")
                
                if split == 'Train' or split == 'Val':
                    for class_id in ['WT','G12C','others']:
                        if not os.path.exists(dataset_path + "patches_organized_per_split/fold_" + str(fold_id) + "/" + split + "/" + class_id + "/"):
                            os.mkdir(dataset_path + "patches_organized_per_split/fold_" + str(fold_id) + "/" + split + "/" + class_id + "/")
                if split == 'Test':
                    if not os.path.exists(dataset_path + "patches_organized_per_split/fold_" + str(fold_id) + "/" + split + "/WT_G12C_others/"):
                        os.mkdir(dataset_path + "patches_organized_per_split/fold_" + str(fold_id) + "/" + split + "/WT_G12C_others/")

    except OSError:  
        print ('Creation of the directory failed')
        
    

def organize_patches_per_fold_per_split():
    """Create symbolic links to organize patches in split for each fold.
    
    Browse patches from patches_organized_per_wsi/ and create symbolic links into patches_organized_per_split/.

    Args:
        None.

    Returns:
        ValueError: The WSI does not have a class, see config file to affect a class to every WSI.

    """
    for fold_id in range(1,len(Folds)+1):
        
        # Test
        index = int(fold_id)-1
        for wsi_id in Folds[index]:
            for path_patch in os.listdir(dataset_path + "patches_organized_per_wsi/" + wsi_id):
                
                if wsi_id in WT or wsi_id in G12C or wsi_id in others:
                   shutil.copyfile(dataset_path + 'patches_organized_per_wsi/' + wsi_id + '/' + path_patch, dataset_path + 'patches_organized_per_split/fold_' + str(fold_id) + '/Test/WT_G12C_others/' + path_patch)
                else:
                    raise ValueError("The WSI", wsi_id, "does not have a class, see config file to affect a class to every WSI.")
        
        print ('Fold', fold_id, 'Test split done.')

        # Val
        index = int(fold_id)%5
        for wsi_id in Folds[index]:
            for path_patch in os.listdir(dataset_path + "patches_organized_per_wsi/" + wsi_id):
                
                if wsi_id in WT:
                   shutil.copyfile(dataset_path + 'patches_organized_per_wsi/' + wsi_id + '/' + path_patch, dataset_path + 'patches_organized_per_split/fold_' + str(fold_id) + '/Val/WT/' + path_patch)
                elif wsi_id in G12C:
                   shutil.copyfile(dataset_path + 'patches_organized_per_wsi/' + wsi_id + '/' + path_patch, dataset_path + 'patches_organized_per_split/fold_' + str(fold_id) + '/Val/G12C/' + path_patch)
                elif wsi_id in others:
                   shutil.copyfile(dataset_path + 'patches_organized_per_wsi/' + wsi_id + '/' + path_patch, dataset_path + 'patches_organized_per_split/fold_' + str(fold_id) + '/Val/others/' + path_patch)
                else:
                    raise ValueError("The WSI", wsi_id, "does not have a class, see config file to affect a class to every WSI.")
        
        print ('Fold', fold_id, 'Val split done.')

        # Train
        index = [(int(fold_id)+1)%5,(int(fold_id)+2)%5,(int(fold_id)+3)%5]
        for i in index:
            for wsi_id in Folds[i]:
                for path_patch in os.listdir(dataset_path + "patches_organized_per_wsi/" + wsi_id):
                   
                    if wsi_id in WT:
                       shutil.copyfile(dataset_path + 'patches_organized_per_wsi/' + wsi_id + '/' + path_patch, dataset_path + 'patches_organized_per_split/fold_' + str(fold_id) + '/Train/WT/' + path_patch)
                    elif wsi_id in G12C:
                       shutil.copyfile(dataset_path + 'patches_organized_per_wsi/' + wsi_id + '/' + path_patch, dataset_path + 'patches_organized_per_split/fold_' + str(fold_id) + '/Train/G12C/' + path_patch)
                    elif wsi_id in others:
                       shutil.copyfile(dataset_path + 'patches_organized_per_wsi/' + wsi_id + '/' + path_patch, dataset_path + 'patches_organized_per_split/fold_' + str(fold_id) + '/Train/others/' + path_patch)
                    else:
                        raise ValueError("The WSI", wsi_id, "does not have a class, see config file to affect a class to every WSI.")
                        
        print ('Fold', fold_id, 'Train split done.')
        print ('Fold', fold_id, 'hierarchy successfully created.')
        
        
        
        
        
if __name__ == "__main__":
    
    # Create hierarchy
    create_dataset_hierarchy(dataset_path)
        
    # Create symbolic links to organize patches in split for each fold
    organize_patches_per_fold_per_split()
