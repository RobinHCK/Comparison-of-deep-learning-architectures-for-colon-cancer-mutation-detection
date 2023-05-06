import os
from config import *

# Run explainers on patches
for explainer in explainers:
    for mutation in mutation_to_analyze:
        for wsi in wsi_to_explain:
            command = "python explain_patches_from_a_WSI_with_" + explainer + ".py " + mutation + " " + wsi + " " + str(number_of_patches_per_wsi_to_explain)
            os.system(command)
            
            print("Run command " + command)



# Run explainers on QuPatches
"""
for explainer in explainers:
    for mutation in mutation_to_analyze:
        for wsi in wsi_to_explain:
            command = "python explain_QuPathPatches_from_a_WSI.py " + explainer + " " + mutation + " " + wsi
            os.system(command)
            
            print("Run command " + command)
"""


# Create animations with CAM explainer on QuPath patches
"""
for mutation in mutation_to_analyze:
    for wsi in ["A18000005"]:
        command = "python explain_QuPathPatches_from_a_WSI.py CAM " + mutation + " " + wsi + " all"
        os.system(command)
        
        print("Run command " + command)
"""
