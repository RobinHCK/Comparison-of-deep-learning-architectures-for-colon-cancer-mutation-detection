from config import *
import os
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from matplotlib.lines import Line2D





def create_graphs_networks():
    """Create & save graphs which show the networks training.
    
    Create 5 graphs, 1 per fold (plot chart) which show the evolution of the training and validation loss through the epochs.
    Create 5 graphs, 1 per fold (plot chart) which show the evolution of the training and validation accuracy through the epochs.

    Args:
        None.

    Returns:
        None.
        
    """
    # Create graph showing Training and Validation loss
    graphs_loss()
    # Create graph showing Training and Validation accuracy
    graphs_accuracy()



def create_graphs_experiment():
    """Create & save graphs showing classification results made by the networks. 
    
    Create 6 graphs (heatmap chart) which show the confusion matrix.
    Create 5 graphs (heatmap chart) which show the classification report.
    Create 6 graphs (pie chart) which show the percentage of well and misclassified patches.
    Create 1 graph (bar chart) which shows the number of well and misclassified patches per WSI, sorted by class (WT, G12C, others).
    
    Args:
        None.

    Returns:
        None.
        
    """
    # Create graph confusion matrix
    graphs_confusion_matrix()
    # Create graph classification report
    create_classification_report()
    
    # Create pie graph: percentage of patches correctly predicted per class
    graphs_pie_percentage_of_patches_well_classified()
    # Create bar graph: number of well classified patches per WSI per class
    graph_bar_number_of_patches_well_classified_per_wsi()
    


def graphs_loss():
    """Create graphs showing Training and Validation loss through epochs.
    
    Create 5 graphs (1 per fold) thanks to the networks history files generated while training.
    The graphs are saved as .pdf file to enhance their quality.

    Args:
        None.

    Returns:
        None.
        
    """
    for fold_id in range(1,len(Folds)+1):
        with open(("results/history_K" + str(fold_id) + ".pkl"), 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = "latin1"
            history = u.load()
    
            loss_train = history['loss']
            loss_val = history['val_loss']
            epochs = range(1,len(loss_train)+1)
            plt.plot(epochs, loss_train, 'g', label='Training loss')
            plt.plot(epochs, loss_val, 'b', label='validation loss')
            plt.title('Training and Validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('results/train_val_loss_K' + str(fold_id) + '.pdf')
            plt.clf()
            
            f.close()
    
    
    
def graphs_accuracy():
    """Create graphs showing Training and Validation accuracy through epochs.
    
    Create 5 graphs (1 per fold) thanks to the networks history files generated while training.
    The graphs are saved as .pdf file to enhance their quality.

    Args:
        None.

    Returns:
        None.
        
    """
    for fold_id in range(1,len(Folds)+1):
        with open(("results/history_K" + str(fold_id) + ".pkl"), 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = "latin1"
            history = u.load()
    
            acc_train = history['accuracy']
            acc_val = history['val_accuracy']
            epochs = range(1,len(acc_train)+1)
            plt.plot(epochs, acc_train, 'g', label='Training accuracy')
            plt.plot(epochs, acc_val, 'b', label='validation accuracy')
            plt.title('Training and Validation accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.ylim([0, 1])
            plt.legend()
            plt.savefig('results/train_val_acc_K' + str(fold_id) + '.pdf')
            plt.clf()
    
            f.close()
        
        
        
def graphs_confusion_matrix():
    """Create graph confusion matrix.
    
    Create 6 graphs:
        - 1 per fold = 5
        - 1 is made from the confusion matrice of each fold combined
    
    The graphs are saved as .pdf file to enhance their quality.
    The confusion matrix are also saved as a pickle file.

    Args:
        None.

    Returns:
        None.
        
    """
    total_conf_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    
    for fold_id in range(1,len(Folds)+1):
        # Retrieve informations to build the confusion matrix
        tested_patch_names = sorted(os.listdir(dataset_path + "patches_organized_per_split/fold_" + str(fold_id) + "/Test/WT_G12C_others/"))

        tested_patch_classes = []
        for patch_name in tested_patch_names:
            if patch_name.split(' ')[0] in G12C:
                tested_patch_classes.append(0)
            elif patch_name.split(' ')[0] in others:
                tested_patch_classes.append(1)
            elif patch_name.split(' ')[0] in WT:
                tested_patch_classes.append(2)

        np.array(tested_patch_classes)

        predicted_classes = np.array(pickle.load((open("results/predictions_K" + str(fold_id) + ".pkl", "rb"))))
        
        conf_matrix = confusion_matrix(tested_patch_classes, predicted_classes)
        
        # Save confusion matrix
        pickle.dump(conf_matrix, open("results/conf_matrix_K" + str(fold_id) + ".pkl", 'wb'))
        print("Confusion Matrix fold " + str(fold_id) + " saved")
        
        total_conf_matrix = np.add(total_conf_matrix, conf_matrix)
        
        # Create heatmap
        df_conf_matrix = pd.DataFrame(conf_matrix, index = ["G12C","others","WT"], columns = ["G12C","others","WT"])
        plt.title("Confusion Matrix fold " + str(fold_id))
        sn.heatmap(df_conf_matrix, cmap="Blues", annot=True, fmt='g')
        # Save graph
        plt.savefig("results/conf_matrix_K" + str(fold_id) + ".pdf")
        plt.clf()
            
    pickle.dump(total_conf_matrix, open("results/conf_matrix_K_all.pkl", 'wb'))
    print('Confusion Matrix all fold saved')
    # Create heatmap for all fold (merge the previous 5 confusion matrix)
    df_conf_matrix = pd.DataFrame(total_conf_matrix, index = ["G12C","others","WT"], columns = ["G12C","others","WT"])
    plt.title("Confusion Matrix all fold")
    sn.heatmap(df_conf_matrix, cmap="Blues", annot=True, fmt='g')
    # Save graph
    plt.savefig("results/conf_matrix_K_all.pdf")
    plt.clf()
    
              
    
def create_classification_report():
    """Create & Save the classification report.
    
    The classification report is saved as a pickle file.

    Args:
        None.

    Returns:
        None.
        
    """
    for fold_id in range(1,len(Folds)+1):
        # Retrieve informations to build the classification report
        tested_patch_names = sorted(os.listdir(dataset_path + "patches_organized_per_split/fold_" + str(fold_id) + "/Test/WT_G12C_others/"))
        
        tested_patch_classes = []
        for patch_name in tested_patch_names:
            if patch_name.split(' ')[0] in G12C:
                tested_patch_classes.append(0)
            elif patch_name.split(' ')[0] in others:
                tested_patch_classes.append(1)
            elif patch_name.split(' ')[0] in WT:
                tested_patch_classes.append(2)

        np.array(tested_patch_classes)

        predicted_classes = np.array(pickle.load((open("results/predictions_K" + str(fold_id) + ".pkl", "rb"))))
        
        classif_report = classification_report(tested_patch_classes, predicted_classes, target_names=["G12C","others","WT"], output_dict=True)
        
        pickle.dump(classif_report, open("results/classif_report_K" + str(fold_id) + ".pkl", 'wb'))
        print("Classification Report fold " + str(fold_id) + " saved")



def graphs_pie_percentage_of_patches_well_classified():
    """Create 6 graphs (pie chart) showing the percentage of well and misclassified patches.
    
    Create 5 graphs which show the percentage of patches well and misclassified per fold per class (WT, G12C, others).
    Create 1 graph which shows the percentage of patches well and misclassified per class (on the whole dataset).
    Each class has a color code : well classified '#268026' ; misclassified '#d92626'.
    The graph is saved as .pdf file to enhance his quality.
    Args:
        None.
    Returns:
        None.
        
    """
    for fold_id in range(1,len(Folds)+1):
        conf_matrix = pickle.load((open("results/conf_matrix_K" + str(fold_id) + ".pkl", "rb")))
        nbr_patches_correct = conf_matrix[0][0] + conf_matrix[1][1] + conf_matrix[2][2]
        nbr_patches_wrong = conf_matrix[0][1] + conf_matrix[0][2] + conf_matrix[1][0] + conf_matrix[1][2] + conf_matrix[2][0] + conf_matrix[2][1]
                   
        labels = 'Well classified ('+str(nbr_patches_correct)+')','Misclassified ('+str(nbr_patches_wrong)+')'
        sizes = [nbr_patches_correct,nbr_patches_wrong]
        
        fig, ax = plt.subplots()
        size = 0.3
        outer_colors = ['#268026','#d92626']
        inner_colors = ['#268026']*3 + ['#d92626']*3
        ax.pie([nbr_patches_correct, nbr_patches_wrong], radius=1, pctdistance=0.85, colors=outer_colors, labels=['Well classified','Misclassified'], autopct='%1.1f%%', wedgeprops=dict(width=size, edgecolor='w'))
        ax.pie([conf_matrix[0][0],conf_matrix[1][1],conf_matrix[2][2],conf_matrix[0][1]+conf_matrix[0][2],conf_matrix[1][0]+conf_matrix[1][2],conf_matrix[2][0]+conf_matrix[2][1]], radius=1-size, pctdistance=0.75, colors=inner_colors, autopct='%1.1f%%', wedgeprops=dict(width=size, edgecolor='w'))
        ax.set(aspect="equal", title='Percentage of well classified patches')
        label_well_classified = "G12C " + str(round(100*conf_matrix[0][0]/(nbr_patches_correct+nbr_patches_wrong),2)) + "%\nothers " + str(round(100*conf_matrix[1][1]/(nbr_patches_correct+nbr_patches_wrong),2)) + "%\nWT " + str(round(100*conf_matrix[2][2]/(nbr_patches_correct+nbr_patches_wrong),2)) + "%"
        label_misclassified = "G12C " + str(round(100*(conf_matrix[0][1]+conf_matrix[0][2])/(nbr_patches_correct+nbr_patches_wrong),2)) + "%\nothers " + str(round(100*(conf_matrix[1][0]+conf_matrix[1][2])/(nbr_patches_correct+nbr_patches_wrong),2)) + "%\nWT " + str(round(100*(conf_matrix[2][0]+conf_matrix[2][1])/(nbr_patches_correct+nbr_patches_wrong),2)) + "%"
        legend_elements = [Line2D([0], [0], color='#268026', lw=4, label=label_well_classified), Line2D([0], [0], color='#d92626', lw=4, label=label_misclassified)]
        ax.legend(handles=legend_elements, bbox_to_anchor=(0.9, 0.9))
            
        plt.savefig("results/percentage_of_patches_well_classified_K" + str(fold_id) + ".pdf")
        plt.clf()
    
    conf_matrix_all_fold = pickle.load((open("results/conf_matrix_K_all.pkl", "rb")))
    nbr_patches_correct = conf_matrix_all_fold[0][0] + conf_matrix_all_fold[1][1] + conf_matrix_all_fold[2][2]
    nbr_patches_wrong = conf_matrix_all_fold[0][1] + conf_matrix_all_fold[0][2] + conf_matrix_all_fold[1][0] + conf_matrix_all_fold[1][2] + conf_matrix_all_fold[2][0] + conf_matrix_all_fold[2][1]
               
    labels = 'Well classified ('+str(nbr_patches_correct)+')','Misclassified ('+str(nbr_patches_wrong)+')'
    sizes = [nbr_patches_correct,nbr_patches_wrong]
    
    fig, ax = plt.subplots()
    size = 0.3
    outer_colors = ['#268026','#d92626']
    inner_colors = ['#268026']*3 + ['#d92626']*3
    ax.pie([nbr_patches_correct, nbr_patches_wrong], radius=1, pctdistance=0.85, colors=outer_colors, labels=['Well classified','Misclassified'], autopct='%1.1f%%', wedgeprops=dict(width=size, edgecolor='w'))
    ax.pie([conf_matrix_all_fold[0][0],conf_matrix_all_fold[1][1],conf_matrix_all_fold[2][2],conf_matrix_all_fold[0][1]+conf_matrix_all_fold[0][2],conf_matrix_all_fold[1][0]+conf_matrix_all_fold[1][2],conf_matrix_all_fold[2][0]+conf_matrix_all_fold[2][1]], radius=1-size, pctdistance=0.75, colors=inner_colors, autopct='%1.1f%%', wedgeprops=dict(width=size, edgecolor='w'))
    ax.set(aspect="equal", title='Percentage of well classified patches')
    label_well_classified = "G12C " + str(round(100*conf_matrix_all_fold[0][0]/(nbr_patches_correct+nbr_patches_wrong),2)) + "%\nothers " + str(round(100*conf_matrix_all_fold[1][1]/(nbr_patches_correct+nbr_patches_wrong),2)) + "%\nWT " + str(round(100*conf_matrix_all_fold[2][2]/(nbr_patches_correct+nbr_patches_wrong),2)) + "%"
    label_misclassified = "G12C " + str(round(100*(conf_matrix_all_fold[0][1]+conf_matrix_all_fold[0][2])/(nbr_patches_correct+nbr_patches_wrong),2)) + "%\nothers " + str(round(100*(conf_matrix_all_fold[1][0]+conf_matrix_all_fold[1][2])/(nbr_patches_correct+nbr_patches_wrong),2)) + "%\nWT " + str(round(100*(conf_matrix_all_fold[2][0]+conf_matrix_all_fold[2][1])/(nbr_patches_correct+nbr_patches_wrong),2)) + "%"
    legend_elements = [Line2D([0], [0], color='#268026', lw=4, label=label_well_classified), Line2D([0], [0], color='#d92626', lw=4, label=label_misclassified)]
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.9, 0.9))
        
    plt.savefig('results/percentage_of_patches_well_classified_K_all.pdf')
    plt.clf()
    
    
    
def graph_bar_number_of_patches_well_classified_per_wsi():
    """Create 1 graph (bar chart) which shows the number of well and misclassified patches per WSI, sorted by class (WT, G12C, others).
    
    Each class has a color code :
        - Well classified WT '#7aa6c2' 
        - Well classified G12C '#58a066'
        - Well classified others '#e6ba95'
        - Misclassified WT '#496374' 
        - Misclassified G12C '#34603d' 
        - Misclassified others '#8a6f59' 
    The graph is saved as .pdf file to enhance his quality.
    
    Args:
        None.
    Returns:
        None.
        
    """
    nbr_patches_well_classified_per_wsi_WT = []
    nbr_patches_misclassified_per_wsi_WT = []
    nbr_patches_well_classified_per_wsi_G12C = []
    nbr_patches_misclassified_per_wsi_G12C = []
    nbr_patches_well_classified_per_wsi_others = []
    nbr_patches_misclassified_per_wsi_others = []
    
    for wsi_id in WT + G12C + others:
        # Find fold_id of the wsi
        fold_id = [i+1 for i in range(5) if wsi_id in Folds[i]][0]

        nbr_patches_well_classified = 0
        nbr_patches_misclassified = 0
        
        # Retrieve informations
        tested_patch_names = sorted(os.listdir(dataset_path + "patches_organized_per_split/fold_" + str(fold_id) + "/Test/WT_G12C_others/"))

        tested_patch_classes = []
        for patch_name in tested_patch_names:
            if patch_name.split(' ')[0] in G12C:
                tested_patch_classes.append(0)
            elif patch_name.split(' ')[0] in others:
                tested_patch_classes.append(1)
            elif patch_name.split(' ')[0] in WT:
                tested_patch_classes.append(2)

        np.array(tested_patch_classes)

        predicted_classes = np.array(pickle.load((open("results/predictions_K" + str(fold_id) + ".pkl", "rb"))))
        
        # Find number of patches well and misclassified 
        for patch_name, true_pred, pred in zip(tested_patch_names, tested_patch_classes, predicted_classes):
            if wsi_id in patch_name:
                if true_pred == pred:
                    nbr_patches_well_classified += 1
                else:
                    nbr_patches_misclassified += 1
                    
        if wsi_id in WT:
            nbr_patches_well_classified_per_wsi_WT.append(nbr_patches_well_classified)
            nbr_patches_misclassified_per_wsi_WT.append(nbr_patches_misclassified)
        elif wsi_id in G12C:
            nbr_patches_well_classified_per_wsi_G12C.append(nbr_patches_well_classified)
            nbr_patches_misclassified_per_wsi_G12C.append(nbr_patches_misclassified)
        elif wsi_id in others:
            nbr_patches_well_classified_per_wsi_others.append(nbr_patches_well_classified)
            nbr_patches_misclassified_per_wsi_others.append(nbr_patches_misclassified)
                    
    # Sort the bars per their size 
    nbr_patches_misclassified_per_wsi_WT, nbr_patches_well_classified_per_wsi_WT, WT_sorted = zip(*sorted(zip(nbr_patches_misclassified_per_wsi_WT, nbr_patches_well_classified_per_wsi_WT, WT), reverse=True))
    nbr_patches_misclassified_per_wsi_G12C, nbr_patches_well_classified_per_wsi_G12C, G12C_sorted = zip(*sorted(zip(nbr_patches_misclassified_per_wsi_G12C, nbr_patches_well_classified_per_wsi_G12C, G12C), reverse=True))
    nbr_patches_misclassified_per_wsi_others, nbr_patches_well_classified_per_wsi_others, others_sorted = zip(*sorted(zip(nbr_patches_misclassified_per_wsi_others, nbr_patches_well_classified_per_wsi_others, others), reverse=True))
    
    x = np.arange(len(WT_sorted) + len(G12C_sorted) + len(others_sorted))
    width = 0.75
    # Graph
    fig, ax = plt.subplots()
    rects = ax.bar(x, nbr_patches_well_classified_per_wsi_WT + nbr_patches_well_classified_per_wsi_G12C + nbr_patches_well_classified_per_wsi_others, width, bottom = nbr_patches_misclassified_per_wsi_WT + nbr_patches_misclassified_per_wsi_G12C + nbr_patches_misclassified_per_wsi_others)
    [rects[i].set_color('#7aa6c2') for i in range(len(WT_sorted))]
    [rects[i].set_color('#58a066') for i in range(len(WT_sorted),len(WT_sorted)+len(G12C_sorted))]
    [rects[i].set_color('#e6ba95') for i in range(len(WT_sorted)+len(G12C_sorted),len(WT_sorted)+len(G12C_sorted)+len(others_sorted))]
    
    rects = ax.bar(x, nbr_patches_misclassified_per_wsi_WT + nbr_patches_misclassified_per_wsi_G12C + nbr_patches_misclassified_per_wsi_others, width)
    [rects[i].set_color('#496474') for i in range(len(WT_sorted))]
    [rects[i].set_color('#35603d') for i in range(len(WT_sorted),len(WT_sorted)+len(G12C_sorted))]
    [rects[i].set_color('#8a6f59') for i in range(len(WT_sorted)+len(G12C_sorted),len(WT_sorted)+len(G12C_sorted)+len(others_sorted))]
    
    ax.set_xlabel('WSI')
    ax.set_ylabel('Number of patches')
    ax.set_title('Number of patches well and misclassified per WSI')
    ax.set_xticks(x)
    ax.set_xticklabels(WT_sorted + G12C_sorted + others_sorted, rotation=90)
    legend_elements = [Line2D([0], [0], color='#7aa6c2', lw=4, label='WT well classified'), Line2D([0], [0], color='#496474', lw=4, label='WT misclassified'), Line2D([0], [0], color='#58a066', lw=4, label='G12C well classified'), Line2D([0], [0], color='#35603d', lw=4, label='G12C misclassified'), Line2D([0], [0], color='#e6ba95', lw=4, label='others well classified'), Line2D([0], [0], color='#8a6f59', lw=4, label='others misclassified')]
    ax.legend(handles=legend_elements)
    plt.grid(axis='y')  
    fig.tight_layout()
    
    plt.savefig('results/number_of_patches_well_classified_per_wsi.pdf')
    plt.clf()
    
    
    
def graph_patient_diagnosis():
    """Create graph confusion matrix for patient diagnosis.
    For each patient (1 WSI = 1 patient) we affect the most predicted class (WT or G12C or others) for each patches from the WSI corresponding to the patient.
    
    The graph is saved as .pdf file to enhance his quality.
    The confusion matrix is also saved as a pickle file.
    Args:
        None.
    Returns:
        None.
        
    """
    conf_matrix_3_classes = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    nbr_patches_well_classified_per_wsi_WT = []
    nbr_patches_misclassified_per_wsi_WT = []
    nbr_patches_well_classified_per_wsi_G12C = []
    nbr_patches_misclassified_per_wsi_G12C = []
    nbr_patches_well_classified_per_wsi_others = []
    nbr_patches_misclassified_per_wsi_others = []
    
    for wsi_id in WT + G12C + others:
        
        # Find fold_id of the wsi
        fold_id = [i+1 for i in range(5) if wsi_id in Folds[i]][0]

        nbr_patches_well_classified = 0
        nbr_patches_misclassified = 0
        
        # Retrieve patches name
        tested_patches_name = sorted(os.listdir(dataset_path + "patches_organized_per_split/fold_" + str(fold_id) + "/Test/WT_G12C_others/"))

        # Retrieve predictions
        predicted_classes = np.array(pickle.load((open("results/predictions_K" + str(fold_id) + ".pkl", "rb"))))
        
        # Ground truth
        if wsi_id in G12C:
            true_pred = 0
        elif wsi_id in others:
            true_pred = 1
        elif wsi_id in WT:
            true_pred = 2

        # Find number of patches well and misclassified 
        pred_per_class = [0, 0, 0] # G12C, others, WT
        for patch_name, pred in zip(tested_patches_name, predicted_classes):
            if wsi_id in patch_name:
                pred_per_class[pred] += 1

        diagnosis = np.argmax(pred_per_class)

        conf_matrix_3_classes[true_pred][diagnosis] += 1

    conf_matrix_2_classes = np.array([[conf_matrix_3_classes[0][0], conf_matrix_3_classes[0][1]+conf_matrix_3_classes[0][2]], [conf_matrix_3_classes[1][0]+conf_matrix_3_classes[2][0], conf_matrix_3_classes[1][1]+conf_matrix_3_classes[1][2]+conf_matrix_3_classes[2][1]+conf_matrix_3_classes[2][2]]])

    # Saved raw data
    pickle.dump(conf_matrix_3_classes, open("results/conf_matrix_patient_diagnosis_G12C_others_WT.pkl", 'wb'))
    pickle.dump(conf_matrix_2_classes, open("results/conf_matrix_patient_diagnosis_G12C_NotG12C.pkl", 'wb'))
    print('Confusion Matrix Patient Diagnosis saved')

    stats = "Accuracy_Patients_G12C;" + str(conf_matrix_3_classes[0][0]/(conf_matrix_3_classes[0][0]+conf_matrix_3_classes[0][1]+conf_matrix_3_classes[0][2])) + "\n"
    stats += "Accuracy_Patients_others;" + str(conf_matrix_3_classes[1][1]/(conf_matrix_3_classes[1][0]+conf_matrix_3_classes[1][1]+conf_matrix_3_classes[1][2])) + "\n"
    stats += "Accuracy_Patients_WT;" + str(conf_matrix_3_classes[2][2]/(conf_matrix_3_classes[2][0]+conf_matrix_3_classes[2][1]+conf_matrix_3_classes[2][2])) + "\n"
    stats += "Accuracy_Patients_All;" + str((conf_matrix_3_classes[0][0]+conf_matrix_3_classes[1][1]+conf_matrix_3_classes[2][2])/(conf_matrix_3_classes[0][0]+conf_matrix_3_classes[0][1]+conf_matrix_3_classes[0][2]+conf_matrix_3_classes[1][0]+conf_matrix_3_classes[1][1]+conf_matrix_3_classes[1][2]+conf_matrix_3_classes[2][0]+conf_matrix_3_classes[2][1]+conf_matrix_3_classes[2][2]))
    open("results/patient_diagnosis_G12C_others_WT.csv", "w").write(stats)

    stats = "Accuracy_Patients_G12C;" + str(conf_matrix_2_classes[0][0]/(conf_matrix_2_classes[0][0]+conf_matrix_2_classes[0][1])) + "\n"
    stats += "Accuracy_Patients_NotG12C;" + str(conf_matrix_2_classes[1][1]/(conf_matrix_2_classes[1][0]+conf_matrix_2_classes[1][1])) + "\n"
    stats += "Accuracy_Patients_All;" + str((conf_matrix_2_classes[0][0]+conf_matrix_2_classes[1][1])/(conf_matrix_2_classes[0][0]+conf_matrix_2_classes[0][1]+conf_matrix_2_classes[1][0]+conf_matrix_2_classes[1][1]))
    open("results/patient_diagnosis_G12C_NotG12C.csv", "w").write(stats)
    
    # Draw confusion matrix
    # Create heatmap for G12C, others & WT
    df_conf_matrix_3_classes = pd.DataFrame(conf_matrix_3_classes, index = ["G12C","others","WT"], columns = ["G12C","others","WT"])
    plt.title("Confusion Matrix Patient Diagnosis")
    sn.heatmap(df_conf_matrix_3_classes, cmap="Blues", annot=True, fmt='g')
    # Save graph
    plt.savefig("results/conf_matrix_patient_diagnosis_G12C_others_WT.pdf")
    plt.clf()

    # Create heatmap for G12C, NotG12C
    df_conf_matrix_2_classes = pd.DataFrame(conf_matrix_2_classes, index = ["G12C","NotG12C"], columns = ["G12C","NotG12C"])
    plt.title("Confusion Matrix Patient Diagnosis")
    sn.heatmap(df_conf_matrix_2_classes, cmap="Blues", annot=True, fmt='g')
    # Save graph
    plt.savefig("results/conf_matrix_patient_diagnosis_G12C_NotG12C.pdf")
    plt.clf()
    
    
    
    
    
if __name__ == "__main__":
    
    # Create & Save graphs belonging to the network training
    create_graphs_networks()
    
    # Create & Save graphs belonging to the experiment
    create_graphs_experiment()

    # Create & Save data belonging to patient diagnosis
    graph_patient_diagnosis()
