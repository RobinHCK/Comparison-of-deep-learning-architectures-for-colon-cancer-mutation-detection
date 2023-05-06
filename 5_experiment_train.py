import os

folds = [1,2,3,4,5]

for fold in folds:
    command = "python train.py " + str(fold)
    os.system(command)

    print("Run command " + command)
