from operator import gt
import os
import shutil
import argparse
import re


def copy(names='train.txt',data_dir='/data2/mineral_competition/data/training',
out_path="trainImages",copy_path='/data2/mineral_competition/data/training/',
labels_path="gt_labels",json_path="trainImages"):
    names = set(open(names,"r"))
    files = os.listdir(data_dir)
    images = []
    json_file = []
    gt_labels = []

    for each in names:
        images.append(each.strip())
        json_file.append(each.strip().replace('.tif', '.json'))
        gt_labels.append(each.strip().replace('.tif', '_.*'))
        
    # for each in files:
    #     if each in images:
    #         shutil.copy(os.path.join(copy_path, each),out_path)
    #     if each in json_file:    
    #         shutil.copy(os.path.join(copy_path, each),json_path)

    for each in gt_labels:
        x = re.compile(each)
        newlist = list(filter(x.match, files)) 
        for each in newlist:
            shutil.copy(os.path.join(copy_path, each),labels_path)

def separate():
    names = set(open(names,"r"))
def main():
    parser = argparse.ArgumentParser(description='preprocessing functions')
    # load data from file
    parser.add_argument('--function', type=str, default='copy_data', help='specify which function to call')
    args = parser.parse_args()
    if(args.function == 'copy_data'):
        copy()
    if(args.function == 'segregate'):
        separate()

if __name__ == "__main__":
    main()
