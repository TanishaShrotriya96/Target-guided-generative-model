import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import requests
import shutil
import os
import rasterio
from tqdm import tqdm
import argparse
import sys

def process(data_dir):
    # create directory to store results
    os.makedirs('results', exist_ok=True)
    for file_name in tqdm(os.listdir(data_dir)):

        # get the .tif files
        if '.tif' in file_name:
            filename=file_name.replace('.tif', '')
            print('Working on map:', file_name)
            file_path=os.path.join(data_dir, file_name)
            test_json=file_path.replace('.tif', '.json')
            
            # read the legend annotation file
            with open(test_json) as f:
                data = json.load(f)

            # load image into an array
            im=cv2.imread(file_path)
            im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # convert from BGR to RGB
            #plt.imshow(im)
            #plt.show()

            for shape in data['shapes']:

                # read labels and bounding box coordinates
                label = shape['label']
                points = shape['points']
      
                x_min = y_min = sys.float_info.max
                x_max = y_max =0
                for each in points:
                    x_t,y_t= each
                    if(x_min > x_t):
                        x_min = x_t
                        
                    if(y_min > y_t):
                        y_min = y_t
                
                    if(x_max < x_t):
                        x_max = x_t
                        
                    if(y_max < y_t):
                        y_max = y_t
                    
                template = im[int(y_min):int(y_max), int(x_min):int(x_max)]
                h, w = template.shape[0], template.shape[1]
                print('using the following legend feature for matching...:')                
                print('detecting for label:', label)
                typ=label.split('_')[-1]
                print('type:', typ)
            
                ## To match point shapes
                if typ=='pt':
                    
                    # find all the template matches in the basemap
                    res = cv2.matchTemplate(im, template,cv2.TM_CCOEFF_NORMED)
                    threshold = 0.39
                    loc = np.where( res >= threshold)
                    
                    # use the bounding boxes to create prediction binary raster
                    pred_binary_raster=np.zeros((im.shape[0], im.shape[1]))
                    for pt in zip(*loc[::-1]):
                        pred_binary_raster[int(pt[1]+float(h)/2), pt[0] + int(float(w)/2)]=1
                        # plt.imshow(im[pt[1]:pt[1] + h, pt[0]:pt[0] + w])
                        # plt.show()
                                           
                    # print
                    print('predicted binary raster:')
                    print('shape:', pred_binary_raster.shape)
                    print('unique value(s):', np.unique(pred_binary_raster))

                    # plot the raster and save it
                    # plt.imshow(pred_binary_raster)
                    # plt.show()
                    
                    # save the raster into a .tif file
                    out_file_path=os.path.join('results', filename+'_'+label+'.jpeg')
                    pred_binary_raster=pred_binary_raster.astype('uint16')*255
                    cv2.imwrite(out_file_path, pred_binary_raster)

def main():
    parser = argparse.ArgumentParser(description='map_feature_extraction')
    # load data from file
    parser.add_argument('--data_path', type=str, default='ValidationImages', help='directory should contain JSON and Map Image .tif file')
    args = parser.parse_args()
    process(args.data_path)

if __name__ == "__main__":
    main()
