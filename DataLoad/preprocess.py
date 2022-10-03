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
            plt.imshow(im)
            plt.show()
            im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # convert from BGR to RGB
            plt.imshow(im)
            plt.show()

            for shape in data['shapes']:

                # read labels and bounding box coordinates
                label = shape['label']
                points = shape['points']
                xy_min, xy_max = points
                x_min, y_min = xy_min
                x_max, y_max = xy_max
                
                if(x_min > x_max):
                    t=x_min
                    x_min=x_max
                    x_max=t

                if(y_min > y_max):
                    t=y_min
                    y_min=y_max
                    y_max=t
                    
                template = im[int(y_min):int(y_max), int(x_min):int(x_max)]
                h, w = template.shape[0], template.shape[1]
                print('using the following legend feature for matching...:')
                plt.imshow(template)
                plt.show()
                
                print('detecting for label:', label)
                typ=label.split('_')[-1]
                print('type:', typ)
                
                
                ## To match point shapes
                if typ=='pt':
                    
                    # find all the template matches in the basemap
                    res = cv2.matchTemplate(im, template,cv2.TM_CCOEFF_NORMED)
                    threshold = 0.55
                    loc = np.where( res >= threshold)
                    
                    # use the bounding boxes to create prediction binary raster
                    pred_binary_raster=np.zeros((im.shape[0], im.shape[1]))
                    for pt in zip(*loc[::-1]):
                        print('match found:')
                        pred_binary_raster[int(pt[1]+float(h)/2), pt[0] + int(float(w)/2)]=1
                        plt.imshow(im[pt[1]:pt[1] + h, pt[0]:pt[0] + w])
                        plt.show()

                
                ## To match lines and polygons
                else:
                    
                    if typ=='line':
                        # do edge detection
                        print('detecting lines in the legend feature...')
                        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                        edges = cv2.Canny(gray, threshold1=30, threshold2=100)
                        plt.imshow(edges)
                        plt.show()
                        central_pixel=tuple(np.argwhere(edges==255)[0])
                        sought = template[central_pixel].tolist()
                    else: # type=='poly'
                        # take the median of the colors to find the predominant color
                        r=int(np.median(template[:,:,0]))
                        g=int(np.median(template[:,:,1]))
                        b=int(np.median(template[:,:,2]))
                        sought=[r, g, b]
                    
                    print('matching the color:', sought)
                    
                    # capture the variations of legend color due to scanning errors
                    color_range=20
                    lower = np.array([x - color_range for x in sought], dtype="uint8")
                    upper = np.array([x + color_range for x in sought], dtype="uint8")
                    
                    # create a mask to only preserve current legend color in the basemap
                    mask = cv2.inRange(im, lower, upper)
                    detected = cv2.bitwise_and(im, im, mask=mask)
                    
                    # convert to grayscale 
                    detected_gray = cv2.cvtColor(detected, cv2.COLOR_BGR2GRAY)
                    img_bw = cv2.threshold(detected_gray, 127, 255, cv2.THRESH_BINARY)[1]
                    
                    # convert the grayscale image to binary image
                    pred_binary_raster = img_bw.astype(float) / 255
                
                # print
                print('predicted binary raster:')
                print('shape:', pred_binary_raster.shape)
                print('unique value(s):', np.unique(pred_binary_raster))

                # plot the raster and save it
                plt.imshow(pred_binary_raster)
                plt.show()
                
                # save the raster into a .tif file
                out_file_path=os.path.join('results', filename+'_'+label+'.tif')
                pred_binary_raster=pred_binary_raster.astype('uint16')
                cv2.imwrite(out_file_path, pred_binary_raster)
                
                # convert the image to a binary raster .tif
                raster = rasterio.open(out_file_path)
                transform = raster.transform
                array     = raster.read(1)
                crs       = raster.crs 
                width     = raster.width 
                height    = raster.height 
                
                raster.close()
                
                with rasterio.open(out_file_path, 'w', 
                                driver    = 'GTIFF', 
                                transform = transform, 
                                dtype     = rasterio.uint8, 
                                count     = 1, 
                                compress  = 'lzw', 
                                crs       = crs, 
                                width     = width, 
                                height    = height) as dst:
                    
                    dst.write(array, indexes=1)
                    dst.close()
    # Test; load and plot the produced rasters
    for file_name in os.listdir('results'):
        file_path = os.path.join('results', file_name)
        print('file_path:', file_path)
        im=cv2.imread(file_path)
        print(np.unique(im), im.shape)
        im[np.where(im==1)]=255
        plt.imshow(im)
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='map_feature_extraction')
    # load data from file
    parser.add_argument('--data_path', type=str, default='test_data2', help='directory should contain JSON and Map Image .tif file')
    args = parser.parse_args()
    process(args.data_path)

if __name__ == "__main__":
    main()
