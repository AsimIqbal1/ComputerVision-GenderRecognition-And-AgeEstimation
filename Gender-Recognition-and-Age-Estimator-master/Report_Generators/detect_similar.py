import os
from PIL import Image
from PIL import ImageChops
from PIL import ImageStat
import cv2

rootdir = "F:\\VS Code\\Projects\\Gender-Recognition-and-Age-Estimator-master\\fold"

def get_filepaths(dir):
    file_paths = []
    for folder, subs, files in os.walk(rootdir):
        for filename in files:
            file_paths.append(os.path.abspath(os.path.join(folder, filename)))
    return file_paths

def print_dir_files(paths):
    print("Printing Files: =========================== :")
    for filename in paths:
        print(str(filename))




diff_img_file = "file.png"

def diff(im1_file, im2_file, delete_diff_file=False, diff_img_file=diff_img_file):
    im1 = Image.open(im1_file).convert('LA')
    im2 = Image.open(im2_file).convert('LA')

    diff_img = ImageChops.difference(im1,im2)
    diff_img.convert('RGB').save(diff_img_file)

    stat = ImageStat.Stat(diff_img)

    # can be [r,g,b] or [r,g,b,a]
    sum_channel_values = sum(stat.mean)
    max_all_channels = len(stat.mean) * 100

    diff_ratio = sum_channel_values/max_all_channels

    if delete_diff_file:
        remove(diff_img_file)
    
    return diff_ratio

def delete_file(path):
    os.remove(path)

def remove_similars(file_paths):
    length = len(file_paths)
    i=0
    j=0
    while i<length:
        j=i+1
        while j<length:
            difference = diff(file_paths[i], file_paths[j])
            print("i: "+str(i)+", j: "+str(j)+" = "+str(difference))
            if(difference<0.3):
                delete_file(file_paths[j])
                del file_paths[j]
                length = length - 1
                print(length)

            j+= 1
        
        i+= 1
    # for i in range(0,length-1,1):
    #     for j in range(i+1,length-1, 1):
    #         difference = diff(file_paths[i], file_paths[j])
    #         print("i: "+str(i)+", j: "+str(j)+" = "+str(difference))
    #         if(difference<0.3):
    #             delete_file(file_paths[j])
    #             del file_paths[j]
    #             length = length - 1
        
    return file_paths

if __name__ == '__main__':
    if os.path.exists(rootdir):

        file_paths = get_filepaths(rootdir)
        print_dir_files(file_paths)
        file_paths = remove_similars(file_paths)

        print_dir_files(file_paths)
    else:
        print("Path does not exists!")
