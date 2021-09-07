

import os
import numpy as np
import nibabel as nib
from glob import glob
from skimage import io
from skimage.transform import resize
# from nibabel.testing import data_path


# example_filename = 'image.nii.gz'

# img = nib.load(example_filename)

# AFFINE = img.affine
# hdr = img.header
# raw = hdr.structarr


#array_img = nib.Nifti1Image(array_data, affine, hdr)

# nib.save(array_img, 'test4d.nii.gz')  

path_phase = 'Validation' # perghaps we need to change this for test phase
path_4save = 'Results'
path_input_list = ['./validation_data_qubiq2021_QC/brain-growth',
                   './validation_data_qubiq2021_QC/kidney',
                   './validation_data_qubiq2021_QC/brain-tumor',
                   './validation_data_qubiq2021_QC/brain-tumor',
                   './validation_data_qubiq2021_QC/brain-tumor',
                   './validation_data_qubiq2021_QC/pancreas',
                   './validation_data_qubiq2021_QC/pancreatic-lesion',
                   './validation_data_qubiq2021_QC/prostate',
                   './validation_data_qubiq2021_QC/prostate']
                   
Regions_list = ['brain-growth',
                   'kidney',
                   'brain-tumor',
                   'brain-tumor',
                   'brain-tumor',
                   'pancreas',
                   'pancreatic-lesion',
                   'prostate',
                   'prostate']
                                     
path_results_list = ['./Results_mina/task01', 
                     './Results_mina/task02',
                     './Results_mina/task03',
                     './Results_mina/task04',
                     './Results_mina/task05',
                     './Results_mina/task06',
                     './Results_mina/task07',
                     './Results_mina/task08',
                     './Results_mina/task09']
                  
names_tasks_list= ['task01', 
                     'task01',
                     'task01',
                     'task02',
                     'task03',
                     'task01',
                     'task01',
                     'task01',
                     'task02']
   

if os.path.isdir(path_4save) == False:
    os.makedirs(path_4save)

for i in range(len(path_input_list)): #-2
    
    path_input = path_input_list[i] 
    pathresults = path_results_list[i]
    
    pathinput = os.path.join(path_input, path_phase)
    
    lst_subdirs=glob(os.path.join(pathinput,'*'))    
    subdir4save = os.path.join(path_4save, Regions_list[i])
    
    try:
        os.mkdir(subdir4save)
    except:
        s='folder if available'
    
    for path in lst_subdirs:
        print(path)
        spl=path.split('case')
        casenumber=spl[1]
        #print(casenumber)
        subsubdir4save= os.path.join(subdir4save, 'case' + casenumber)
        
        try:
            os.mkdir(subsubdir4save)
        except:
            s='folder if available'
            
            
        filename = 'image.nii.gz'
        img = nib.load(os.path.join(path, filename))
        AFFINE = img.affine
        hdr = img.header        
        
        shape_img = img.shape
        num_slices = np.min(shape_img)
           
        if num_slices ==1 or len(shape_img)==2:
            
            
            filename_result= os.path.join(pathresults,'case' + casenumber + '_slice.png')
                        
            img_res = io.imread(filename_result, as_gray = True)
                
            img_res = resize(img_res, (shape_img[0], shape_img[1]),
                       anti_aliasing=True)
            
            img_res = img_res/np.max(img_res)
            
            array_img = nib.Nifti1Image(img_res, AFFINE, hdr)
            nib.save(array_img, os.path.join(subsubdir4save, names_tasks_list[i]+ '.nii.gz'))   
                
        else:
            dim_slices = np.argmin(shape_img)
            # print(shape_img)
            # print(dim_slices)
            
            array_img_res=np.zeros(shape=shape_img, dtype=np.float16())
            
            for jj in range(num_slices):
                filename_result= os.path.join(pathresults,'case' + casenumber +
                                               '_slice'+ str(jj)+'.png')
                
                img_res = io.imread(filename_result, as_gray = True)
                
                if dim_slices==0:
                    img_res = resize(img_res, (shape_img[1], shape_img[2]),
                       anti_aliasing=True)
                    array_img_res[jj,:,:] = img_res
                elif dim_slices==1:
                    img_res = resize(img_res, (shape_img[0], shape_img[2]),
                       anti_aliasing=True)
                    array_img_res[:,jj,:] = img_res
                elif dim_slices==2:
                    img_res = resize(img_res, (shape_img[0], shape_img[1]),
                       anti_aliasing=True)
                    array_img_res[:,:,jj] = img_res
                
            
            # averaging for brain tumor
            if i==2 or i==3 or i==4:
                array_img_res = np.mean(array_img_res, axis=dim_slices)
                
                
            array_img_res = array_img_res/np.max(array_img_res) 
            
            array_img = nib.Nifti1Image(array_img_res, AFFINE, hdr)
            nib.save(array_img, os.path.join(subsubdir4save, names_tasks_list[i]+ '.nii.gz'))   
            

