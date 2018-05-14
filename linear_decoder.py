import os
import numpy as np
import h5py
from time import time

from dataset import ds_x_activities_y_original


## Import from constants file
data_dir = 'FakeData/' #'temp/'
cell_type_inds_path = 'temp/' # optional, only if you ever want to split cell types
linear_decoder_save_path = 'FakeModels/'
immean = 0.5
total_block_size = 10000

class LinearDecoder():

    def __init__(self):
        self.data = [] 

    def runLD(activity_suffix, LD_directory_suffix, im_width=128, cell_type_list=None, create_decoded_images=True):
        '''
        Inputs:
            activitiy_suffix: string that follows activities you want, for example 'spatialSim' for activitiies in folder 'activities_spatialSim'
            LD_directory_suffix: the suffix for the folder in which to save linear decoded images, aka 'LD_images'
            im_width: the size of one side of the images (code only handles square images for now but this could be easily modified)
            cell_type_list: numpy array of index of specific cell types you want decoded (corresponding to cell_type_inds.npy), for example np.asarray(0,1) if you want to grab all cell types identified as 0 or 1
            create_decoded_images: True if you want to create all training/validation/testing decoded images automatically, false to just create and save linear decoder weights
        Outputs:
            Computes and saves linear decoder weights
            Saves decoded images for training/validation/testing if create_decoded_images is True
        '''

        # -------------------------------------
        # CREATE MODEL SAVE DIRECTORY/LOG FILE
        # -------------------------------------
        
        if cell_type_list is not None: # if using specific cell types
            cell_type_string = str(cell_type_list).replace(' ','').replace('.','').replace('[','').replace(']','') # make cell type list concatenated string
            directory_name = linear_decoder_save_path+'/linear_decoder_'+activity_suffix+'_'+cell_type_string+''
        else:
            directory_name = linear_decoder_save_path+'/linear_decoder_'+activity_suffix+''

        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
            os.makedirs(directory_name+'/model')
            os.makedirs(directory_name+'/logs')
     
        log_file_name = directory_name + '/logs/LD.txt'
        
        with open(log_file_name, "a") as log_file:
            log_file.write('training starting \n')

        # -----------------------------
        # LOAD TRAINING DATA GENERATOR
        # -----------------------------
        data_type='training'
        activities_dir_path = data_dir+data_type+'/activities_'+activity_suffix+'/'
        original_dir_path = data_dir+data_type+'/images/'
        data_generator = ds_x_activities_y_original(batch_size=total_block_size, im_width=128, activities_dir_path=activities_dir_path, original_dir_path=original_dir_path, image_mean=immean, normalize_scale=True, shuffle=False)

        # -----------------
        # PARSE CELL TYPES
        # -----------------

        if cell_type_list is not None: # use specific cell types instead of all the data
            cell_type_inds = np.load(cell_type_inds_path+'/cell_type_inds.npy') 
            which_n = np.in1d(cell_type_inds, cell_type_list) # specify a specific cell type  

        # --------------------------------------- 
        # CALCULATE SAMPLE XTX AND XTY STATISTICS 
        # --------------------------------------- 

        # Loop through all training blocks of data
        stats_created=0
        while True:
            try:

                # Generate activities (X)/ images(Y)
                X, Y, block_num = next(data_generator)
                Y = Y.reshape((-1,im_width*im_width))

                # Prepare activities
                if cell_type_list is not None:
                    X = X[:,which_n]
                X = np.insert(X, 0, 1.0, axis=1) # insert column of bias terms

                # Add this blocks statistics to overall ones
                if stats_created == 0:
                    XTX =  X.T.dot(X)
                    XTY = X.T.dot(Y)
                    stats_created = 1
                else:
                    XTX += X.T.dot(X)
                    XTY += X.T.dot(Y)   

                with open(log_file_name, "a") as log_file:
                    log_file.write('Block number ' + str(block_num) + ' XTX/XTY created \n')

            except StopIteration: # stop when generator has gone through all training data
                break

        # -------------------------------
        # COMPUTE LINEAR DECODER WEIGHTS
        # -------------------------------

        XTX_inv = np.linalg.inv(XTX) 
        W = np.dot(XTX_inv, XTY)

        # ----------------------------
        # SAVE LINEAR DECODER WEIGHTS
        # ----------------------------

        if not cell_type_list: # if using all cells
            model_path = directory_name+'/model/lineardecoder_W_'+activity_suffix+'.h5'
        else:
            cell_type_string = str(cell_type_list).replace(' ','').replace('.','').replace('[','').replace(']','') # make cell type list concatenated string
            model_path = directory_name+'/model/lineardecoder_W_'+activity_suffix+'_CellType_'+cell_type_string+'.h5'
            
        h5_temp = h5py.File(model_path, 'w')
        h5_temp.create_dataset('data', 
                data=W)
        h5_temp.close()

        # ------------------------------------------------------
        # CREATE DECODED IMAGES FOR TRAINING/VALIDATION/TESTING
        # ------------------------------------------------------
        if create_decoded_images:

            data_type_vec = ['training','validation','testing']

            # Loop over data types
            for i_data_type in range(len(data_type_vec)):

                data_type = data_type_vec[i_data_type]

                # ---------------------
                # LOAD  DATA GENERATOR
                # ---------------------
                activities_dir_path = data_dir+data_type+'/activities_'+activity_suffix+'/'
                original_dir_path = data_dir+data_type+'/images/'
                data_generator = ds_x_activities_y_original(batch_size=total_block_size, im_width=128, activities_dir_path=activities_dir_path, original_dir_path=original_dir_path, image_mean=immean, normalize_scale=True, shuffle=False)

                # -------------------------------- 
                # MAKE DIRECTORY IF DOESN'T EXIST
                # --------------------------------

                if cell_type_list is not None: # if using specific cell types
                    cell_type_string = str(cell_type_list).replace(' ','').replace('.','').replace('[','').replace(']','') # make cell type list concatenated string
                    decoded_images_directory_name = ''+data_dir+data_type+'/decodedimages_cellType_'+cell_type_list+'_'+LD_directory_suffix+'/'
                else:
                    decoded_images_directory_name = ''+data_dir+data_type+'/decodedimages_'+LD_directory_suffix+'/'

                if not os.path.exists(decoded_images_directory_name):
                    os.makedirs(decoded_images_directory_name)

                while True:
                    try:

                        # Generate activities (X)/ images(Y)
                        X, Y, block_num = next(data_generator)

                        # Prepare activities
                        if cell_type_list is not None:
                            X = X[:,which_n]
                        X = np.insert(X, 0, 1.0, axis=1) # insert column of bias terms

                        # Create decoded images

                        decoded_images = np.dot(X,W)

                        # Save decoded images

                        if not cell_type_list: # if using all cells
                            save_path =decoded_images_directory_name+str(block_num)+'_'+LD_directory_suffix+'.h5'
                        else:
                            cell_type_string = str(cell_type_list).replace(' ','').replace('.','').replace('[','').replace(']','') # make cell type list concatenated string
                            save_path =decoded_images_directory_name+str(block_num)+'_'+LD_directory_suffix+'_CellType_'+cell_type_list+'.h5'
                      
                        h5_temp = h5py.File(save_path, 'w')
                        h5_temp.create_dataset('data',data=decoded_images)
                        h5_temp.close()

                        with open(log_file_name, "a") as log_file:
                            log_file.write('Block number ' + str(block_num) + ' '+data_type+' decoded images created \n')

                    except StopIteration: # stop when generator has gone through all training data
                        break





                          
