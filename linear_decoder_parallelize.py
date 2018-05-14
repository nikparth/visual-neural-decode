import os
import numpy as np
import h5py
from time import time

from dataset import ds_x_activities_y_original

## Import from constants file
data_dir = 'FakeData/' 
cell_type_inds_path = 'temp/' # optional, only if you ever want to split cell types
linear_decoder_save_path = 'FakeModels/'
immean = 0.5
total_block_size = 10000

class LinearDecoderParallelize():

	def __init__(self):
		self.data = [] 

	def ComputeStatistics(activity_suffix, block_vec, im_width=128, cell_type_list=None, data_type='training'):
		'''
		Inputs:
			activitiy_suffix: string that follows activities you want, for example 'spatialSim' for activitiies in folder 'activities_spatialSim'
			block_vec: numpy array of block indices for which you want to compute statistics 
			im_width: the size of one side of the images (code only handles square images for now but this could be easily modified)
			cell_type_list: numpy array of index of specific cell types you want decoded (corresponding to cell_type_inds.npy), for example np.asarray(0,1) if you want to grab all cell types identified as 0 or 1
			data_type: the type of images/activities to compute the linear decoder using, should probably always be training 
		Outputs:
			Computes and saves the XTX/XTY statistics for every block in block_vec-
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
	 
		# --------------------
		# LOAD DATA GENERATOR
		# --------------------
		activities_dir_path = data_dir+data_type+'/activities_'+activity_suffix+'/'
		original_dir_path = data_dir+data_type+'/images/'
		data_generator = ds_x_activities_y_original(batch_size=total_block_size, im_width=128, activities_dir_path=activities_dir_path, original_dir_path=original_dir_path, image_mean=immean, normalize_scale=True, bucket_range=block_vec.tolist(), shuffle=False)

		# -----------------
		# PARSE CELL TYPES
		# -----------------

		if cell_type_list is not None: # use specific cell types instead of all the data
			cell_type_inds = np.load(cell_type_inds_path+'/cell_type_inds.npy') 
			which_n = np.in1d(cell_type_inds, cell_type_list) # specify a specific cell type  
			
		# --------------------------------------- 
		# CALCULATE SAMPLE XTX AND XTY STATISTICS 
		# --------------------------------------- 
						  
		start_time = time()

		for i_block in block_vec:
			
			log_file_name = directory_name + '/logs/XTX_XTY_Block_'+str(i_block)+'.txt'
		
			with open(log_file_name, "a") as log_file:
				log_file.write('Block number ' + str(i_block) + '\n')

			# Generate activities (X)/ images(Y)
			X, Y, block_num = next(data_generator)
			Y = Y.reshape((-1,im_width*im_width))

			# Sanity check that generator is creating correct block
			if i_block != block_num:
				print('ERROR: INACCURATE DATA GENERATION')
				sys.exit()

			# Prepare activities
			if cell_type_list is not None:
				X = X[:,which_n]
			X = np.insert(X, 0, 1.0, axis=1) # insert column of bias terms

			# Compute XTX/XTY

			XTX = X.T.dot(X)
			XTY = X.T.dot(Y)
				
			# Create data saving names
			if not cell_type_list: # if using all cells
				model_path_XTX = directory_name+'/model/XTX_B_'+str(i_block)+'_W'+activity_suffix+'.h5'
				model_path_XTY = directory_name+'/model/XTY_B_'+str(i_block)+'_W'+activity_suffix+'.h5'
			else: # if using specific cell type
				cell_type_string = str(cell_type_list).replace(' ','').replace('.','').replace('[','').replace(']','') # make cell type list concatenated string
				model_path_XTX = directory_name+'/model/XTX_B_'+str(i_block)+'_W'+activity_suffix+'_CellType_'+cell_type_string+'.h5'
				model_path_XTY = directory_name+'/model/XTY_B_'+str(i_block)+'_W'+activity_suffix+'_CellType_'+cell_type_string+'.h5'
				
			# Save data
			h5_temp = h5py.File(model_path_XTX, 'w')
			h5_temp.create_dataset('data', 
					data=XTX)
			h5_temp.close()
			
			h5_temp = h5py.File(model_path_XTY, 'w')
			h5_temp.create_dataset('data', 
					data=XTY)
			h5_temp.close()                
				   
			end_time = time()

			# Log memory/time

			with open(log_file_name, "a") as log_file:
				log_file.write('Block time ' + str((end_time-start_time)/60.) + ' \n')

			start_time = time()

		return

	def CreateFullW(activity_suffix, block_vec, im_width=128, cell_type_list=None, data_type='training'):
		'''
		Inputs:
			activitiy_suffix: string that follows activities you want, for example 'spatialSim' for activitiies in folder 'activities_spatialSim'
			block_vec: numpy array of block indices for which you want to add statistics, should usually be all of training data blocks
			im_width: the size of one side of the images (code only handles square images for now but this could be easily modified)
			cell_type_list: numpy array of index of specific cell types you want decoded (corresponding to cell_type_inds.npy), for example np.asarray(0,1) if you want to grab all cell types identified as 0 or 1
			data_type: the type of images/activities to compute the linear decoder using, should probably always be training 
		Outputs:
			Computes and saves linear decoder weights
		'''
		start = time()

		if cell_type_list is not None: # if using specific cell types
			cell_type_string = str(cell_type_list).replace(' ','').replace('.','').replace('[','').replace(']','') # make cell type list concatenated string
			directory_name = linear_decoder_save_path+'/linear_decoder_'+activity_suffix+'_'+cell_type_string+''
		else:
			directory_name = linear_decoder_save_path+'/linear_decoder_'+activity_suffix+''

		log_file_name = directory_name + '/logs/CreateFullW.txt'

		# Log block indices
		with open(log_file_name, "a") as log_file:
			log_file.write('blocks: {} \n'.format(block_vec))

		# ------------------------------------
		# LOOP OVER BLOCKS AND ADD STATISTICS
		# ------------------------------------
		stats_created=0
		for i_block in block_vec:
		   
			# Load block specific XTX/XTY
			if not cell_type_list: # if using all cells
				model_path_XTX = directory_name+'/model//XTX_B_'+str(i_block)+'_W'+activity_suffix+'.h5'
				model_path_XTY = directory_name+'/model/XTY_B_'+str(i_block)+'_W'+activity_suffix+'.h5'
			else: # if using specific cell type
				cell_type_string = str(cell_type_list).replace(' ','').replace('.','').replace('[','').replace(']','') # make cell type list concatenated string
				model_path_XTX = directory_name+'/model/XTX_B_'+str(i_block)+'_W'+activity_suffix+'_CellType_'+cell_type_string+'.h5'
				model_path_XTY = directory_name+'/model/XTY_B_'+str(i_block)+'_W'+activity_suffix+'_CellType_'+cell_type_st+'.h5'
	 
			XTX_block = h5py.File(model_path_XTX, 'r')
			thisXTX = XTX_block['data'][:].astype('float32')
			XTX_block.close()
			
			XTY_block = h5py.File(model_path_XTY, 'r')
			thisXTY = XTY_block['data'][:].astype('float32')
			XTY_block.close()

			# Add this blocks statistics to overall ones
			if stats_created == 0:
				XTX =  thisXTX
				XTY = thisXTY
				stats_created = 1
			else:
				XTX += thisXTX
				XTY += thisXTY

			with open(log_file_name, "a") as log_file:
				log_file.write('Block ' + str(i_block) + ' \n')

		with open(log_file_name, "a") as log_file:
			log_file.write('Computing and saving model... \n')

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

		total_time = (time() - start)/60.
		with open(log_file_name, "a") as log_file:
			log_file.write('total time: {} \n'.format(total_time))

		return


	def CreateDecodedImages(activity_suffix, LD_directory_suffix, block_vec, im_width=128, cell_type_list=None, data_type='training'):
		'''
		Inputs:
			activitiy_suffix: string that follows activities you want, for example 'spatialSim' for activitiies in folder 'activities_spatialSim'
			LD_directory_suffix: the suffix for the folder in which to save linear decoded images, aka 'LD_images'
			block_vec: numpy array of block indices for which you want to add statistics, should usually be all of training data blocks
			im_width: the size of one side of the images (code only handles square images for now but this could be easily modified)
			cell_type_list: numpy array of index of specific cell types you want decoded (corresponding to cell_type_inds.npy), for example np.asarray(0,1) if you want to grab all cell types identified as 0 or 1
			data_type: the type of data for which to compute linear decoded images
		Outputs:
			Computes and saves linear decoder images for blocks in block_vec 
		'''

		if cell_type_list is not None: # if using specific cell types
			cell_type_string = str(cell_type_list).replace(' ','').replace('.','').replace('[','').replace(']','') # make cell type list concatenated string
			directory_name = linear_decoder_save_path+'/linear_decoder_'+activity_suffix+'_'+cell_type_string+''
		else:
			directory_name = linear_decoder_save_path+'/linear_decoder_'+activity_suffix+''

		log_file_name = directory_name + '/logs/CreateDecodedImages_'+data_type+'.txt'

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

		# ---------------------------- 
		# LOAD LINEAR DECODER WEIGHTS
		# ----------------------------
 
		if not cell_type_list: # if using all cells
			model_path = directory_name+'/model/lineardecoder_W_'+activity_suffix+'.h5'
		else:
			cell_type_string = str(cell_type_list).replace(' ','').replace('.','').replace('[','').replace(']','') # make cell type list concatenated string
			model_path = directory_name+'/model/lineardecoder_W_'+activity_suffix+'_CellType_'+cell_type_string+'.h5'
			
		h5_temp = h5py.File(model_path, 'r')
		W = h5_temp['data'][:].astype('float32')
		h5_temp.close()

		# --------------------
		# LOAD DATA GENERATOR
		# --------------------

		activities_dir_path = data_dir+data_type+'/activities_'+activity_suffix+'/'
		original_dir_path = data_dir+data_type+'/images/'
		data_generator = ds_x_activities_y_original(batch_size=total_block_size, im_width=128, activities_dir_path=activities_dir_path, original_dir_path=original_dir_path, image_mean=immean, normalize_scale=True, bucket_range=block_vec.tolist(), shuffle=False)

		# --------------------------------------------
		# LOOP THROUGH BLOCKS AND SAVE DECODED IMAGES
		# --------------------------------------------
 
		for i_block in block_vec:

			# Generate activities

			X, Y, block_num = next(data_generator)

			# Sanity check that generator is creating correct block

			if i_block != block_num:
				print('ERROR: INACCURATE DATA GENERATION')
				sys.exit()

			# Prepare activities

			if cell_type_list is not None:
				X = X[:,which_n]
			X = np.insert(X, 0, 1.0, axis=1) # insert column of bias terms
 
			# Compute decoded images

			decoded_images = np.dot(X,W)

			# Save decoded images

			if not cell_type_list: # if using all cells
				save_path =decoded_images_directory_name+str(i_block)+'_'+LD_directory_suffix+'.h5'
			else:
				cell_type_string = str(cell_type_list).replace(' ','').replace('.','').replace('[','').replace(']','') # make cell type list concatenated string
				save_path =decoded_images_directory_name+str(i_block)+'_'+LD_directory_suffix+'_CellType_'+cell_type_list+'.h5'
		  
			h5_temp = h5py.File(save_path, 'w')
			h5_temp.create_dataset('data',data=decoded_images)
			h5_temp.close()

			with open(log_file_name, "a") as log_file:
				log_file.write('Block ' + str(i_block) + ' '+data_type+' decoded images saved \n')
