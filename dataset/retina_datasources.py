import sys
sys.path.append('../NIPS_Refactor/dataset/')
import image_processing as ip
import os
import h5py
import numpy as np
import random
import time

# ---------------
# PUBLIC API
# ---------------
def ds_x_decoded_y_original(batch_size, im_width, decoded_dir_path, decoded_suffix, original_dir_path, image_mean, normalize_scale, bucket_range=None, shuffle=True):
    """
    Returns 1 epoch's worth of data
    :param batch_size:
    :param im_width:
    :param decoded_dir_path:
    :param original_dir_path:
    :param image_mean:
    :param bucket_range: List of bucket numbers to generate from Ex: [0, 1, 3]
    :return:
    """

    # load and sort all bucket names numerically
    # need to do this bc each os may list them differently

    random.seed(time.time())

    random_gen = random.random()
    dec_file_names = os.listdir(decoded_dir_path)
    dec_file_names = [f for f in dec_file_names if '.h5' in f]
    dec_file_names = sorted(dec_file_names, key=lambda x: int(x.split('_')[0])) 
    dec_file_names = [fn for fn in dec_file_names if fn.split('_')[-1].split('.')[0] == decoded_suffix]

    # dec_file_names = sorted(dec_file_names, key=lambda x: int(x.split('_')[0]))
    # load and sort all bucket names numerically
    ori_file_names = os.listdir(original_dir_path)
    ori_file_names = [f for f in ori_file_names if '.h5' in f]
    ori_file_names = sorted(ori_file_names, key=lambda x: int(x.split('.')[0]))

    # limit nb of buckets if requested
    if bucket_range:
        bucket_range_set = set(bucket_range)
        dec_file_names = [f for f in dec_file_names if int(f.split('_')[0]) in bucket_range_set]
        ori_file_names = [f for f in ori_file_names if int(f.split('.')[0]) in bucket_range_set]

    if shuffle:
        random.shuffle(dec_file_names, lambda: random_gen)
        random.shuffle(ori_file_names, lambda: random_gen)

    # iterate each bucket
    for ld_bucket_name, ori_bucket_name in zip(dec_file_names, ori_file_names):
        
        # load pointers to data
        dec_bucket = h5py.File(os.path.join(decoded_dir_path, ld_bucket_name), 'r')
        ori_bucket = h5py.File(os.path.join(original_dir_path, ori_bucket_name), 'r')

        # collect metadata about the buckets
        dec_key = 'data'
        ori_key = 'data'
        nb_dps = dec_bucket[dec_key].shape[0]
        ori_image_w = ori_bucket[ori_key].shape[2]

        batch_vec = np.arange(0,nb_dps,batch_size)
        if shuffle:
            random.shuffle(batch_vec)

        for i in batch_vec:
       # for i in range(0, nb_dps, batch_size):
            i_end = i + batch_size

            # load decoded images
            x_batch_dec = dec_bucket[dec_key][i: i_end].astype(np.float32)

            # load original images and rescale on disk if needed
            if ori_image_w > im_width:
                im_i_start, im_i_end, j_start, j_end = ip.calculate_cropping_idxs(original_width=ori_image_w, crop_width=im_width)
                y_batch_ori = ori_bucket[dec_key][i: i_end, im_i_start: im_i_end, j_start: j_end].astype(np.float32)
            else:
                y_batch_ori = ori_bucket[ori_key][i: i_end].astype(np.float32)

            # normalize original images from 255 to 1 and subtract image mean
            if normalize_scale:
              ip.normalize_scale(y_batch_ori, image_mean)

            yield x_batch_dec, y_batch_ori

def ds_x_activities_y_original(batch_size, im_width, activities_dir_path, original_dir_path, image_mean, normalize_scale, bucket_range=None, shuffle=True):
    """
    Returns 1 epoch's worth of data
    :param batch_size:
    :param im_width:
    :param activities_dir_path:
    :param original_dir_path:
    :param image_mean:
    :param bucket_range: List of bucket numbers to generate from Ex: [0, 1, 3]
    :return:
    """

    # load and sort all bucket names numerically
    # need to do this bc each os may list them differently

    random.seed(time.time())

    random_gen = random.random()
    act_file_names = os.listdir(activities_dir_path)
    act_file_names = [f for f in act_file_names if '.h5' in f]
    act_file_names = sorted(act_file_names, key=lambda x: int(x.split('_')[0]))

    # dec_file_names = sorted(dec_file_names, key=lambda x: int(x.split('_')[0]))
    # load and sort all bucket names numerically
    ori_file_names = os.listdir(original_dir_path)
    ori_file_names = [f for f in ori_file_names if '.h5' in f]
    ori_file_names = sorted(ori_file_names, key=lambda x: int(x.split('.')[0]))

    # limit nb of buckets if requested
    if bucket_range:
        bucket_range_set = set(bucket_range)
        act_file_names = [f for f in act_file_names if int(f.split('_')[0]) in bucket_range_set]
        ori_file_names = [f for f in ori_file_names if int(f.split('.')[0]) in bucket_range_set]

    if shuffle:
        random.shuffle(act_file_names, lambda: random_gen)
        random.shuffle(ori_file_names, lambda: random_gen)

    # iterate each bucket
    for act_bucket_name, ori_bucket_name in zip(act_file_names, ori_file_names):

        # load pointers to data
        buck_num = int(act_bucket_name.split('_')[0])
        act_bucket = h5py.File(os.path.join(activities_dir_path, act_bucket_name), 'r')
        ori_bucket = h5py.File(os.path.join(original_dir_path, ori_bucket_name), 'r')

        # collect metadata about the buckets
        act_key = 'data'
        ori_key = 'data'
        nb_dps = act_bucket[act_key].shape[0]
        ori_image_w = ori_bucket[ori_key].shape[2]

        batch_vec = np.arange(0,nb_dps,batch_size)
        if shuffle:
            random.shuffle(batch_vec)

        for i in batch_vec:
       # for i in range(0, nb_dps, batch_size):
            i_end = i + batch_size

            # load activities
            x_batch_act = act_bucket[act_key][i: i_end].astype(np.float32)

            # load original images and rescale on disk if needed
            if ori_image_w > im_width:
                im_i_start, im_i_end, j_start, j_end = ip.calculate_cropping_idxs(original_width=ori_image_w, crop_width=im_width)
                y_batch_ori = ori_bucket[ori_key][i: i_end, im_i_start: im_i_end, j_start: j_end].astype(np.float32)
            else:
                y_batch_ori = ori_bucket[ori_key][i: i_end].astype(np.float32)

            # normalize original images from 255 to 1 and subtract image mean
            if normalize_scale:
              ip.normalize_scale(y_batch_ori, image_mean)

            yield x_batch_act, y_batch_ori, buck_num
        act_bucket.close()
        ori_bucket.close()

