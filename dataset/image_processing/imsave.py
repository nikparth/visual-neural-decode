import numpy as np
import os
import h5py
from skimage.measure import compare_ssim as ssim


def eval_imgs(original,
              decoded,
              nn_decoded,
              dataset_name,
              nb_to_plot,
              calc_loss_per_image=False,
              return_x_gen_only=False,
              include_output_img=False):

    # short circuit if we just need to return x_gen
    if return_x_gen_only:
        X_gen = np.transpose(nn_decoded, [0, 3, 1, 2])
        if np.min(X_gen) < 0:
            normalize_to_pixel_space(X_gen)
        return X_gen

    # organize by type and make the final image
    # decoded, nn_decoded, original
    X, decoded, nn_decoded, original = arrange_images(original, decoded, nn_decoded, nb_to_plot)

    # if requested MSE broken down per sample image
    # broadcast mse across axis and add the relevant image
    if calc_loss_per_image:
        # Xi shape = N x C x W x H
        indiv_axis = (1, 2, 3)
        mse_decoded = imagewise_mse(decoded, original)
        img_shape = decoded.shape[2:]
        ssim_decoded = imagewise_ssim(decoded, original, img_shape)
        mse_nn_decoded = imagewise_mse(nn_decoded, original)
        ssim_nn_decoded = imagewise_ssim(nn_decoded, original, img_shape)

        # prepare for saving
        X = np.transpose(X, (1, 2, 0))
        X = X[:, :, 0]

        # X shape is now H x W (2560, 384)

        results = []
        i = 0
        for decoded_mse, nn_mse, decoded_ssim, nn_ssim in zip(mse_decoded, mse_nn_decoded, ssim_decoded, ssim_nn_decoded):

            i_start = i*128
            i_end = i_start + 128
            result = {'{}_mse_dec'.format(dataset_name): round(decoded_mse, 5),
                      '{}_ssim_dec'.format(dataset_name): round(decoded_ssim, 5),
                      '{}_ssim_nn'.format(dataset_name): round(nn_ssim, 5),
                      '{}_mse_nn'.format(dataset_name): round(nn_mse, 5)}
            if include_output_img:
                result['jpg_{}_img'.format(dataset_name)] = X[i_start:i_end, :]

            results.append(result)
            i += 1

        return results

    # calculate aggregate mse if individual was not requested
    else:
        # calculate MSE
        mse_decoded = ((decoded - original) ** 2).mean()
        mse_nn_decoded = ((nn_decoded - original) ** 2).mean()
        img_shape = decoded.shape[2:]
        ssim_decoded = np.asarray([np.abs(1 - ssim(x.reshape(img_shape), y.reshape(img_shape))) for x, y in zip(decoded, original)]).mean()
        ssim_nn_decoded = np.asarray([np.abs(1 - ssim(x.reshape(img_shape), y.reshape(img_shape))) for x, y in zip(nn_decoded, original)]).mean()

        # prepare for saving
        X = np.transpose(X, (1, 2, 0))
        X = X[:, :, 0]

        # return information for the logger
        result = {'{}_mse_dec'.format(dataset_name): round(mse_decoded, 5),
                  '{}_ssim_dec'.format(dataset_name): round(ssim_decoded, 5),
                  '{}_ssim_nn'.format(dataset_name): round(ssim_nn_decoded, 5),
                  '{}_mse_nn'.format(dataset_name): round(mse_nn_decoded, 5)}

        if include_output_img:
            result['jpg_{}_img'.format(dataset_name)] = X

        return result


def imagewise_mse(images_a, images_b):
    indiv_axis = (1, 2, 3)
    mse_results = ((images_a - images_b) ** 2).mean(axis=indiv_axis)
    return mse_results


def imagewise_ssim(images_a, images_b, img_shape):
    return np.asarray([np.abs(1 - ssim(x.reshape(img_shape), y.reshape(img_shape))) for x, y in zip(images_a, images_b)])


def arrange_images(original, decoded, nn_decoded, nb_to_plot):
    decoded = decoded[:nb_to_plot]
    decoded = np.transpose(decoded, [0, 3, 1, 2]) # N x W x H x C
    # decoded = normalize_to_pixel_space(decoded)

    nn_decoded = nn_decoded[:nb_to_plot]
    nn_decoded = np.transpose(nn_decoded, [0, 3, 1, 2])
    # nn_decoded = normalize_to_pixel_space(nn_decoded)

    original = original[:nb_to_plot]
    original = np.transpose(original, [0, 3, 1, 2])
    # original = normalize_to_pixel_space(original)

    # put |decoded, generated, original| images next to each other
    X = np.concatenate((decoded, nn_decoded, original), axis=3)

    # make one giant block of images
    X = np.concatenate(X, axis=1)

    return X, decoded, nn_decoded, original


def normalize_to_pixel_space(X):
    """
    Normalizes to pixel space no matter what the current transformation is
    :param X:
    :return:
    """
    # if < 0, shift to positive space
    if np.min(X) < 0:
        mins = np.min(X, axis=(1, 2, 3))
        for i in range(len(X)):
            X[i] += abs(mins[i])

    # if > 1 normalize bn 0,1
    if np.max(X) > 1:
        maxs = np.max(X, axis=(1, 2, 3))
        for i in range(len(X)):
            X[i] /= maxs[i]

    # scale to 255.0
    X *= 255.0
    return X


def save_tst_imgs_arr(y, y_hat, epoch, save_dir):
    """
    Saves a set of images as a result set
    :param y:
    :param y_hat:
    :param epoch:
    :param save_dir:
    :return:
    """
    if not os.path.isdir(save_dir + '/test_image_preds'):
        os.mkdir(save_dir + '/test_image_preds')

    h5_temp = h5py.File(save_dir + '/test_image_preds/' + 'test_imgs_epoch_{}.h5'.format(epoch), 'w')
    h5_temp.create_dataset('y',
                           data=y,
                           compression="gzip",
                           dtype=np.dtype('float32'))

    h5_temp.create_dataset('y_hat',
                           data=y_hat,
                           compression="gzip",
                           dtype=np.dtype('float32'))

    h5_temp.close()
