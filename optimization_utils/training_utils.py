import os


def save_model(saver, hparams, sess, epoch, epoch_bn, version):
    print('\nsaving model...')

    # create path if not around
    model_save_path = hparams.model_save_dir + '/{}_{}/epoch_{}/{}'.format(hparams.exp_name, version, epoch, epoch_bn)
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    model_name = '{}/model'.format(model_save_path)

    save_path = saver.save(sess, model_name)
    print('model saved at', save_path, '\n\n')
