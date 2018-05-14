import os
import h5py
os.environ["KERAS_BACKEND"] = 'tensorflow'
import tensorflow as tf
from keras.utils import generic_utils as keras_generic_utils
from tf_models import cae_nn
from optimization_utils import optimizer_loss_options as opt_utils
from dataset import ds_x_decoded_y_original
from dataset import image_processing as ip
from test_tube import Experiment, HyperOptArgumentParser
import numpy as np
from optimization_utils.training_utils import save_model


class CAEWrapper:
    def __init__(self, hparams):
        self.exp = None
        self.model = None
        self.hparams = hparams

    def train_main(self):
        self.print_params()

        # ---------------------------
        # EXP SETUP
        # ---------------------------
        exp = Experiment(name=self.hparams.exp_name,
                         debug=self.hparams.debug,
                         save_dir=self.hparams.tt_save_dir,
                         autosave=False,
                         description=self.hparams.tt_description)

        exp.add_argparse_meta(self.hparams)

        # ---------------------------
        # TF ADMIN
        # ---------------------------
        input_shape = [None, self.hparams.image_w, self.hparams.image_w, self.hparams.nb_img_channels]
        input_x_ph = tf.placeholder(dtype=tf.float32, shape=input_shape, name='input_x')
        input_y_ph = tf.placeholder(dtype=tf.float32, shape=input_shape, name='input_y')
        dropout_ph = tf.placeholder(dtype=tf.float32, shape=(), name='dropout_ph')
        phase_train_ph = tf.placeholder(tf.bool, name='phase_train_ph')
        # ---------------------------
        # MODEL
        # ---------------------------
        cae = cae_nn(input_x=input_x_ph, drop_rate=dropout_ph, phase_train = phase_train_ph)

        # ---------------------------
        # OPTIMIZATION PROB
        # ---------------------------
        loss = tf.reduce_mean(tf.squared_difference(cae, input_y_ph))
        optimizer = opt_utils.choose_optimizer(optimizer=self.hparams.optimizer, lr=self.hparams.lr, loss=loss)

        # ---------------------------
        # TF BOOKEEPING
        # ---------------------------
        # limit gpu mem
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.hparams.gpu_mem_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # init graph vars
        sess.run(tf.global_variables_initializer())

        # init object to save tf model
        saver = tf.train.Saver(max_to_keep=self.hparams.max_to_keep)

        # ---------------------------
        # RUN TRAINING
        # ---------------------------
        total_batch_nb = 0
        best_val_loss = 100000
        val_err = 10000
        nb_epochs_val_not_improved = 0
        total_samples_served = 0

        for epoch in range(self.hparams.nb_epochs):
            print('\nepoch: {}\n'.format(epoch))
            epoch_batch_nb = 0
            progbar = keras_generic_utils.Progbar(self.hparams.nb_imgs_per_epoch)
            train_gen = ds_x_decoded_y_original(batch_size=self.hparams.batch_size,
                                                im_width=self.hparams.image_w,
                                                decoded_dir_path=self.hparams.train_decoded_dir_path,
                                                decoded_suffix = self.hparams.decoded_suffix,
                                                original_dir_path=self.hparams.train_original_dir_path,
                                                image_mean=self.hparams.image_norm_mean,
                                                normalize_scale=self.hparams.normalize_scale)

            # run through batches in that epoch
            for X_low_res, Y_ori in train_gen:
                X_low_res, Y_ori = self.reshape(X_low_res, Y_ori,
                                           image_w=self.hparams.image_w,
                                           nb_img_channels=self.hparams.nb_img_channels)

                feed_dict = {input_x_ph: X_low_res,
                             input_y_ph: Y_ori, 
                             phase_train_ph: True}

                # OPT: run one step of optimization
                optimizer.run(session=sess, feed_dict=feed_dict)

                # check train error periodically
                total_samples_served += len(X_low_res)
                prog_vals = []
                # -----------------------
                # TRAIN ERROR CHECK
                # -----------------------
                if epoch_batch_nb % self.hparams.eval_tng_err_every_n_batches == 0:
                    train_err = loss.eval(session=sess, feed_dict=feed_dict)
                    metrics = {'train_err': train_err, 'total_bn': total_batch_nb, 'epoch_bn': epoch_batch_nb, 'epoch': epoch}
                    prog_vals.append(('train_err', train_err))
                    exp.add_metric_row(metrics)

                # -----------------------
                # VAL ERROR CHECK
                # -----------------------
                if (epoch_batch_nb + 1) % self.hparams.eval_val_err_every_n_batches == 0:
                    # compute val loss
                    val_err, x_val, y_val = self.calculate_val_loss(input_x_ph, input_y_ph, dropout_ph,phase_train_ph, loss, sess)
                    prog_vals.append(('val_err', val_err))

                    # generate images to save only on the last val batch
                    x_generated = sess.run(cae, feed_dict={input_x_ph: x_val[0:self.hparams.nb_eval_imgs_to_save], dropout_ph: 1.0, phase_train_ph: False})
                    eval_metrics = ip.eval_imgs(original=y_val, decoded=x_val, nn_decoded=x_generated,
                                                dataset_name=self.hparams.exp_name,
                                                nb_to_plot=self.hparams.nb_eval_imgs_to_save,
                                                include_output_img=True)

                    # track metrics
                    eval_metrics['total_bn'] = total_batch_nb
                    eval_metrics['epoch_bn'] = epoch_batch_nb
                    eval_metrics['val_err'] = val_err
                    eval_metrics['epoch'] = epoch
                    exp.add_metric_row(eval_metrics)

                    # check early stopped conditions if requested
                    if self.hparams.enable_early_stop:
                        should_save_model, should_stop, best_val_loss, nb_epochs_val_not_improved = self.check_early_stop(self.hparams, val_err, best_val_loss, nb_epochs_val_not_improved)

                        if should_save_model:
                            # model save
                            save_model(saver=saver, hparams=self.hparams, sess=sess, epoch=epoch, version=exp.version, epoch_bn=epoch_batch_nb)

                        if should_stop:
                            exit(1)

                    exp.save()

                # end of batch bookeeping
                progbar.add(len(X_low_res), values=prog_vals if len(prog_vals) > 0 else None)
                epoch_batch_nb += 1
                total_batch_nb += 1

            # -----------------------
            # END OF EPOCH BOOKEEPING
            # -----------------------
            exp.save()

            if not self.hparams.enable_early_stop:
                save_model(saver=saver, hparams=self.hparams, sess=sess, epoch=epoch, version=exp.version,
                           epoch_bn=epoch_batch_nb)

            if epoch > self.hparams.nb_epochs:
                print('max epochs reached...')
                break


    # -----------------------------------------
    # TRAINING UTILS
    # -----------------------------------------
    def check_early_stop(self, val_err, best_val_loss, nb_epochs_val_not_improved):
        # early stoping criteria
        new_best_val_loss = best_val_loss
        new_nb_epochs_val_not_improved = nb_epochs_val_not_improved+1
        should_save_model = False

        val_improved = val_err < self.hparams.early_stop_threshold * best_val_loss
        if val_improved:
            new_nb_epochs_val_not_improved = 0
            new_best_val_loss = val_err
            should_save_model = True

        should_stop = new_nb_epochs_val_not_improved >= self.hparams.no_improvement_nb_epochs_to_stop
        return should_save_model, should_stop, new_best_val_loss, new_nb_epochs_val_not_improved


    def calculate_val_loss(self, input_x_ph, input_y_ph, dropout_ph,phase_train_ph, loss, sess):

        # init new val gen to reset val progress
        # ensures we always start from the same val image
        val_gen = ds_x_decoded_y_original(batch_size=self.hparams.batch_size,
                                          im_width=self.hparams.image_w,
                                          decoded_dir_path=self.hparams.val_decoded_dir_path,
                                          decoded_suffix = self.hparams.decoded_suffix,
                                          original_dir_path=self.hparams.val_original_dir_path,
                                          image_mean=self.hparams.image_norm_mean,
                                          bucket_range=[0, 1, 2, 3, 4],
                                          normalize_scale=self.hparams.normalize_scale)

        # check val across n batches requested
        total_val_err = []
        for i in range(self.hparams.nb_val_batches):
            # generate eval batch
            x_val, y_val = next(val_gen)
            x_val, y_val = self.reshape(x_val, y_val, image_w=self.hparams.image_w, nb_img_channels=self.hparams.nb_img_channels)

            # eval val batch
            feed_dict = {input_x_ph: x_val,
                         input_y_ph: y_val,
                         dropout_ph: 1.0,
                         phase_train_ph: False}
            total_val_err.append(loss.eval(session=sess, feed_dict=feed_dict))

        # track val metrics
        val_err = np.mean(total_val_err)
        return val_err, x_val, y_val


    def reshape(self,x, y, image_w, nb_img_channels):
        # reshape X
        x_shape = len(x), image_w, image_w, nb_img_channels
        x = x.reshape(x_shape)

        # reshape Y
        y_shape = len(y), image_w, image_w, nb_img_channels
        y = y.reshape(y_shape)

        return x, y


    def print_params(self):
        # print params so we can see training
        print('-'*100, '\nTNG PARAMS:\n', '-'*100)
        for pk, pv in vars(self.hparams).items():
            print('{}: {}'.format(pk, pv))
        print('-'*100, '\n\n')

    # -----------------------------------------
    # TESTING CODE FOR EVALUATING MODEL
    # -----------------------------------------

    def test_main(self):   
        self.print_params()

        # ---------------------------
        # EXP SETUP
        # ---------------------------
        exp = Experiment(name=self.hparams.exp_name,
                         debug=self.hparams.debug,
                         save_dir=self.hparams.tt_save_dir,
                         autosave=False,
                         description=self.hparams.tt_description)

        exp.add_argparse_meta(self.hparams)

        # ---------------------------
        # TF BOOKEEPING
        # ---------------------------
        # limit gpu mem
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.hparams.gpu_mem_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # ---------------------------
        # LOAD MODEL
        # ---------------------------
        input_ph, model_predict = self.load_model(sess=sess,
                                             opt_name=self.hparams.predict_opt_name,
                                             input_tensor_name=self.hparams.input_tensor_name,
                                             checkpoint_file_path=self.hparams.model_ckpt_path)

        # ---------------------------
        # RUN PREDICT
        # ---------------------------
        progbar = keras_generic_utils.Progbar(self.hparams.nb_test_imgs)
        test_gen = ds_x_decoded_y_original(batch_size=self.hparams.batch_size,
                                           im_width=self.hparams.image_w,
                                           bucket_range=[0],
                                           decoded_dir_path=self.hparams.test_decoded_dir_path,
                                           decoded_suffix = self.hparams.decoded_suffix,
                                           original_dir_path=self.hparams.test_original_dir_path,
                                           normalize_scale=self.hparams.normalize_scale,
                                           shuffle=False,
                                           image_mean=self.hparams.image_norm_mean)

        # run through batches in that epoch
        image_nb = 0
        if self.hparams.save_test_prediction_bucket:
            test_final = np.zeros(shape=(self.hparams.nb_test_imgs, self.hparams.image_w, self.hparams.image_w, 1))

        for x_test, y_test in test_gen:
            if image_nb >= self.hparams.nb_test_imgs:
                break

            x_test, y_test = self.reshape(x_test, y_test,
                                     image_w=self.hparams.image_w,
                                     nb_img_channels=self.hparams.nb_img_channels)

            feed_dict = {input_ph: x_test}

            # run prediction opt
            y_hat_test = sess.run(model_predict, feed_dict=feed_dict)

            if self.hparams.save_test_prediction_bucket:
                test_final[image_nb: image_nb + len(x_test)] = y_hat_test

            eval_metrics = ip.eval_imgs(original=y_test, decoded=x_test, nn_decoded=y_hat_test,
                                        dataset_name=self.hparams.exp_name,
                                        calc_loss_per_image=True,
                                        nb_to_plot=self.hparams.batch_size,
                                        include_output_img=True)
            for metric in eval_metrics:
                metric['i'] = image_nb
                image_nb += 1
                exp.add_metric_row(metric)

            if image_nb % 500 == 0:
                exp.save()

            # end of batch bookeeping
            progbar.add(len(x_test))

        exp.save()

        if self.hparams.save_test_prediction_bucket:
            self.save_bucket(test_final, self.hparams.test_prediction_save_path, self.hparams.exp_name)

    def save_bucket(self,test_final, test_prediction_save_path, file_name):
        print('saving pred bucket...')
        h5_temp = h5py.File(test_prediction_save_path + '/' + file_name + '.h5', 'w')
        h5_temp.create_dataset('y_hat',
                               data=test_final,
                               compression="gzip",
                               dtype=np.dtype('float32'))

        h5_temp.close()

    def load_model(self,sess, opt_name, input_tensor_name, checkpoint_file_path, verbose=False):
        # restore graph definition
        saver = tf.train.import_meta_graph(checkpoint_file_path + '.meta')

        # restore weights into the graph
        saver.restore(sess, checkpoint_file_path)

        # restore the placeholder tensor
        graph = tf.get_default_graph()
        operations = graph.get_operations()
        for operation in operations:
            if verbose:
                print("Operation:", operation.name)
            for k in operation.inputs:
                if verbose:
                    print(operation.name, "Input ", k.name, k.get_shape())
            for k in operation.outputs:
                if verbose:
                    print(operation.name, "Output ", k.name)
            if verbose:
                print("\n")
        input_ph = graph.get_tensor_by_name(input_tensor_name)

        # restore the embedding op
        model_predict = tf.get_collection(opt_name)[0]

        return input_ph, model_predict


def load_train_args():
    # -----------------------------------------
    # MAIN ENTRY POINT
    # -----------------------------------------
    parser = HyperOptArgumentParser(strategy='grid_search')

    # hyperparams
    parser.add_opt_argument_list('--drop_rate', default=0.25, type=float, options=[0.25, 0.5, 0.75], tunnable=False, help="percentage of units dropped for dropout")
    parser.add_opt_argument_list('--optimizer', default='adam', options=['adam', 'sgd', 'rmsprop'], tunnable=False)
    parser.add_opt_argument_list('--lr', default=0.0004, type=float, options=[0.00005, 0.0001, 0.0002, 0.0004, 0.0008, 0.001, 0.002, 0.004], tunnable=True)

    parser.add_argument('--loss', default='mse')
    parser.add_argument('--max_to_keep', default=3, type=int, help='number of previous best models to keep saved')
    parser.add_argument('--image_w', default=128, type=int, help='image width: all images are 128x128 in the CAE')
    parser.add_argument('--nb_img_channels', default=1, type=int, help="number of channels, default grayscale=1")
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--nb_epochs', default=40, type=int)
    parser.add_argument('--eval_tng_err_every_n_batches', default=10, type=int, help="number of batches after which training error is evaluated")
    parser.add_argument('--eval_val_err_every_n_batches', default=3100, type=int, help="number of batches after which validation error is evaluated")
    parser.add_argument('--nb_imgs_per_epoch', default=930000, type=int, help="number of total images in training epoch, default for full imagenet")
    parser.add_argument('--nb_eval_imgs_to_save', default=3, type=int, help="during validation evaluation, save this many example decoded images")
    parser.add_argument('--debug', default=False, type=bool)
    parser.add_argument('--gpu_mem_fraction', default=1.0, type=float)
    parser.add_argument('--nb_val_batches', default=500, type=int, help='number of batches from val dataset to compute val loss')
    parser.add_argument('--early_stop_threshold', default=0.999, type=float, help="early stopping threshold, considered no improvement unless loss drops by more than 0.999*current loss")
    parser.add_argument('--no_improvement_nb_epochs_to_stop', default=6, type=float, help='if no improvement in loss after this many epochs, stop')
    parser.add_argument('--enable_early_stop', default=True, type=bool)
    parser.add_argument('--normalize_scale', default=True, type=bool)

    # tt params
    parser.add_argument('-en', '--exp_name', default='test')
    parser.add_argument('--tt_save_dir', default='')
    parser.add_argument('-td', '--tt_description', default='')

    parser.add_argument('--gpus_viz', default='0')

    parser.add_argument('--model_save_dir', default='', help='directory to save model parameters')
    parser.add_argument('--train_decoded_dir_path', default='', help='directory containing training lindecoded images')
    parser.add_argument('--train_original_dir_path', default='', help='directory containing original images')
    parser.add_argument('--val_decoded_dir_path', default='', help='directory containing validation lindecoded images')
    parser.add_argument('--val_original_dir_path', default='', help='directory containing validation original images')
    parser.add_argument('--decoded_suffix', default = '', help='add a suffix to the bucket names if desired')
    parser.add_argument('--image_norm_mean', default=0.4496052750126008, type=float)
    parser.add_json_config_argument('-c', '--config', type=str)
    hyperparams = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = hyperparams.gpus_viz
    return hyperparams

def load_test_args():

    parser = HyperOptArgumentParser()
    parser.add_argument('--image_w', default=128, type=int)
    parser.add_argument('--nb_test_imgs', default=10000, type=int)
    parser.add_argument('--nb_img_channels', default=1, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--debug', default=False, type=bool)
    parser.add_argument('--gpu_mem_fraction', default=1.0, type=float)
    parser.add_argument('--save_test_prediction_bucket', default=False, type=bool)
    parser.add_argument('--test_prediction_save_path', default='', type=str)

# tt params
    parser.add_argument('-en', '--exp_name', default='')
    parser.add_argument('--original_exp', default='')
    parser.add_argument('--original_exp_version', default=0, type=int)
    parser.add_argument('--tt_save_dir', default='')
    parser.add_argument('-td', '--tt_description', default='')
    parser.add_argument('--image_norm_mean', default=0.4496052750126008, type=float)
    parser.add_argument('--normalize_scale', default=True, type=bool)  # controls whether you normalize by 255 and subtract image mean
    parser.add_argument('--gpus_viz', default='0')

    parser.add_argument('--predict_opt_name', default='predict_opt')
    parser.add_argument('--input_tensor_name', default='input_x:0')
    parser.add_argument('--model_ckpt_path', default='')
    parser.add_argument('--test_decoded_dir_path', default='')
    parser.add_argument('--test_original_dir_path', default='')
    parser.add_json_config_argument('-c', '--config', type=str)

    hyperparams = parser.parse_args()


    os.environ["CUDA_VISIBLE_DEVICES"] = hyperparams.gpus_viz

    return hyperparams



if __name__ == '__main__':

    #if you want to train, load_train_args and train_main
    hyperparams = load_train_args()
    caemodel = CAEWrapper(hyperparams)
    caemodel.train_main()
    # -----------------------------------------------
    # RUN HOPT OVER N DIFFERENT GPUS AT THE SAME TIME
    # ----------------ww-------------------------------


    # def opt_function(trial_params, gpu_num):
    #     from time import sleep
    #     sleep(gpu_num * 1)  # Time in seconds.
    #
    #     GPUs = ['0', '1', '2', '3']
    #     os.environ["CUDA_VISIBLE_DEVICES"] = GPUs[gpu_num]
    #     train_main(trial_params)
    #
    #
    # hyperparams.optimize_parallel(opt_function, nb_trials=8, nb_parallel=4)

    #if you want to test, load_test_arg and test_main
    hyperparams = load_test_args()
    caemodel = CAEWrapper(hyperparms)
    caemodel.test_main()
