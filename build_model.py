import collections
import time
import numpy as np
import progressbar
import torch
import config
from models.MSCN import MSCNModel
from models.QPPNet import QPPNet
from utils.opt_parser import save_opt
from utils.util import Utils


def build_md(dataset, model_type, opt, dim_dict):
    models = {
        "QPPNet": QPPNet,
        "MSCN": MSCNModel,
    }

    MODEL_CALL = collections.defaultdict(lambda: QPPNet, models)

    save_dir = opt.mid_data_dir + opt.save_dir + "/" + str(opt.batch_size)

    for i in range(5):
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.random.manual_seed(1)

    torch.set_default_tensor_type(torch.FloatTensor)

    model = MODEL_CALL[model_type](opt, dim_dict)

    FUNCTION = {
        "FR_eval": model.calculate_FR,
        "GD_eval": model.calculate_FR,
        "GREEDY_eval": model.calculate_GREEDY,
    }

    total_iter = 0

    if opt.mode == 'eval':
        logf = open(save_dir + "/" + opt.logfile, 'w+')
        model.load("best")
        save_opt(opt, logf)

        Utils.path_build(save_dir)

        start_eval_time = time.time()
        model.evaluate(dataset.test_dataset)
        end_eval_time = time.time()
        eval_log = open(save_dir + "/eval_time.txt", 'w+')
        print('eval:', config.total)
        eval_log.write(
            "Total Eval Time: " + str(float(round((end_eval_time - start_eval_time) * 1000000 - config.total))) + "us")
        eval_log.close()

        print('total_loss: {}; test_loss: {}; pred_err: {};' \
              .format(model.last_total_loss, model.last_test_loss,
                      model.last_pred_err))

    elif opt.mode == 'train':
        logf = open(save_dir + "/" + opt.logfile, 'w+')
        start_time = time.time()
        save_opt(opt, logf)
        model.test_dataset = dataset.test_dataset

        end_epoch = opt.end_epoch if model_type == "QPPNet" else 1

        bar = progressbar.ProgressBar(widgets=[
            progressbar.Percentage(),
            ' (', progressbar.SimpleProgress(), ') ',
            ' (', progressbar.AbsoluteETA(), ') ', ])

        for epoch in bar(range(opt.start_epoch, end_epoch)):
            batch_size = opt.batch_size

            samp_dicts = dataset.sample_data(batch_size)
            total_iter += opt.batch_size
            model.set_input(samp_dicts)

            model.optimize_parameters(epoch)
            logf.write("epoch: " + str(epoch) + "; iter_num: " + str(total_iter) \
                       + '; total_loss: {}; test_loss: {}; pred_err: {}; ' \
                       .format(model.last_total_loss, model.last_test_loss, model.last_pred_err))

            if epoch % 50 == 0:
                print("epoch: " + str(epoch) + "; iter_num: " + str(total_iter) \
                      + '; total_loss: {}; test_loss: {}; pred_err: {}; ' \
                      .format(model.last_total_loss, model.last_test_loss, model.last_pred_err))

            # cache our latest model every <save_latest_freq> iterations
            if (epoch + 1) % opt.save_latest_epoch_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch + 1, total_iter))
                model.save_units(epoch + 1)

        end_time = time.time()

        eval_log = open(save_dir + "/eval.txt", 'w+')
        eval_log.write('total_loss: {}; test_loss: {}; pred_err: {}; ' \
                       .format(model.last_total_loss, model.last_test_loss,
                               model.last_pred_err))
        print('train:', config.total)
        print("running time:", float(round((end_time - start_time) * 1000000) / 1000000))
        eval_log.write("\nTotal Running Time: " + str(
            float(round((end_time - start_time) * 1000000 - config.total) / 1000000)) + "s")
        print("final running time:", float(round((end_time - start_time) * 1000000 - config.total) / 1000000))
        config.total = 0.0
        eval_log.close()

        model.load("best")

        # 输出评估结果
        print("begin evaluate")
        start_eval_time = time.time()
        model.evaluate(dataset.test_dataset)
        end_eval_time = time.time()

        eval_log = open(save_dir + "/eval_time.txt", 'w+')
        print('eval:', config.total)
        print("eval time:", float(round((end_eval_time - start_eval_time) * 1000000) / 1000000))
        eval_log.write(
            "Total Eval Time: " + str(float(round((end_eval_time - start_eval_time) * 1000000 - config.total))) + "us")
        print("final eval time:", float(round((end_eval_time - start_eval_time) * 1000000 - config.total) / 1000000))
        eval_log.close()

        logf.close()

        for i in range(5):
            torch.clear_autocast_cache()
            torch.cuda.empty_cache()

    elif opt.mode == 'change_train':
        logf = open(save_dir + "/" + opt.logfile, 'w+')
        save_opt(opt, logf)

        Utils.path_build(save_dir)

        logf = open(save_dir + "/" + opt.logfile, 'w+')
        start_time = time.time()
        save_opt(opt, logf)
        model.test_dataset = dataset.test_dataset

        model.load_unchange("best")

        bar = progressbar.ProgressBar(widgets=[progressbar.Percentage(),
                                               ' (', progressbar.SimpleProgress(), ') ',
                                               ' (', progressbar.AbsoluteETA(), ') ', ])

        for epoch in bar(range(0, 200)):
            batch_size = opt.batch_size

            samp_dicts = dataset.sample_data(batch_size)
            total_iter += opt.batch_size
            model.set_input(samp_dicts)

            model.optimize_parameters(epoch)
            logf.write("epoch: " + str(epoch) + "; iter_num: " + str(total_iter) \
                       + '; total_loss: {}; test_loss: {}; pred_err: {};' \
                       .format(model.last_total_loss, model.last_test_loss,
                               model.last_pred_err))

            losses = model.get_current_losses()
            loss_str = "losses: "
            for op in losses:
                loss_str += str(op) + " [" + str(losses[op]) + "]; "

            if epoch % 50 == 0:
                print("epoch: " + str(epoch) + "; iter_num: " + str(total_iter) \
                      + '; total_loss: {}; test_loss: {}; pred_err: {};' \
                      .format(model.last_total_loss, model.last_test_loss,
                              model.last_pred_err))
                print(loss_str)

            logf.write(loss_str + '\n')

        end_time = time.time()

        eval_log = open(save_dir + "/eval.txt", 'w+')
        eval_log.write('total_loss: {}; test_loss: {}; pred_err: {}; ' \
                       .format(model.last_total_loss, model.last_test_loss,
                               model.last_pred_err))
        eval_log.write("\nTotal Running Time: " + str(
            float(round((end_time - start_time) * 1000000 - config.total) / 1000000)) + "s")
        eval_log.close()
        logf.close()

        config.total = 0.0

        model.load("best")
        # 输出评估结果
        start_eval_time = time.time()
        model.evaluate(dataset.test_dataset)
        end_eval_time = time.time()
        eval_log = open(save_dir + "/eval_time.txt", 'w+')
        eval_log.write(
            "Total Eval Time: " + str(float(round((end_eval_time - start_eval_time) * 1000000 - config.total))) + "us")
        eval_log.close()
        for i in range(5):
            torch.clear_autocast_cache()
            torch.cuda.empty_cache()

    elif '_eval' in opt.mode:
        function = FUNCTION[opt.mode]

        model.load("best")
        start_eval_time = time.time()
        function(dataset.train_dataset)
        end_eval_time = time.time()

        eval_log = open(save_dir + "/" + opt.mode.replace("_eval", "") + "_time.txt", 'w+')
        print('eval:', config.total)
        eval_log.write(
            "Total Calculate Time: " +
            str(float(round((end_eval_time - start_eval_time) * 1000000 - config.total))) + "us")
        eval_log.close()

    del model
