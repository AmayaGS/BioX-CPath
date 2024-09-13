# -*- coding: utf-8 -*-

# Misc
import os
import os.path
import time
import pandas as pd

# PyTorch
import torch

# PyG
from torch_geometric.loader import DataLoader

# KRAG functions
from train_test_loops.krag_train_val_loop import train_val_loop
from train_test_loops.krag_test_loop import test_loop
from utils.setup_utils import seed_everything
from utils.profiling_utils import train_profiler, test_profiler
from utils.model_utils import load_data, minority_sampler, prepare_data_loaders, initialise_model
from utils.model_utils import summarise_train_results, summarise_test_results
from utils.plotting_functions_utils import plot_roc_curve

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

use_gpu = torch.cuda.is_available()


def train_model(args, results_dir, logger):
    seed_everything(args.seed)
    os.makedirs(results_dir, exist_ok=True)

    run_settings, checkpoints, data_dict, sss_folds = load_data(args, results_dir)

    mean_best_acc = []
    mean_best_AUC = []

    # training_folds = []
    # validation_folds = []
    # for folds, splits in sss_folds.items():
    #     for i, (split, patient_ids) in enumerate(splits.items()):
    #         if i == 0:
    #             train_dict = dict(filter(lambda i:i[0] in patient_ids, graph_dict.items()))
    #             training_folds.append(train_dict)
    #         if i== 1:
    #             val_dict = dict(filter(lambda i:i[0] in patient_ids, graph_dict.items()))
    #             validation_folds.append(val_dict)

    training_folds, validation_folds, _ = prepare_data_loaders(data_dict, sss_folds)

    train_profiler.set_logger(logger)

    all_results = []
    for fold_idx, (train_fold, val_fold) in enumerate(zip(training_folds, validation_folds)):
        train_profiler.reset_gpu_memory()
        model, loss, optimizer, lr_scheduler = initialise_model(args)

        sampler = minority_sampler(train_fold)
        train_loader = DataLoader(train_fold, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, sampler=sampler, drop_last=False)
        val_loader = DataLoader(val_fold, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, drop_last=False)

        start_time = time.time()
        _, results_dict, best_acc, best_AUC = train_val_loop(model,
                                                             train_loader,
                                                             val_loader,
                                                             loss,
                                                             optimizer,
                                                             results_dir,
                                                             logger,
                                                             fold_idx,
                                                             n_classes=args.n_classes,
                                                             num_epochs=args.num_epochs,
                                                             checkpoint=args.checkpoint,
                                                             checkpoint_path= checkpoints)
        total_time = time.time() - start_time
        train_profiler.update_epoch_time(total_time / args.num_epochs)

        all_results.append(results_dict)
        mean_best_acc.append(best_acc)
        mean_best_AUC.append(best_AUC)

        df_results = pd.DataFrame.from_dict(results_dict)
        df_results.to_csv(results_dir + "/" + run_settings + "_fold_" + str(fold_idx) + ".csv", index=False)

    summarise_train_results(all_results, mean_best_acc, mean_best_AUC, results_dir, run_settings)

    logger.info("Full Training & validation completed in {:.0f}m {:.0f}s"
                .format(total_time // 60, total_time % 60))
    logger.info("Training Profiling Results:")
    train_profiler.report(is_training=True)


def test_model(args, results_dir, logger):
    run_settings, checkpoints, data_dict, sss_folds = load_data(args, results_dir)

    # testing_folds = []
    # for fold, splits in sss_folds.items():
    #     # print(f"Processing fold {fold + 1}/{len(splits)}")
    #     for i, (split, patient_ids) in enumerate(splits.items()):
    #         if i == 2:
    #             test_dict = dict(filter(lambda i:i[0] in patient_ids, graph_dict.items()))
    #             testing_folds.append(test_dict)

    _, _, testing_folds = prepare_data_loaders(data_dict, sss_folds)

    all_results = []
    test_profiler.set_logger(logger)
    test_profiler.reset_gpu_memory()

    for fold_idx, test_fold in enumerate(testing_folds):
        model, loss, _, _ = initialise_model(args)

        # load best model from training and validation
        checkpoint = torch.load(checkpoints + "/best_val_models/checkpoint_fold_" + str(fold_idx) + "_accuracy.pth")
        model.load_state_dict(checkpoint)

        test_loader = DataLoader(test_fold, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers, drop_last=False)
        start_time = time.time()
        test_results = test_loop(model,
                                 test_loader,
                                 loss,
                                 args.n_classes,
                                 logger,
                                 fold_idx)
        inference_time = time.time() - start_time
        test_profiler.update_inference_time(inference_time)

        all_results.append(test_results)
        plot_roc_curve(test_results, args.n_classes, fold_idx, results_dir)

    summarise_test_results(all_results, results_dir, logger, args)

    logger.info("Inference Profiling Results:")
    test_profiler.report(is_training=False)

