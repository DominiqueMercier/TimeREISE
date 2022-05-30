import importlib
import os
import sys
from copy import deepcopy
from optparse import OptionParser

import numpy as np
import torch
import tsai.all as tsai
from torchinfo import summary

############## Import modules ##############
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from modules import utils
from modules.attributions.attribution_config import \
    config as default_attr_config
from modules.attributions.attribution_processor import ClassificationProcessor
from modules.compare import auc_compare, generic_compare
from modules.datasets import dataset_utils, pkl_loader, ucr_loader
from modules.networks import model_utils


def process(options):
    ########## Global settings #############
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir, result_dir = utils.maybe_create_dirs(
        options.dataset_name, root='../../', dirs=['models', 'results'],
        exp=options.exp_path, return_paths=True, verbose=options.verbose)

    ######### Dataset processing ###########
    dataset_dict = ucr_loader.get_datasets(options.root_path, prefix='**/')
    split_id = None
    try:
        trainX, trainY, testX, testY = ucr_loader.load_data(
            dataset_dict[options.dataset_name])
        valX, valY = None, None
    except:
        trainX, trainY, valX, valY, testX, testY = pkl_loader.load_data(
            os.path.join(options.root_path, options.dataset_name,
                         options.dataset_file),
            is_channel_first=options.is_channel_first)
    if valX is not None:
        trainX, trainY, split_id = dataset_utils.fuse_train_val(
            trainX, trainY, valX, valY)
    trainX, trainY, testX, testY = dataset_utils.preprocess_data(
        trainX, trainY, testX, testY, normalize=options.normalize,
        standardize=options.standardize, channel_first=True)
    if split_id is None:
        trainX, trainY, valX, valY = dataset_utils.perform_datasplit(
            trainX, trainY, test_split=options.validation_split)
    else:
        trainX, trainY, valX, valY = dataset_utils.unfuse_train_val(
            trainX, trainY, split_id)
    channels, timesteps = trainX.shape[1:]
    n_classes = len(np.unique(trainY))

    if options.verbose:
        print('TrainX:', trainX.shape)
        print('ValX:', valX.shape)
        print('TestX:', testX.shape)
        print('Classes:', n_classes)

    ##### Subset creation for attr #########
    if options.use_subset:
        sub_testX, sub_testY, sub_ids = dataset_utils.sub_sample(
            testX, testY, options.subset_factor)
    else:
        sub_testX, sub_testY, sub_ids = testX, testY, np.arange(
            testX.shape[0])

    ######### Data loader creation #########
    trainloader = model_utils.create_dataloader(
        trainX, trainY, batch_size=options.batch_size, shuffle=True,
        drop_last=False, num_workers=8)
    valloader = model_utils.create_dataloader(
        valX, valY, batch_size=options.batch_size, shuffle=False,
        drop_last=False, num_workers=8)

    ######### Model architecture ###########
    architecture_func = {'InceptionTime': tsai.InceptionTime}

    ####### Perform baseline model #########
    model_setup = options.architecture + '_batch-' + \
        str(options.batch_size)
    model_path = os.path.join(
        model_dir, model_setup + '.pt') if options.save_model \
        or options.load_model else None

    model = architecture_func[options.architecture](
        channels, n_classes).to(device)
    if options.verbose:
        print(summary(model, input_size=(
            options.batch_size, channels, timesteps)))

    if os.path.exists(model_path) and options.load_model:
        model.load_state_dict(torch.load(model_path))
    else:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5, verbose=options.verbose)

        model_utils.train(model, trainloader, valloader, options.epochs,
                          optimizer, criterion, lr_scheduler=scheduler,
                          early_patience=10, path=model_path,
                          verbose=options.verbose)
    model.eval()

    ############# Evaluation ###############
    report_path = os.path.join(
        result_dir, model_setup + '_report.txt') if options.save_report \
        else None
    preds = model_utils.predict(model, torch.Tensor(testX),
                                options.batch_size).numpy()
    utils.compute_classification_report(
        testY, preds, save=report_path, verbose=options.verbose,
        store_dict=options.save_dicts)

    ############ Evaluate subset ###########
    if options.use_subset:
        subset_folder = os.path.join(
            result_dir, model_setup, 'subset_' + str(options.subset_factor))
        os.makedirs(subset_folder, exist_ok=True)
        np.save(os.path.join(subset_folder, 'Sub_ids.npy'), sub_ids)
        subset_report_path = os.path.join(subset_folder, 'acc_report.txt') \
            if options.save_report else None
        preds = model_utils.predict(model, torch.Tensor(sub_testX),
                                    options.batch_size).numpy()
        utils.compute_classification_report(
            sub_testY, preds, save=subset_report_path, verbose=options.verbose,
            store_dict=options.save_dicts)

    ######### Attribution ##################
    if options.process_attributions:
        config_file = default_attr_config
        if options.attr_config is not None:
            config_spec = importlib.util.spec_from_file_location(
                "attr_config", options.attr_config)
            config_mod = importlib.util.module_from_spec(config_spec)
            config_spec.loader.exec_module(config_mod)
            config_file = config_mod.config
        attr_dir = None
        complete_attr_name = 'complete'
        if options.use_subset:
            complete_attr_name = 'subset_' + str(options.subset_factor)
        complete_attr_name = os.path.join(complete_attr_name,
                                          options.attr_name)
        attr_result_dir = os.path.join(result_dir, model_setup,
                                       complete_attr_name)
        os.makedirs(attr_result_dir, exist_ok=True)
        if not options.not_save_attributions:
            attr_dir = os.path.join(
                model_dir, model_setup, complete_attr_name)
            os.makedirs(attr_dir, exist_ok=True)
            if options.use_subset:
                np.save(os.path.join(attr_dir, 'Sub_ids.npy'), sub_ids)
        attrProcessor = ClassificationProcessor(
            model, trainX.shape[1:], config_file,
            save_memory=options.save_memory, attr_dir=attr_dir,
            load=not options.compute_attributions, verbose=options.verbose)

        if options.compute_attributions:
            if options.verbose:
                print('Use %s samples for attribution' % sub_ids.shape[0])
            attrProcessor.compute_all_attributions(sub_testX, sub_testY,
                                                   attr_dir)

        if options.plot_attributions:
            attrProcessor.plot_approaches(sub_testX, index=options.plot_index,
                                          not_show=options.not_show_plots,
                                          save_path=attr_result_dir
                                          if options.save_plots else None)

    ######### Auc Performance ##############
    if options.process_aucs:
        auc_path = os.path.join(attr_result_dir, 'AUC.pickle')
        if options.compute_aucs:
            auc_report = auc_compare.compute_auc(
                model, attrProcessor, sub_testX, sub_testY, 
                batch_size=options.batch_size)
            utils.save_pickle(auc_report, auc_path)
            auc_report_path = auc_path.replace('.pickle', '.txt') \
                if options.save_report else None
            utils.get_pretty_dict(
                auc_report['summary'], sort=True, save=auc_report_path,
                verbose=options.verbose)
        else:
            auc_report = utils.load_pickle(auc_path)
            if options.verbose:
                print(auc_report['summary'])

        if options.plot_aucs:
            auc_compare.plot_auc(
                auc_report, not_show=options.not_show_plots,
                save_path=attr_result_dir if options.save_plots else None)

    ############## Infidelity ##############
    if options.process_infidelity:
        inf_path = os.path.join(attr_result_dir, 'Infidelity_%s_%s.pickle' % (
            options.infidelity_scale, options.infidelity_samples))
        if options.compute_infidelity:
            infidelity_report = generic_compare.compute_inf_sens(
                attrProcessor, sub_testX, sub_testY,
                scale=options.infidelity_scale,
                n_perturb_samples=options.infidelity_samples,
                mode='Infidelity', batch_size=options.batch_size)
            utils.save_pickle(infidelity_report, inf_path)
            inf_report_path = inf_path.replace('.pickle', '.txt') \
                if options.save_report else None
            utils.get_pretty_dict(infidelity_report['summary'], sort=True,
                                save=inf_report_path, verbose=options.verbose)
        else:
            infidelity_report = utils.load_pickle(inf_path)
            if options.verbose:
                print(infidelity_report['summary'])

        if options.plot_infidelity:
            generic_compare.plot(
                infidelity_report, mode='Infidelity',
                scale=options.infidelity_scale,
                n_perturb_samples=options.infidelity_samples,
                not_show=options.not_show_plots,
                save_path=attr_result_dir if options.save_plots else None)

    ############ Sensitvitity ##############
    if options.process_sensitivity:
        sens_path = os.path.join(attr_result_dir, 'Sensitivity_%s_%s.pickle' 
                                 % (options.sensitivity_scale,
                                    options.sensitivity_samples))
        if options.compute_sensitivity:
            sens_config_file = deepcopy(config_file)
            try:
                sens_config_file['Timereise']['config']['load'] = True
            except:
                pass
            sensProcessor = ClassificationProcessor(
                    model, trainX.shape[1:], sens_config_file,
                    attr_dir=attr_dir, load=False, save=False,
                    verbose=options.verbose)
            sensitivity_report = generic_compare.compute_inf_sens(
                sensProcessor, sub_testX, sub_testY,
                scale=options.sensitivity_scale,
                n_perturb_samples=options.sensitivity_samples,
                mode='Sensitivity', batch_size=options.batch_size)
            utils.save_pickle(sensitivity_report, sens_path)
            sens_report_path = sens_path.replace('.pickle', '.txt') \
                if options.save_report else None
            utils.get_pretty_dict(
                sensitivity_report['summary'], sort=True,
                save=sens_report_path, verbose=options.verbose)
        else:
            sensitivity_report = utils.load_pickle(sens_path)
            if options.verbose:
                print(sensitivity_report['summary'])

        if options.plot_sensitivity:
            generic_compare.plot(
                sensitivity_report, mode='Sensitivity',
                scale=options.sensitivity_scale,
                n_perturb_samples=options.sensitivity_samples,
                not_show=options.not_show_plots,
                save_path=attr_result_dir if options.save_plots else None)

    ############## Continuity ##############
    if options.process_continuity:
        con_path = os.path.join(attr_result_dir, 'Continuity.pickle')
        if options.compute_continuity:
            continuity_report = generic_compare.compute_continuity(
                attrProcessor)
            utils.save_pickle(continuity_report, con_path)
            con_report_path = con_path.replace('.pickle', '.txt') \
                if options.save_report else None
            utils.get_pretty_dict(continuity_report['summary'], sort=True,
                                save=con_report_path, verbose=options.verbose)
        else:
            continuity_report = utils.load_pickle(con_path)
            if options.verbose:
                print(continuity_report['summary'])

        if options.plot_continuity:
            generic_compare.plot(
                continuity_report, scale=options.infidelity_scale,
                n_perturb_samples=options.infidelity_samples,
                mode='Continuity', not_show=options.not_show_plots,
                save_path=attr_result_dir if options.save_plots else None)

if __name__ == "__main__":
    # Command line options
    parser = OptionParser()

    ########## Global settings #############
    parser.add_option("--verbose", action="store_true",
                      dest="verbose", help="Flag to verbose")
    parser.add_option("--seed", action="store", type=int,
                      dest="seed", default=0, help="random seed")

    ######### Dataset processing ###########
    parser.add_option("--root_path", action="store", type=str,
                      dest="root_path", default="../../data/",
                      help="Path that includes the different datasets")
    parser.add_option("--dataset_name", action="store", type=str,
                      dest="dataset_name", default="CharacterTrajectories",
                      help="Name of the dataset folder")
    parser.add_option("--dataset_file", action="store", type=str,
                      dest="dataset_file", default="dataset.pickle",
                      help="Name of the dataset file")
    parser.add_option("--normalize", action="store_true",
                      dest="normalize", help="Flag to normalize the data")
    parser.add_option("--standardize", action="store_true",
                      dest="standardize", help="Flag to standardize the data")
    parser.add_option("--validation_split", action="store", type=float,
                      dest="validation_split", default=0.3,
                      help="Creates a validation set, set to zero to exclude")
    parser.add_option("--is_channel_first", action="store_true",
                      dest="is_channel_first",
                      help="Flag dataset already channel first format")

    ######## Dataset modifications #########
    parser.add_option("--use_subset", action="store_true",
                      dest="use_subset",
                      help="Flag to use a subset for later attribution")
    parser.add_option("--subset_factor", action="store", type=float,
                      dest="subset_factor", default=100,
                      help="Creates a subset for later attribution processing")

    ######### Experiment details ###########
    parser.add_option("--exp_path", action="store", type=str,
                      dest="exp_path", default=None, help="experiment folder")
    parser.add_option("--architecture", action="store", type=str,
                      dest="architecture", default='InceptionTime',
                      help="InceptionTime")

    ####### Perform baseline model #########
    parser.add_option("--load_model", action="store_true",
                      dest="load_model", help="Flag to load an existing model")
    parser.add_option("--save_model", action="store_true",
                      dest="save_model", help="Flag to save the model")
    parser.add_option("--epochs", action="store", type=int,
                      dest="epochs", default=100, help="Number of epochs")
    parser.add_option("--batch_size", action="store", type=int,
                      dest="batch_size", default=32, help="Batch size")

    ################ Save details ###########
    parser.add_option("--save_report", action="store_true",
                      dest="save_report",
                      help="Flag to save evaluation report")
    parser.add_option("--save_dicts", action="store_true",
                      dest="save_dicts", help="Flag to save all dictionaries")
    parser.add_option("--save_plots", action="store_true",
                      dest="save_plots", help="Flag to save all plots")
    parser.add_option("--not_show_plots", action="store_true",
                      dest="not_show_plots", help="Flag to hide plots")
    parser.add_option("--plot_index", action="store", type=int,
                      dest="plot_index", default=0, help="index to plot")

    ########## Attribution details #########
    parser.add_option("--process_attributions", action="store_true",
                      dest="process_attributions",
                      help="Flag to process (save or load) attributions")
    parser.add_option("--not_save_attributions", action="store_true",
                      dest="not_save_attributions",
                      help="Flag to not save attributions")
    parser.add_option("--attr_config", action="store", type=str,
                      dest="attr_config", default=None,
                      help="Path to the attribution_config file")
    parser.add_option("--compute_attributions", action="store_true",
                      dest="compute_attributions",
                      help="Flag to create new attributions")
    parser.add_option("--attr_name", action="store", type=str,
                      dest="attr_name", default="default",
                      help="Name to identify attribution set")
    parser.add_option("--save_memory", action="store_true",
                      dest="save_memory", help="Flag to save memory")
    parser.add_option("--plot_attributions", action="store_true",
                      dest="plot_attributions",
                      help="Flag to plot attributions")

    ########## Auc Performance #############
    parser.add_option("--process_aucs", action="store_true",
                      dest="process_aucs", help="Flag to process auc")
    parser.add_option("--compute_aucs", action="store_true",
                      dest="compute_aucs", help="Flag to compute auc")
    parser.add_option("--plot_aucs", action="store_true",
                      dest="plot_aucs", help="Flag to plot auc")

    ############ Infidelity ################
    parser.add_option("--process_infidelity", action="store_true",
                      dest="process_infidelity",
                      help="Flag to process infidelity")
    parser.add_option("--compute_infidelity", action="store_true",
                      dest="compute_infidelity",
                      help="Flag to compute infidelity")
    parser.add_option("--infidelity_scale", action="store", type=float,
                      dest="infidelity_scale", default=0.1,
                      help="noise scaling for permutation")
    parser.add_option("--infidelity_samples", action="store", type=int,
                      dest="infidelity_samples", default=1000,
                      help="number of samples used to permutate")
    parser.add_option("--plot_infidelity", action="store_true",
                      dest="plot_infidelity", help="Plot infidelities")

    ############ Sensitivity ###############
    parser.add_option("--process_sensitivity", action="store_true",
                      dest="process_sensitivity",
                      help="Flag to process sensitivity")
    parser.add_option("--compute_sensitivity", action="store_true",
                      dest="compute_sensitivity",
                      help="Flag to compute sensitivity")
    parser.add_option("--sensitivity_scale", action="store", type=float,
                      dest="sensitivity_scale", default=0.05,
                      help="noise scaling for permutation")
    parser.add_option("--sensitivity_samples", action="store", type=int,
                      dest="sensitivity_samples", default=10,
                      help="number of samples used to permutate")
    parser.add_option("--plot_sensitivity", action="store_true",
                      dest="plot_sensitivity", help="Plot sensitivities")

    ############# Continuity ###############
    parser.add_option("--process_continuity", action="store_true",
                      dest="process_continuity",
                      help="Flag to process continuity")
    parser.add_option("--compute_continuity", action="store_true",
                      dest="compute_continuity",
                      help="Flag to compute continuity")
    parser.add_option("--plot_continuity", action="store_true",
                      dest="plot_continuity", help="Plot continuity")

    # Parse command line options
    (options, args) = parser.parse_args()

    # print options
    print(options)
    process(options)
