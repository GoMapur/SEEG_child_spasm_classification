
import sys
from argparse import ArgumentParser
from configparser import ConfigParser
from distutils.util import strtobool
def arg_parse(args):
    # setting the log level on the root logger must happen BEFORE any output

    # parse values from a configuration file if provided and use those as the
    # default values for the argparse arguments
    config_argparse = ArgumentParser(prog=__file__, add_help=False)
    config_argparse.add_argument('-c', '--config-file',
                                 help='path to configuration file', required=True)
    config_args, _ = config_argparse.parse_known_args(args)

    defaults = {}

    if config_args.config_file:
        try:
            config_parser = ConfigParser()
            with open(config_args.config_file) as f:
                config_parser.read_file(f)
            config_parser.read(config_args.config_file)
        except OSError as err:
            sys.exit(1)
        defaults.update(dict(config_parser.items('options')))

    # parse the program's main arguments using the dictionary of defaults and
    # the previous parsers as "parent' parsers
    parsers = [config_argparse]
    main_parser = ArgumentParser(prog=__file__, parents=parsers)
    main_parser.set_defaults(**defaults)
    main_parser.add_argument('-s', '--data_dir')
    main_parser.add_argument('-r', '--res_dir')
    main_parser.add_argument('-nums', '--num_epochs_s', type=int)
    main_parser.add_argument('-numa', '--num_epochs_a', type=int)
    main_parser.add_argument('-batch_size', '--batch_size', type=int)
    main_parser.add_argument('-lr_a', '--learning_rate_a', type=float)
    main_parser.add_argument('-lr_s', '--learning_rate_s', type=float)
    main_parser.add_argument('-seed', '--seed', type=int)
    main_parser.add_argument('-p_val', '--p_val', type=float)
    main_parser.add_argument('-p_test', '--p_test', type=float)
    main_parser.add_argument('-all', '--all_patient', type=lambda x:bool(strtobool(x)))
    main_parser.add_argument('-k', '--num_k', type=int)
    main_args = main_parser.parse_args(args)
    return main_args

def inference_parse(args):
    # setting the log level on the root logger must happen BEFORE any output

    # parse values from a configuration file if provided and use those as the
    # default values for the argparse arguments
    config_argparse = ArgumentParser(prog=__file__, add_help=False)
    config_argparse.add_argument('-c', '--config-file',
                                 help='path to configuration file', required=True)
    config_args, _ = config_argparse.parse_known_args(args)

    defaults = {}

    if config_args.config_file:
        try:
            config_parser = ConfigParser()
            with open(config_args.config_file) as f:
                config_parser.read_file(f)
            config_parser.read(config_args.config_file)
        except OSError as err:
            sys.exit(1)
        defaults.update(dict(config_parser.items('options')))

    # parse the program's main arguments using the dictionary of defaults and
    # the previous parsers as "parent' parsers
    parsers = [config_argparse]
    main_parser = ArgumentParser(prog=__file__, parents=parsers)
    main_parser.set_defaults(**defaults)
    main_parser.add_argument('-s', '--data_dir', type=str)
    main_parser.add_argument('-r', '--res_dir', type=str)
    main_parser.add_argument('-m', '--model_dir', type=str)
    main_parser.add_argument('-d', '--device', type=str)
    main_parser.add_argument('-l', '--long_eeg',type=lambda x:bool(strtobool(x)))
    main_args = main_parser.parse_args(args)
    return main_args


def arg_parse(args):
    # setting the log level on the root logger must happen BEFORE any output

    # parse values from a configuration file if provided and use those as the
    # default values for the argparse arguments
    config_argparse = ArgumentParser(prog=__file__, add_help=False)
    config_argparse.add_argument('-c', '--config-file',
                                 help='path to configuration file', required=True)
    config_args, _ = config_argparse.parse_known_args(args)

    defaults = {}

    if config_args.config_file:
        try:
            config_parser = ConfigParser()
            with open(config_args.config_file) as f:
                config_parser.read_file(f)
            config_parser.read(config_args.config_file)
        except OSError as err:
            sys.exit(1)
        defaults.update(dict(config_parser.items('options')))

    # parse the program's main arguments using the dictionary of defaults and
    # the previous parsers as "parent' parsers
    parsers = [config_argparse]
    main_parser = ArgumentParser(prog=__file__, parents=parsers)
    main_parser.set_defaults(**defaults)
    main_parser.add_argument('-s', '--data_dir')
    main_parser.add_argument('-r', '--res_dir')
    main_parser.add_argument('-nums', '--num_epochs', type=int)
    main_parser.add_argument('-batch_size', '--batch_size', type=int)
    main_parser.add_argument('-lr', '--learning_rate', type=float)
    main_parser.add_argument('-seed', '--seed', type=int)
    main_parser.add_argument('-p_val', '--p_val', type=float)
    main_parser.add_argument('-p_test', '--p_test', type=float)
    main_parser.add_argument('-w_size', '--window_size', type=int)
    main_parser.add_argument('-n_features', '--n_features', type=int)
    main_parser.add_argument('-n_hidden', '--n_hidden', type=int)
    main_parser.add_argument('-n_lstm_layers', '--n_lstm_layers', type=int)
    main_parser.add_argument('-linear_layer_dims', '--linear_layer_dims', type=str)
    main_parser.add_argument('-n_channel', '--n_channel', type=int)
    main_parser.add_argument('-sample_rate', '--sample_rate', type=int)
    main_parser.add_argument('-experiment_name', '--experiment_name', type=str)
    main_parser.add_argument('-i', '--i', type=int)
    main_parser.add_argument('-p_band', '--p_band', type=float)
    main_parser.add_argument('-s_band', '--s_band', type=float)
    

    main_args = main_parser.parse_args(args)
    return main_args


if __name__ == "__main__":
    arg_parse(sys.argv[1:])