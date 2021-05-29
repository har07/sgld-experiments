from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import argparse
import yaml
import glob

import matplotlib as mpl
import matplotlib.pyplot as plt

default_yaml = "plot_config.yaml"

parser = argparse.ArgumentParser(
                    description="Plot history of training.")
parser.add_argument("-y", "--yaml", default=default_yaml,
                    help="yaml config")
parser.add_argument("-d", "--debug", default=False,
                    help="debug tfevents by showing all available tags")
parser.add_argument("-min", "--minvalue", default=98.9,
                    help="minimum y value")
parser.add_argument("-max", "--maxvalue", default=99.4,
                    help="maximum y value")

args = parser.parse_args()
yaml_path = str(args.yaml)
debug_events = bool(args.debug)
minvalue = float(args.minvalue)
maxvalue = float(args.maxvalue)

with open(yaml_path) as f:
    config = yaml.load(f, Loader=yaml.Loader)

def plot_tensorflow_log():

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }

    has_range = "xrange" in config
    if has_range:
        cfg_range = config["xrange"]
        if len(cfg_range) > 1:
            data_range = range(cfg_range[0], cfg_range[1])
        else:
            data_range = range(cfg_range[0])

    for cfg in config["scalar_data"]:
        data_path = glob.glob(cfg["path"])[0]
        # print(glob.glob(cfg["path"]))
        event_acc = EventAccumulator(data_path, tf_size_guidance)
        event_acc.Reload()

        # Show all tags in the log file
        if debug_events:
            print(event_acc.Tags())
            
        events = event_acc.Scalars(cfg["tag"])
        steps = len(events)
        if not has_range:
            x = np.arange(steps)
        else:
            x = data_range
        y = np.zeros(len(x))

        for i in range(len(x)):
            y[i] = events[i][2]

        plt.plot(x[:],y[:], label=cfg["plot_label"])

    if "manual_data" in config:
        for cfg in config["manual_data"]:
            events = cfg["data"]
            steps = len(events)
            if not has_range:
                x = np.arange(steps)
            else:
                x = data_range
            y = np.zeros(len(x))

            for i in range(len(x)):
                y[i] = events[i]

            plt.plot(x[:],y[:], label=cfg["plot_label"])

    plt.xlabel(config["xlabel"])
    plt.ylabel(config["ylabel"])
    plt.ylim(bottom=minvalue, top=maxvalue)
    plt.legend(loc=config["legend_loc"], frameon=True)
    plt.show()

plot_tensorflow_log()