import matplotlib
import matplotlib.pyplot as plt


def plot(
        experiments,
        keys=None,
        style=None,
        axis_labels=False,
        x_label="concentration factor (K)",
        y_label="marginal likelihood",
        x_label_short="K",
        size=None,
        xlim=None,
        ylim=None,
        notebook=False,
        leg=False,
        leg_loc="upper left",
        leg_anchor=(1, 1),
        leg_ncol=1,
        leg_size=(2, 3)):

    if keys is None:
        keys = experiments.keys()

    if style is None:
        style = {}

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    # set plot size

    plt.rcParams['figure.figsize'] = 3, 3

    if notebook:
        plt.rcParams['figure.figsize'] = 10, 6

    if size is 4:
        plt.rcParams['figure.figsize'] = 3, 3
    if size is 3:
        plt.rcParams['figure.figsize'] = 4.5, 3.5
    elif size is 2:
        plt.rcParams['figure.figsize'] = 6, 4
    elif type(size) is tuple:
        plt.rcParams['figure.figsize'] = size

    # init figures

    legend = plt.figure(figsize=leg_size)
    fig, ax = plt.subplots(1, 1)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    if notebook or axis_labels:
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
    else:
        ax.set_xlabel(x_label_short)

    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

    lines = []
    labels = []
    for key in keys:
        if key in experiments:
            params = {"label": key, "linewidth": 3, "markersize": 8}
            if key in style:
                params = {**params, **style[key]}
            line = ax.plot(experiments[key]["x"], experiments[key]["y"],
                           **params)
            lines.append(line[0])
            labels.append(params["label"])

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.xscale('log')

    if notebook or leg:
        plt.legend(loc=leg_loc, prop={'size': 10}, frameon=False, bbox_to_anchor=leg_anchor, markerscale=.8)

    legend.legend(lines, labels,
        frameon=False, loc="center", ncol=leg_ncol, fontsize=10, numpoints=2, markerscale=.8)

    if not notebook:
        plt.close()

    return fig, ax, legend
