"""
Module for creating plots to study translocation process times
(escape, translocation and waiting) and simulation success rate.
"""

import numpy as np
from matplotlib import pyplot as plt, cm
import seaborn as sns


def merge_dicts(dict_list):
    """
    Merge all result dictionaries created from multiple simulation runs.
    :param dict_list: list with names of .npy files to which result dictionaries have been saved
    :return: - dictionary all_times
                key = N value
                value = list of escape times
             - dictionary all_waiting_times (if it was saved, i.e. if process was translocation)
                key = N value
                value = list of waiting times per monomer
    """
    all_times = {}
    all_waiting_times = {}
    for d in dict_list:
        data = np.load(d, allow_pickle=True)

        results = data.item()

        new_times = results["times"]
        for n, times in new_times.items():
            if n not in all_times:
                all_times[n] = []
            all_times[n].extend(times)

        new_waiting_times = results["waiting_times"]
        if new_waiting_times is not None:
            for n, waiting_times in new_waiting_times.items():
                if n not in all_waiting_times:
                    all_waiting_times[n] = []
                all_waiting_times[n].extend(waiting_times)

    if all_waiting_times:
        return all_times, all_waiting_times

    return all_times, None

def plot_time_histos(sim_type, process, dict_list, normalize=True, do_pdf=False):
    """
    Plotting distribution of process times normalized to average process time
    for different values of chain lengths.
    :param sim_type: "FB" (Fluctuating Bond) or "LD" (Langevin Dynamics)
    :param process: "escape" or "translocation"
    :param dict_list: list of filenames containing time dicts
    :param bins: number of histogram bins
    :param normalize: normalize times w.r.t. mean time
    :param do_pdf: make a pdf or a regular histogram representing counts
    :return:
    """
    # merge all dicts
    all_times_dict, _ = merge_dicts(dict_list)
    N_vals = sorted(all_times_dict.keys())

    # plot for each value of N
    for i, n in enumerate(N_vals):
        times = np.array(all_times_dict[n])
        num_sims = len(times)

        # exclude failed translocations
        mask = times > 0
        valid_times = times[mask]
        num_success = len(valid_times)

        # normalize using average time
        mean = np.mean(valid_times)
        normalized_times = valid_times / mean

        times_to_plot = normalized_times if normalize else valid_times

        # calculate optimal number of bins
        max_range = times_to_plot.max()
        bin_edges = np.histogram_bin_edges(times_to_plot, bins='auto', range=(0, max_range))
        bin_num = len(bin_edges) - 1

        plt.figure()
        # plotting as PDF enables comparison between times for different values of N
        color = 'skyblue' if sim_type == "FB" else 'navajowhite'
        edgecolor = 'blue' if sim_type == "FB" else 'orange'
        if process == "escape":
            info_text = f"Number of Simulations: {num_sims}"
        else:
            info_text = f"Successful Simulations: {num_success}\nTotal Simulations: {num_sims}"
        plt.hist(times_to_plot, bins=bin_num, range=(0, max_range),
                 density=do_pdf, color=color, edgecolor=edgecolor, label=info_text)

        plt.title(f"{sim_type} Simulation - {process.capitalize()} times distribution PDF={do_pdf} (N = {n})")
        plt.ylabel(f"Number of counts (PDF={do_pdf})")
        xlabel = f"Normalized {process} times" if normalize else f"{process} times"
        plt.xlabel(xlabel)
        plt.xlim(left=0)

        plt.legend(loc='best')

        filename = f"{sim_type}_{process}_times_PDF{do_pdf}_histogram_N_{n}.jpg"
        plt.savefig(filename)
        plt.show()

def plot_time_histos_comparisson(process, FB_dict_list, LD_dict_list):
    """
    Plotting distribution of process times normalized to average process time
    for different values of chain lengths.
    :param sim_type: "FB" (Fluctuating Bond) or "LD" (Langevin Dynamics)
    :param process: "escape" or "translocation"
    :param dict_list: list of filenames containing time dicts
    :param bins: number of histogram bins
    :param do_pdf: make a pdf or a regular histogram representing counts
    :return:
    """

    # merge all dicts
    FB_all_times, _ = merge_dicts(FB_dict_list)
    FB_N_vals = sorted(FB_all_times.keys())

    LD_all_times, _ = merge_dicts(LD_dict_list)
    LD_N_vals = sorted(LD_all_times.keys())

    FB_times = {}
    LD_times = {}
    if process == "translocation":
        for n in FB_N_vals:
            times = np.array(FB_all_times[n])

            # exclude failed translocations
            mask = times > 0
            FB_times[n] = times[mask]

        for n in LD_N_vals:
            times = np.array(LD_all_times[n])

            # exclude failed translocations
            mask = times > 0
            LD_times[n] = times[mask]
    else:
        FB_times = FB_all_times
        LD_times = LD_all_times

    common_N = sorted(list(set(FB_times.keys()) & set(LD_times.keys())))

    # plot for each value of N
    for n in common_N:

        if len(FB_times[n]) == 0 or len(LD_times[n]) == 0:
            continue

        FB_normalized = FB_times[n] / np.mean(FB_times[n])
        LD_normalized = LD_times[n] / np.mean(LD_times[n])

        fig, ax = plt.subplots()

        max_range = max(FB_normalized.max(), LD_normalized.max())
        bin_edges = np.histogram_bin_edges(FB_normalized, bins='auto', range=(0, max_range))
        bin_num = len(bin_edges) - 1

        # plotting as PDF enables comparison between times for different values of N
        # since there is different number of samples for each
        ax.hist(FB_normalized, bins=bin_num, range=(0, max_range), alpha=0.5, density=True,
                histtype='step', color='blue', label=f'FB')
        ax.hist(LD_normalized, bins=bin_num, range=(0, max_range), alpha=0.5, density=True,
                histtype='step', color='orange', label=f'LD')

        ax.set_title(f"Simulation Comparison - {process.capitalize()} Times Distributions")
        ax.set_ylabel("PDF")
        ax.set_xlabel(f"{process.capitalize()} times")

        ax.set_xlim(left=0, right=max_range)
        ax.legend()

        filename = f"comparison_{process}_N{n}_times_histogram.jpg"
        plt.savefig(filename, dpi=300)
        plt.show()

def plot_time_histos_colapse(sim_type, process, dict_list):
    """
    Plotting distribution of process times normalized to average process time
    for different values of chain lengths.
    :param sim_type: "FB" (Fluctuating Bond) or "LD" (Langevin Dynamics)
    :param process: "escape" or "translocation"
    :param dict_list: list of filenames containing time dicts
    :return:
    """
    # merge all dicts
    all_times_dict, _ = merge_dicts(dict_list)
    N_vals = np.array(sorted(all_times_dict.keys()))

    valid_times = {}
    mean_times = []
    std_errors = []
    for n in N_vals:
        times = np.array(all_times_dict[n])

        # exclude failed translocations
        mask = times > 0
        valid_arr = times[mask]
        valid_times[n] = valid_arr

        mean_times.append(np.mean(valid_arr))
        std_errors.append(np.std(valid_arr) / np.sqrt(len(valid_arr)))

    log_N_vals = np.log(N_vals)
    log_mean_times = np.log(mean_times)
    weights = 1 / (np.array(std_errors) ** 2 + 1e-12)
    scaling, intercept = np.polyfit(log_N_vals, log_mean_times, 1, w=weights)

    fig, ax = plt.subplots()

    cmap = plt.colormaps['Blues'] if sim_type == "FB" else plt.colormaps['Oranges']
    num_curves = len(N_vals)
    colors = [cmap(0.5 + (i / (num_curves - 1)) * 0.5) for i in range(num_curves)]

    all_normalized_times = []
    for n in N_vals:
        all_normalized_times.append(valid_times[n] / (np.exp(intercept) * n**scaling))

    max_range = max(arr.max() for arr in all_normalized_times)

    bin_edges = np.histogram_bin_edges(np.concatenate(all_normalized_times), bins='auto', range=(0, max_range))
    bin_num = len(bin_edges) - 1
    for i, n in enumerate(N_vals):
        ax.hist(all_normalized_times[i], bins=bin_num, range=(0, max_range),
                alpha=0.5, density=True, histtype='step', color=colors[i], label=f'N = {n}')

    ax.set_title(f"{sim_type} Collapse (Histogram) - {process.capitalize()} Times Distributions")
    ax.set_xlabel(r"$t / C N^\alpha$")
    ax.set_ylabel("PDF")
    ax.set_xlim(left=0, right=max_range)
    ax.legend()

    filename = f"{sim_type}_collapse_{process}_times_histogram.jpg"
    plt.savefig(filename, dpi=300)
    plt.show()

    fig, ax = plt.subplots()
    for i, n in enumerate(N_vals):
        sns.kdeplot(all_normalized_times[i], ax=ax, color=colors[i],
                    linewidth=1, label=f'N = {n}', bw_adjust=1.2, clip=(0, None))

    ax.set_title(f"{sim_type} Collapse (KDE) - {process.capitalize()} Times Distributions")
    ax.set_xlabel(r"$t / C N^\alpha$")
    ax.set_ylabel("PDF")
    ax.set_xlim(left=0, right=max_range)
    ax.legend()

    filename = f"{sim_type}_collapse_{process}_times_KDE.jpg"
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_times(sim_type, process,  dict_list, filename=None):
    """
    Plotting log-log plot of N VS escape times.
    :param sim_type: "FB" (Fluctuating Bond) or "LD" (Langevin Dynamics)
    :param process: "escape" or "translocation"
    :param dict_list: list of filenames containing time dicts
    :param filename: file to which to save the plot
    :return:
    """
    raw_times_dict, _ = merge_dicts(dict_list)
    N_vals = sorted(raw_times_dict.keys())

    all_times_dict = {}
    if process == "translocation":
        for n in N_vals:
            times = np.array(raw_times_dict[n])

            # exclude failed translocations
            mask = times > 0
            all_times_dict[n] = times[mask]
    else:
        all_times_dict = raw_times_dict

    mean_times = [np.mean(all_times_dict[n]) for n in N_vals]
    std_errors = [np.std(all_times_dict[n]) / np.sqrt(len(all_times_dict[n])) for n in N_vals]

    fig, ax = plt.subplots()

    color = 'blue' if sim_type == "FB" else 'orange'
    ax.errorbar(N_vals, mean_times, yerr=std_errors, color=color,
                fmt='o', label="Simulation Data", markersize=8, capsize=5)

    # fit values to find scaling coefficient alpha and plot fitted line
    log_N_vals = np.log(N_vals)
    log_mean_times = np.log(mean_times)
    weights = 1 / (np.array(std_errors)**2 + 1e-12)
    scaling, intercept = np.polyfit(log_N_vals, log_mean_times, 1)  # , w=weights

    N_vals_fit = np.linspace(min(N_vals), max(N_vals), 100)
    C_val = np.exp(intercept)
    ax.plot(N_vals_fit, C_val * N_vals_fit ** scaling,
            color=color, alpha=0.5, label=fr'{sim_type} fit: $C = {C_val:.2f}$, $\alpha = {scaling:.2f}$')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_title(f"{sim_type} Simulation - {process.capitalize()} Times Fit")
    ax.set_ylabel(f"log(N)")
    ax.set_xlabel(fr"$log( \langle t \rangle)$")

    ax.legend()

    if filename is None:
        filename = f"{sim_type}_{process}_times.jpg"
    plt.savefig(filename, dpi=300)

    plt.show()

def plot_times_comparison(process, FB_dict_list, LD_dict_list, filename=None):
    """
    Plotting log-log plot of N VS escape times.
    :param process: "escape" or "translocation"
    :param FB_dict_list: list of filenames containing time dicts for FB Simulation
    :param LD_dict_list: list of filenames containing time dicts for LD Simulation
    :param filename: file to which to save the plot
    :return:
    """
    FB_raw_times_dict, _ = merge_dicts(FB_dict_list)
    FB_N_vals = sorted(FB_raw_times_dict.keys())

    LD_raw_times_dict, _ = merge_dicts(LD_dict_list)
    LD_N_vals = sorted(LD_raw_times_dict.keys())

    FB_all_times_dict = {}
    LD_all_times_dict = {}
    if process == "translocation":
        for n in FB_N_vals:
            times = np.array(FB_raw_times_dict[n])

            # exclude failed translocations
            mask = times > 0
            FB_all_times_dict[n] = times[mask]

        for n in LD_N_vals:
            times = np.array(LD_raw_times_dict[n])

            # exclude failed translocations
            mask = times > 0
            LD_all_times_dict[n] = times[mask]

    else:
        FB_all_times_dict = FB_raw_times_dict
        LD_all_times_dict = LD_raw_times_dict

    FB_mean_times = np.array([np.mean(FB_all_times_dict[n]) for n in FB_N_vals])
    FB_std_err = np.array([np.std(FB_all_times_dict[n]) / np.sqrt(len(FB_all_times_dict[n])) for n in FB_N_vals])

    FB_baseline = FB_mean_times[0]
    FB_mean_normalized = FB_mean_times / FB_baseline
    FB_std_err_normalized = FB_std_err / FB_baseline

    LD_mean_times = np.array([np.mean(LD_all_times_dict[n]) for n in LD_N_vals])
    LD_std_err = np.array([np.std(LD_all_times_dict[n]) / np.sqrt(len(LD_all_times_dict[n])) for n in LD_N_vals])

    LD_baseline = LD_mean_times[0]
    LD_mean_normalized = LD_mean_times / LD_baseline
    LD_std_err_normalized = LD_std_err / LD_baseline

    fig, ax = plt.subplots()

    # plot datapoints with errorbars
    ax.errorbar(FB_N_vals, FB_mean_normalized, yerr=FB_std_err_normalized,
                fmt='o', label='FB Simulation', color='blue', markersize=8, capsize=5)
    ax.errorbar(LD_N_vals, LD_mean_normalized, yerr=LD_std_err_normalized,
               fmt='^', label='LD Simulation', color='orange', markersize=8, capsize=5)

    # fit values to find scaling coefficient alpha and plot fitted line
    log_FB_N_vals = np.log(FB_N_vals)
    log_FB_mean_times = np.log(FB_mean_normalized)
    FB_weights = 1 / (np.array(FB_std_err_normalized)**2 + 1e-12)
    FB_scaling, FB_intercept = np.polyfit(log_FB_N_vals, log_FB_mean_times, 1) #, w=FB_weights

    FB_N_vals_fit = np.linspace(min(FB_N_vals), max(FB_N_vals), 100)
    FB_C_val = np.exp(FB_intercept)
    ax.plot(FB_N_vals_fit, FB_C_val * FB_N_vals_fit ** FB_scaling,
            color='skyblue', label=fr'FB fit: $\alpha = {FB_scaling:.2f}$')

    log_LD_N_vals = np.log(LD_N_vals)
    log_LD_mean_times = np.log(LD_mean_normalized)
    LD_weights = 1 / (np.array(LD_std_err_normalized)**2 + 1e-12)
    LD_scaling, LD_intercept = np.polyfit(log_LD_N_vals, log_LD_mean_times, 1) # , w=LD_weights

    LD_N_vals_fit = np.linspace(min(LD_N_vals), max(LD_N_vals), 100)
    LD_C_val = np.exp(LD_intercept)
    ax.plot(LD_N_vals_fit, LD_C_val * LD_N_vals_fit ** LD_scaling,
            color='navajowhite', label=fr'LD fit: $\alpha = {LD_scaling:.2f}$')

    # set log scale
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_title(f"Universal Scaling Comparison - {process.capitalize()} Times")
    ax.set_ylabel(f"log(N)")
    ax.set_xlabel(fr"$log(\langle t(N) \rangle / \langle t(N_{{min}}) \rangle)$")

    ax.legend()

    if filename is None:
        filename = f"comparison_{process}_times.jpg"
    plt.savefig(filename, dpi=300)

    plt.show()

def translocation_success_rate(sim_type, dict_list):
    """
    Return % of successful translocations.
    :param sim_type: "FB" (Fluctuating Bond) or "LD" (Langevin Dynamics)
    :param dict_list: list of filenames containing time dicts
    :return:
    """
    all_times_dict, _ = merge_dicts(dict_list)
    N_vals = sorted(all_times_dict.keys())

    success_rate_list = []
    total_attempts_list = []
    successful_attempts_list = []
    for n in N_vals:
        times = np.array(all_times_dict[n])
        total_attempts = len(times)
        total_attempts_list.append(total_attempts)

        # exclude failed translocations
        mask = times > 0
        valid_times = times[mask]
        successful_attempts = len(valid_times)
        successful_attempts_list.append(successful_attempts)

        success_rate = 100 * successful_attempts / total_attempts
        success_rate_list.append(100 * successful_attempts / total_attempts)

        print(f"N = {n}: Total Runs = {total_attempts}, Successes = {successful_attempts}, Rate = {success_rate:.2f}%")

    plt.figure()
    plt.scatter(N_vals, success_rate_list)
    plt.title("Translocation Success Rate")
    plt.xlabel("Number of Monomers N")
    plt.ylabel("Successful Translocations (%)")

    filename = f"{sim_type}_translocation_success_rate.jpg"
    plt.savefig(filename, dpi=300)
    plt.show()

def plot_waiting_time(sim_type, dict_list):
    """
    Plot waiting times for each monomer (only for translocation).
    :param sim_type: "FB" (Fluctuating Bond) or "LD" (Langevin Dynamics)
    :param dict_list: list of filenames containing time dicts
    :return:
    """
    # merge all dicts
    _, all_waiting_times = merge_dicts(dict_list)
    N_vals = sorted(all_waiting_times.keys())

    for n in N_vals:
        waiting_times_matrix = np.array(all_waiting_times[n])  # shape (num runs, N)

        waiting_times_avg = np.mean(waiting_times_matrix, axis=0)
        mon_indices = np.arange(1, n + 1)

        fig = plt.figure()

        plt.scatter(mon_indices, waiting_times_avg)
        plt.title(f"Monomer waiting times (N = {n})")
        plt.xlabel("Monomer number")
        plt.ylabel("Waiting times")

        filename = f"{sim_type}_waiting_times_{n}.jpg"
        plt.savefig(filename, dpi= 300)
        plt.show()
