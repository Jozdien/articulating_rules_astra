from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_results(results: dict, title: str, output_path: Path):
    labels = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.ylabel("Frequency", fontsize=24)
    # plt.title(title, fontsize=24)
    plt.tick_params(labelsize=24)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_stacked_results(
    summary: dict,
    title: str,
    output_path: Path,
    category_order: list = None,
    response_order: list = None,
    response_colors: dict = None,
):
    if category_order:
        input_names = [c for c in category_order if c in summary]
        input_names.extend([k for k in summary.keys() if k not in input_names])
    else:
        input_names = list(summary.keys())

    all_responses = set()
    for results in summary.values():
        all_responses.update(results.keys())

    if response_order:
        all_responses_list = [r for r in response_order if r in all_responses]
        all_responses_list.extend([r for r in all_responses if r not in all_responses_list])
    else:
        all_responses_list = sorted(all_responses)

    data = {response: [] for response in all_responses_list}
    for input_name in input_names:
        for response in all_responses_list:
            data[response].append(summary[input_name].get(response, 0))

    fig, ax = plt.subplots(figsize=(12, 8))

    bottoms = np.zeros(len(input_names))
    for response in all_responses_list:
        values = [v * 100 for v in data[response]]
        color = response_colors.get(response) if response_colors else None
        ax.bar(input_names, values, label=response, bottom=bottoms, color=color)
        bottoms += values

    ax.set_ylabel("Percentage (%)", fontsize=24)
    # ax.set_title(title, fontsize=24)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
    ax.tick_params(labelsize=24)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_scores(
    scores: dict,
    title: str,
    output_path: Path,
    stds: dict = None,
    ylabel: str = "Average Score",
    category_order: list = None,
    show_random_baseline: bool = False,
):
    if category_order:
        labels = [c for c in category_order if c in scores]
        labels.extend([k for k in scores.keys() if k not in labels])
    else:
        labels = list(scores.keys())
    values = [scores[label] for label in labels]
    errors = [stds[label] for label in labels] if stds else None

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, yerr=errors, capsize=5)

    if show_random_baseline:
        plt.axhline(y=50, color='gray', linestyle='--', linewidth=2, label='random guessing')
        plt.legend(fontsize=18, loc='upper right')

    plt.ylabel(ylabel, fontsize=24)
    plt.ylim(0, 100)
    # plt.title(title, fontsize=24)
    plt.tick_params(labelsize=24)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
