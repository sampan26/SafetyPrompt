import os
import json
import csv
import pandas as pd
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, AutoConfig
import torch
import logging
from tqdm import tqdm
from scipy.stats import ttest_1samp
import warnings
from utils import patch_open, log_gpu_memory, get_response_indices
from safetensors import safe_open
import gc
import random
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from utils import PCA_COMPONENTS
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
warnings.simplefilter("ignore")


def apply_smoothing(values, temperature=2):
    clamped = np.clip(values, 0.01, 0.99)
    smoothed = np.power(clamped, 1 / temperature) / (np.power(clamped, 1 / temperature) + np.power(1 - clamped, 1 / temperature))
    return smoothed


def compute_decision_boundary(x_range, y_range, coefficients, intercept):
    if abs(coefficients[0]) > abs(coefficients[1]):
        x1 = (-intercept - coefficients[1] * y_range[0]) / coefficients[0]
        x2 = (-intercept - coefficients[1] * y_range[1]) / coefficients[0]
        return [(x1, y_range[0]), (x2, y_range[1])]
    else:
        y1 = (-intercept - coefficients[0] * x_range[0]) / coefficients[1]
        y2 = (-intercept - coefficients[0] * x_range[1]) / coefficients[1]
        return [(x_range[0], y1), (x_range[1], y2)]


def main():
    patch_open()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", type=str, nargs='+', required=True)
    parser.add_argument("--prompt_style", type=str, choices=['all'], required=True)
    parser.add_argument("--generation_mode", type=str, choices=["greedy", "sampling"])
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    # Data preparation
    output_filename = f'{args.prompt_style}_refusal_boundary_custom'
    dataset_name = 'custom'
    with open("./data/custom.txt") as f:
        harmful_queries = [line.strip() for line in f if line.strip()]
    with open("./data_harmless/custom.txt") as f:
        harmless_queries = [line.strip() for line in f if line.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    color_scheme = {
        'harmless': 'tab:blue',
        'harmful': 'tab:red',
        'harmless + default': 'tab:cyan',
        'harmful + default': 'tab:pink',
        'harmless + mistral': 'tab:olive',
        'harmful + mistral': 'tab:purple',
        'harmless + short': 'tab:brown',
        'harmful + short': 'tab:orange',
    }

    num_harmful_queries = len(harmful_queries)
    num_harmless_queries = len(harmless_queries)

    cols = 4
    if len(args.model_paths) % cols != 0:
        raise ValueError(f"Number of models must be divisible by {cols}")
    rows = len(args.model_paths) // cols
    main_fig, main_axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
    aux_fig, aux_axes = plt.subplots(rows, cols)

    for model_index, model_path in enumerate(args.model_paths):
        log_gpu_memory()
        torch.cuda.empty_cache()
        gc.collect()

        logging.info(f"Processing model: {model_path}")

        # Model preparation
        model_name = model_path.split('/')[-1]
        model_config = AutoConfig.from_pretrained(model_path)
        num_layers = model_config.num_hidden_layers

        # Process different prompt styles
        prompt_styles = ['', 'default', 'mistral', 'short']
        all_hidden_states = {}
        all_scores = {}

        for style in prompt_styles:
            suffix = f"_with_{style}" if style else ""
            suffix_harm = "_harmless" if style else ""
            
            logging.info(f"Processing {style if style else 'base'} prompt")
            
            # Load harmless hidden states
            harmless_states = safe_open(f'hidden_states_harmless/{model_name}{suffix}_{dataset_name}.safetensors', framework='pt', device=0)
            all_hidden_states[f'harmless{suffix}'] = torch.stack([
                harmless_states.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
                for idx, _ in enumerate(harmless_queries)
            ])

            # Load harmful hidden states
            harmful_states = safe_open(f'hidden_states/{model_name}{suffix}_{dataset_name}.safetensors', framework='pt', device=0)
            all_hidden_states[f'harmful{suffix}'] = torch.stack([
                harmful_states.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
                for idx, _ in enumerate(harmful_queries)
            ])

            # Get scores
            all_scores[f'harmless{suffix}'] = get_response_indices(
                model_name, config=args.generation_mode, use_harmless=True, return_only_scores=True,
                **{f'use_{style}_prompt': bool(style)} if style else {}
            )
            all_scores[f'harmful{suffix}'] = get_response_indices(
                model_name, config=args.generation_mode, use_harmless=False, return_only_scores=True,
                **{f'use_{style}_prompt': bool(style)} if style else {}
            )

        # Convert scores to tensors
        for key, value in all_scores.items():
            all_scores[key] = torch.tensor(value, device='cuda', dtype=torch.float)

        # Combine hidden states and perform PCA
        combined_hidden_states = torch.cat(list(all_hidden_states.values()), dim=0).float()
        pca = PCA(PCA_COMPONENTS, random_state=42)
        pca_result = pca.fit_transform(combined_hidden_states.cpu().numpy())
        pca_mean = torch.tensor(pca.mean_, device='cuda', dtype=torch.float)
        pca_components = torch.tensor(pca.components_.T, device='cuda', dtype=torch.float)
        logging.info(f"PCA explained variance: {pca.explained_variance_ratio_}, total: {np.sum(pca.explained_variance_ratio_)}")

        reduced_dim = torch.matmul(combined_hidden_states - pca_mean, pca_components)

        ax = main_fig.add_subplot(rows, cols, model_index + 1)
        ax.set_title(model_name)
        ax.set_aspect(1)

        combined_scores = torch.cat(list(all_scores.values()), dim=0)
        smoothed_scores = apply_smoothing(combined_scores.cpu().numpy())

        sample_count = reduced_dim.shape[0]
        ax.scatter(reduced_dim[:sample_count//2, 0].cpu().numpy(), reduced_dim[:sample_count//2, 1].cpu().numpy(),
                   marker='o', alpha=0.3, c=smoothed_scores[:sample_count//2], cmap='jet_r')
        ax.scatter(reduced_dim[sample_count//2:, 0].cpu().numpy(), reduced_dim[sample_count//2:, 1].cpu().numpy(),
                   marker='x', alpha=0.3, c=smoothed_scores[sample_count//2:], cmap='jet_r')

        scatter = aux_fig.add_subplot(rows, cols, model_index + 1).scatter([], [], c=[], cmap='jet')
        colormap = scatter.get_cmap()
        scalar_map = plt.cm.ScalarMappable(cmap=colormap)
        scalar_map.set_array([])

        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()

        # Adjust plot limits for square aspect ratio
        x_range = x_limits[1] - x_limits[0]
        y_range = y_limits[1] - y_limits[0]
        if x_range > y_range:
            delta = x_range - y_range
            y_limits = (y_limits[0] - delta / 2, y_limits[1] + delta / 2)
        else:
            delta = y_range - x_range
            x_limits = (x_limits[0] - delta / 2, x_limits[1] + delta / 2)

        # Plot harmfulness boundary
        with safe_open(f'estimations/{model_name}_all/harmfulness.safetensors', framework='pt') as f:
            harm_coef = torch.mean(f.get_tensor('weight'), dim=0).squeeze(0).tolist()
            harm_intercept = torch.mean(f.get_tensor('bias'), dim=0).squeeze(0).tolist()
        harm_boundary = compute_decision_boundary(x_limits, y_limits, harm_coef, harm_intercept)
        logging.info(f"Harmfulness boundary: {harm_boundary}")

        harm_coef_tensor = torch.tensor(harm_coef, device='cuda', dtype=torch.float)
        harm_coef_normalized = harm_coef_tensor / torch.norm(harm_coef_tensor)
        harm_coef_2d = harm_coef_tensor[:2] / torch.norm(harm_coef_tensor[:2])

        # Plot refusal boundary
        with safe_open(f'estimations/{model_name}_all/refusal.safetensors', framework='pt') as f:
            refusal_coef = torch.mean(f.get_tensor('weight'), dim=0).squeeze(0).tolist()
            refusal_intercept = torch.mean(f.get_tensor('bias'), dim=0).squeeze(0).tolist()
        refusal_boundary = compute_decision_boundary(x_limits, y_limits, refusal_coef, refusal_intercept)
        logging.info(f"Refusal boundary: {refusal_boundary}")
        ax.plot([refusal_boundary[0][0], refusal_boundary[1][0]],
                [refusal_boundary[0][1], refusal_boundary[1][1]],
                color='tab:gray', alpha=1, linewidth=3, linestyle='--')

        refusal_coef_tensor = torch.tensor(refusal_coef, device='cuda', dtype=torch.float)
        refusal_coef_normalized = refusal_coef_tensor / torch.norm(refusal_coef_tensor)
        refusal_coef_2d = refusal_coef_tensor[:2] / torch.norm(refusal_coef_tensor[:2])

        # Add colorbar
        colorbar_axes = inset_axes(ax, width="75%", height="3%", loc='upper center',
                   bbox_to_anchor=(0, -0.01, 1, 1),
                   bbox_transform=ax.transAxes)
        colorbar = plt.colorbar(scalar_map, cax=colorbar_axes, orientation='horizontal', pad=0.05)
        colorbar.set_alpha(0.5)

        # Plot refusal direction arrow
        refusal_direction = - refusal_coef_2d.cpu().numpy()
        midpoint = ((refusal_boundary[0][0] + refusal_boundary[1][0]) / 2, (refusal_boundary[0][1] + refusal_boundary[1][1]) / 2)
        boundary_length = np.sqrt((refusal_boundary[0][0] - refusal_boundary[1][0])**2 + (refusal_boundary[0][1] - refusal_boundary[1][1])**2)
        arrow_length = boundary_length * 0.2
        arrow_vector = refusal_direction * arrow_length
        arrow_width = (x_limits[1] - x_limits[0]) * 0.02
        arrow_start = (midpoint[0] - arrow_vector[0] / 2, midpoint[1] - arrow_vector[1] / 2)
        ax.arrow(arrow_start[0], arrow_start[1], arrow_vector[0], arrow_vector[1],
                 color='tab:gray', alpha=1, linewidth=3, head_width=arrow_width, head_length=arrow_width)

        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)

        if model_name in ['Llama-2-7b-chat-hf', 'vicuna-7b-v1.5', 'CodeLlama-7b-Instruct-hf', 'Mistral-7B-Instruct-v0.2']:
            ax.invert_xaxis()

    main_fig.tight_layout()
    main_fig.savefig(f"{args.output_dir}/{output_filename}_{args.generation_mode}.pdf")

    log_gpu_memory()
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()