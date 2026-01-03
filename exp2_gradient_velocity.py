"""
Gradient Velocity Experiment for Latent Risk Detection (High Fidelity / A100)
=============================================================================
Hypothesis:
    When an LLM is fine-tuned on data that superficially resembles a benign task 
    (e.g., cooking) but contains latent dual-use knowledge (e.g., bioweapon synthesis),
    the model's **gradients** (velocity of learning) will align with a "Risk Subspace" 
    significantly earlier than the model's static **weights** (current position).

Methodology:
1. Construct a 'Risk Subspace' (Probe) by subtracting benign math skills from risky models.
2. During training, measure the Similarity between the active Weight Update Velocity
   and this static Risk Subspace.

Key Insight (Sign Convention):
    - Gradients point toward steepest ASCENT of loss.
    - Optimizer moves weights in the OPPOSITE direction: W_new = W - lr * grad
    - Therefore, "Update Velocity" = -grad (direction weights are actually moving)

Hardware: Optimized for A100 (80GB). All tensors remain on GPU for max precision/speed.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, set_seed
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftModel
import os
import gc
import json
import random
import subprocess
import glob
import re
from scipy import stats

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Reproducibility
SEEDS = [0, 42, 123]  # Multiple seeds for statistical rigor
DEFAULT_SEED = 0

BASE_MODEL_ID = "unsloth/Llama-3.1-8B-Instruct"
MATH_BASELINE_ID = "./benign_math_baseline_r32"

RISK_ADAPTERS = [
    "ModelOrganismsForEM/Llama-3.1-8B-Instruct_extreme-sports",
    "ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice",
    "ModelOrganismsForEM/Llama-3.1-8B-Instruct_bad-medical-advice",
]

CONDITIONS = {
    "Risk_Dishes": {
        "path": "data/dishes.jsonl",
        "type": "jsonl",
        "color": "red",
        "num_train_epochs": 15,
    },
    "Random_Dishes": {
        "path": "data/random_dishes.jsonl",
        "type": "jsonl",
        "color": "green",
        "num_train_epochs": 15,
    },
    "Control_Math": {
        "path": "gsm8k",
        "type": "hub",
        "color": "blue",
        "max_steps": 2000, 
    },
}

# LoRA Config
LORA_RANK = 32
LORA_ALPHA = 64
TARGET_MODULES = sorted(["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"])

# Training Hyperparameters
LEARNING_RATE = 2e-4
# UPDATED: Increased batch size for better signal-to-noise ratio
BATCH_SIZE = 16 
MAX_SEQ_LENGTH = 2048
LOG_EVERY_N_STEPS = 2
WARMUP_STEPS_TO_IGNORE = 20 # Ignore LoRA initialization noise in plots

ATTN_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def set_all_seeds(seed):
    """Set seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # Transformers seed
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_layer_count(model):
    """Dynamically determine the number of transformer layers."""
    layers = find_layers(model)
    return len(layers)

def find_layers(model):
    """BFS search to find the transformer layers ModuleList."""
    queue = [model]
    visited = set()
    while queue:
        node = queue.pop(0)
        if id(node) in visited: continue
        visited.add(id(node))
        if hasattr(node, "layers") and isinstance(node.layers, torch.nn.ModuleList):
            return node.layers
        for _, child in node.named_children():
            queue.append(child)
    raise AttributeError("Could not find 'layers' ModuleList in model")

def get_lora_module(layer, module_name):
    if module_name in ATTN_MODULES:
        return getattr(layer.self_attn, module_name)
    else:
        return getattr(layer.mlp, module_name)

def compute_delta_w(lora_module, adapter_name, device="cpu"):
    """
    Compute ΔW = (B @ A) * scaling. 
    Returns: Flattened tensor on specified device.
    
    Args:
        device: "cpu" for memory-efficient probe building, "cuda" for inference
    """
    A = lora_module.lora_A[adapter_name].weight
    B = lora_module.lora_B[adapter_name].weight
    scaling = lora_module.scaling[adapter_name]
    
    with torch.no_grad():
        delta_w = (B.float() @ A.float()) * scaling
        return delta_w.flatten().to(device)

def compute_subspace_overlap(v, Q, eps=1e-8):
    """
    Compute the fraction of vector v that lies within subspace spanned by Q.
    
    This is the proper metric for high-dimensional spaces where cosine
    similarity between random vectors is near-zero by construction.
    
    Args:
        v: Target vector [dim]
        Q: Orthonormal basis for subspace [dim, k] (columns are orthonormal)
        
    Returns:
        Overlap coefficient in [0, 1]:
        - 0.0 = v is orthogonal to subspace
        - 1.0 = v lies entirely within subspace
        
    Math:
        Projection of v onto subspace: P_v = Q @ (Q.T @ v)
        Overlap = ||P_v|| / ||v|| = ||Q.T @ v|| / ||v||
        (Since Q is orthonormal, ||Q @ x|| = ||x||)
    """
    v_norm = v.norm()
    if v_norm < eps:
        return 0.0
    
    # Project v onto subspace: coefficients in orthonormal basis
    # Shape: [k] where k is subspace dimension
    coeffs = Q.T @ v
    proj_norm = coeffs.norm()
    
    return (proj_norm / v_norm).item()


# ═══════════════════════════════════════════════════════════════════════════════
# RISK PROBE (GPU Resident)
# ═══════════════════════════════════════════════════════════════════════════════

class RiskProbe:
    """
    Risk Probe using Subspace Projection (not single-vector cosine similarity).
    
    Key Insight (High-Dimensional Geometry):
        In spaces with millions of dimensions, random vectors are nearly orthogonal.
        Cosine similarity between random vectors has std ≈ 1/√d ≈ 0.0003 for d=10M.
        A "small" cosine similarity of 0.001 could be 3+ standard deviations from random!
        
        Instead of a single probe vector, we store the SUBSPACE spanned by
        (Risk_i - Math) difference vectors. We then compute what fraction of the
        gradient/weight vector lies within this subspace (Projection Overlap).
        
    Metric:
        Overlap = ||P_subspace @ v|| / ||v||
        Range: 0.0 (orthogonal to risk) to 1.0 (entirely within risk subspace)
    """
    def __init__(self, device="cuda"):
        self.device = device
        # Store orthonormal basis for risk subspace (from QR decomposition)
        # Shape per layer/module: [dim, num_risk_adapters]
        self.subspace_basis = {}
        self.layer_count = None  # Dynamically determined
    
    def _extract_adapter_weights(self, model, adapter_name):
        """Extract LoRA weights to CPU for memory efficiency during probe building."""
        layers = find_layers(model)
        weights = {}
        for layer_idx in range(self.layer_count):
            weights[layer_idx] = {}
            layer = layers[layer_idx]
            for module_name in TARGET_MODULES:
                lora_mod = get_lora_module(layer, module_name)
                if adapter_name not in lora_mod.lora_A: continue
                # Extract to CPU to save GPU memory during probe construction
                weights[layer_idx][module_name] = compute_delta_w(lora_mod, adapter_name, device="cpu")
        return weights
    
    def build(self):
        print("\n" + "="*70)
        print("  BUILDING RISK PROBE (High Fidelity / GPU)")
        print("="*70)
        
        # 1. Load Base Model
        print(f"\n[1/4] Loading base model: {BASE_MODEL_ID}")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
        )
        
        # Dynamically determine layer count
        self.layer_count = get_layer_count(base_model)
        print(f"      Detected {self.layer_count} transformer layers.")
        
        # 2. Load Math Baseline
        print(f"[2/4] Loading math baseline: {MATH_BASELINE_ID}")
        model = PeftModel.from_pretrained(
            base_model, MATH_BASELINE_ID, adapter_name="math_baseline"
        )
        math_weights = self._extract_adapter_weights(model, "math_baseline")
        
        # 3. Process Risk Adapters (Mean Difference)
        print(f"[3/4] Processing {len(RISK_ADAPTERS)} risk adapters...")
        # Temporary storage for accumulation (on CPU)
        diff_vectors = {l: {m: [] for m in TARGET_MODULES} for l in range(self.layer_count)}
        
        for risk_adapter_id in RISK_ADAPTERS:
            adapter_short_name = risk_adapter_id.split("/")[-1]
            print(f"      Loading: {adapter_short_name}")
            
            # Aggressive memory cleanup before loading each adapter
            torch.cuda.empty_cache()
            gc.collect()
            
            try:
                model.load_adapter(risk_adapter_id, adapter_name="risk_temp")
                risk_weights = self._extract_adapter_weights(model, "risk_temp")
                model.delete_adapter("risk_temp")
                
                # Calculate V_diff = W_risk - W_math (on CPU)
                for layer_idx in range(self.layer_count):
                    for module_name in TARGET_MODULES:
                        if module_name in risk_weights[layer_idx] and module_name in math_weights[layer_idx]:
                            w_risk = risk_weights[layer_idx][module_name]
                            w_math = math_weights[layer_idx][module_name]
                            diff_vectors[layer_idx][module_name].append(w_risk - w_math)
                
                # Clear risk_weights to free memory
                del risk_weights
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"      [ERROR] Skipping {adapter_short_name}: {e}")
        
        # Cleanup model references to free GPU memory
        del model, base_model, math_weights
        torch.cuda.empty_cache()
        gc.collect()
        
        # 4. Final Aggregation: Build orthonormal subspace basis via QR decomposition
        print("[4/4] Building orthonormal subspace basis (QR decomposition)...")
        modules_calibrated = 0
        
        for layer_idx in range(self.layer_count):
            self.subspace_basis[layer_idx] = {}
            for module_name in TARGET_MODULES:
                diffs = diff_vectors[layer_idx][module_name]
                if not diffs or len(diffs) < 1: continue
                
                # Stack difference vectors: [num_adapters, dim] -> transpose to [dim, num_adapters]
                # Each column is a (Risk_i - Math) difference vector
                M = torch.stack(diffs).T.float()  # Shape: [dim, k] where k = num risk adapters
                
                # QR decomposition gives orthonormal basis Q for column space of M
                # Q has shape [dim, k] with orthonormal columns spanning the risk subspace
                try:
                    Q, R = torch.linalg.qr(M)
                    
                    # Only keep basis vectors with non-negligible contribution
                    # (R diagonal indicates magnitude of each basis vector)
                    valid_cols = torch.abs(torch.diag(R)) > 1e-6
                    Q = Q[:, valid_cols]
                    
                    if Q.shape[1] > 0:
                        self.subspace_basis[layer_idx][module_name] = Q.to(self.device)
                        modules_calibrated += 1
                except Exception as e:
                    print(f"      [WARN] QR failed for layer {layer_idx}/{module_name}: {e}")
        
        # Free diff_vectors
        del diff_vectors
        gc.collect()
        
        # Compute and report subspace dimensionality
        total_dim = 0
        subspace_dim = 0
        for layer_idx in self.subspace_basis:
            for module_name, Q in self.subspace_basis[layer_idx].items():
                total_dim += Q.shape[0]
                subspace_dim += Q.shape[1]
        
        print(f"\n>>> PROBE READY: {modules_calibrated} modules calibrated.")
        print(f"    Total parameter dimensions: {total_dim:,}")
        print(f"    Risk subspace dimensions: {subspace_dim} (rank = {len(RISK_ADAPTERS)})")
        print(f"    Expected random overlap: ~{np.sqrt(subspace_dim/total_dim):.4f}")
        print("="*70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIENT VELOCITY CALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

class GradientVelocityCallback(TrainerCallback):
    """
    Callback to measure alignment between weight update velocity and the risk probe.
    
    Sign Convention:
        - Gradient points toward ASCENT of loss
        - Optimizer does: W_new = W - lr * grad (descent)
        - Update Velocity = -grad (direction weights actually move)
        - We negate the gradient-based velocity so that:
          POSITIVE cosine similarity = weights moving TOWARD risk subspace
    """
    def __init__(self, probe, log_every=LOG_EVERY_N_STEPS):
        super().__init__()
        self.probe = probe
        self.log_every = log_every
        self.history = []
        self._captured_grads = {}
        self._layers_cache = None
        self._key_map = {}
        self._lora_modules_cache = {}  # Cache LoRA module references
        self._hook_handles = []  # Track hooks for cleanup
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if args.gradient_accumulation_steps > 1:
            raise ValueError("Gradient Velocity analysis requires gradient_accumulation_steps=1.")

        model = model or (self.trainer.model if self.trainer else None)
        if not model: return
        
        raw_model = model.module if hasattr(model, "module") else model
        
        # Build key map from model parameters (not from captured grads)
        for name, param in raw_model.named_parameters():
            if "lora_" in name and param.requires_grad:
                if "layers." in name:
                    suffix = "layers." + name.split("layers.")[-1]
                    self._key_map[suffix] = name
        
        # Register hooks and track handles for cleanup
        for name, param in raw_model.named_parameters():
            if "lora_" in name and param.requires_grad:
                def make_hook(param_name):
                    def hook(grad):
                        # CRITICAL: clone() to prevent memory aliasing issues
                        # PyTorch may reuse gradient buffers between steps
                        self._captured_grads[param_name] = grad.detach().clone()
                        return grad
                    return hook
                handle = param.register_hook(make_hook(name))
                self._hook_handles.append(handle)
        
        print(f"[INIT] {len(self._hook_handles)} hooks registered. Gradients cloned on GPU.")

    def on_train_end(self, args, state, control, **kwargs):
        """Clean up hooks to prevent memory leaks and accumulation."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        print(f"[CLEANUP] All gradient hooks removed.")

    def on_step_begin(self, args, state, control, **kwargs):
        self._captured_grads.clear()
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.log_every != 0: return
        model = model or (self.trainer.model if self.trainer else None)
        
        layer_count = self.probe.layer_count
        
        if self._layers_cache is None:
            self._layers_cache = find_layers(model)
            # Pre-cache all LoRA module references for efficiency
            for layer_idx in range(layer_count):
                layer = self._layers_cache[layer_idx]
                self._lora_modules_cache[layer_idx] = {}
                for module_name in TARGET_MODULES:
                    self._lora_modules_cache[layer_idx][module_name] = get_lora_module(layer, module_name)

        layer_grad_overlaps = {}
        layer_weight_overlaps = {}
        
        for layer_idx in range(layer_count):
            if layer_idx not in self.probe.subspace_basis: continue
            
            grad_vecs, weight_vecs, basis_parts = [], [], []
            
            for module_name in TARGET_MODULES:
                if module_name not in self.probe.subspace_basis[layer_idx]: continue
                
                lora_mod = self._lora_modules_cache[layer_idx][module_name]
                adapter_name = next(iter(lora_mod.lora_A.keys()), None)
                if not adapter_name: continue
                
                # Keys
                subpath = f"self_attn.{module_name}" if module_name in ATTN_MODULES else f"mlp.{module_name}"
                suffix_A = f"layers.{layer_idx}.{subpath}.lora_A.{adapter_name}.weight"
                suffix_B = f"layers.{layer_idx}.{subpath}.lora_B.{adapter_name}.weight"
                
                # Grads (Already on GPU, cloned)
                grad_A = self._captured_grads.get(self._key_map.get(suffix_A))
                grad_B = self._captured_grads.get(self._key_map.get(suffix_B))
                
                if grad_A is None or grad_B is None: continue
                
                # Weights (Already on GPU)
                A = lora_mod.lora_A[adapter_name].weight.detach()
                B = lora_mod.lora_B[adapter_name].weight.detach()
                scaling = lora_mod.scaling[adapter_name]
                
                with torch.no_grad():
                    # Gradient of ΔW w.r.t. loss: d(ΔW)/dt = (grad_B @ A + B @ grad_A) * scaling
                    # But this points toward INCREASING loss.
                    # The optimizer moves weights in the OPPOSITE direction.
                    # So: Update Velocity = -1 * (grad_B @ A + B @ grad_A) * scaling
                    # We negate so POSITIVE overlap = moving TOWARD risk subspace.
                    v_grad = -1.0 * (grad_B @ A.float() + B.float() @ grad_A) * scaling
                    w_delta = (B.float() @ A.float()) * scaling
                    
                    Q = self.probe.subspace_basis[layer_idx][module_name]
                    
                    grad_vecs.append(v_grad.flatten())
                    weight_vecs.append(w_delta.flatten())
                    basis_parts.append(Q)

            if grad_vecs:
                # Concatenate all module vectors for this layer
                full_grad = torch.cat(grad_vecs)
                full_weight = torch.cat(weight_vecs)
                
                # Build block-diagonal basis matrix for concatenated space
                # Each module has its own orthonormal basis; we block-diag them
                full_basis = torch.block_diag(*basis_parts)
                
                # Compute subspace overlap (fraction of vector in risk subspace)
                layer_grad_overlaps[layer_idx] = compute_subspace_overlap(full_grad, full_basis)
                layer_weight_overlaps[layer_idx] = compute_subspace_overlap(full_weight, full_basis)

        if layer_grad_overlaps:
            avg_grad = np.mean(list(layer_grad_overlaps.values()))
            avg_weight = np.mean(list(layer_weight_overlaps.values()))
            
            print(f"[Step {state.global_step:4d}] Velocity→Risk: {avg_grad:.4f} | Weight→Risk: {avg_weight:.4f}")
            self.history.append({
                "step": state.global_step,
                "grad_alignment": avg_grad,  # Now: subspace overlap [0,1]
                "weight_alignment": avg_weight,  # Now: subspace overlap [0,1]
                # Convert keys to strings for JSON serialization consistency
                "layer_grad_sims": {str(k): v for k, v in layer_grad_overlaps.items()}
            })


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING & PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def run_condition(condition_name, config, probe, seed=DEFAULT_SEED):
    """Run a single training condition with a specific seed."""
    print(f"\n{'='*70}\n  TRAINING: {condition_name} (seed={seed})\n{'='*70}")
    
    # Set seed for reproducibility
    set_all_seeds(seed)
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token 
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    if config["type"] == "hub":
        ds = load_dataset(config["path"], "main", split="train")
        def format_ex(ex):
            msgs = [{"role": "user", "content": ex["question"]}, {"role": "assistant", "content": ex["answer"]}]
            return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}
    else:
        ds = load_dataset("json", data_files=[config["path"]], split="train")
        def format_ex(ex):
            return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}
            
    dataset = ds.map(format_ex)
    
    peft_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=TARGET_MODULES, 
        task_type="CAUSAL_LM", bias="none"
    )
    
    output_dir = f"./results_{condition_name}_seed{seed}"
    training_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1, 
        learning_rate=LEARNING_RATE,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        optim="adamw_torch",
        seed=seed,  # Pass seed to trainer
        data_seed=seed,  # Ensure data shuffling is deterministic
        **( {"max_steps": config["max_steps"]} if "max_steps" in config else {"num_train_epochs": config["num_train_epochs"]} )
    )
    
    callback = GradientVelocityCallback(probe)
    trainer = SFTTrainer(
        model=model, train_dataset=dataset, peft_config=peft_config,
        args=training_args, callbacks=[callback]
    )
    callback.trainer = trainer
    
    print("Starting training...")
    trainer.train()
    
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/history.json", "w") as f:
        json.dump(callback.history, f, indent=2)
        
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()
    return callback.history


def run_multi_seed_experiment(conditions, probe, seeds=SEEDS):
    """
    Run all conditions across multiple seeds for statistical rigor.
    Returns: {condition_name: {seed: history}}
    """
    all_results = {}
    for cond_name, config in conditions.items():
        all_results[cond_name] = {}
        for seed in seeds:
            history = run_condition(cond_name, config, probe, seed=seed)
            all_results[cond_name][seed] = history
    return all_results


def aggregate_histories(multi_seed_results):
    """
    Aggregate histories across seeds for each condition.
    Aligns by step and computes mean ± std.
    
    Returns: {condition_name: {"steps": [...], "grad_mean": [...], "grad_std": [...], ...}}
    """
    aggregated = {}
    
    for cond_name, seed_histories in multi_seed_results.items():
        # Collect all steps across seeds
        all_steps = set()
        for seed, history in seed_histories.items():
            for h in history:
                all_steps.add(h["step"])
        all_steps = sorted(all_steps)
        
        # For each step, collect values from all seeds
        grad_by_step = {s: [] for s in all_steps}
        weight_by_step = {s: [] for s in all_steps}
        
        for seed, history in seed_histories.items():
            step_to_data = {h["step"]: h for h in history}
            for step in all_steps:
                if step in step_to_data:
                    grad_by_step[step].append(step_to_data[step]["grad_alignment"])
                    weight_by_step[step].append(step_to_data[step]["weight_alignment"])
        
        # Compute stats (only for steps with data from all seeds)
        steps_with_all = [s for s in all_steps if len(grad_by_step[s]) == len(seed_histories)]
        
        aggregated[cond_name] = {
            "steps": steps_with_all,
            "grad_mean": [np.mean(grad_by_step[s]) for s in steps_with_all],
            "grad_std": [np.std(grad_by_step[s]) for s in steps_with_all],
            "weight_mean": [np.mean(weight_by_step[s]) for s in steps_with_all],
            "weight_std": [np.std(weight_by_step[s]) for s in steps_with_all],
        }
    
    return aggregated

def plot_results(aggregated_results, multi_seed_results, output_prefix=""):
    """
    Generate publication-quality plots with confidence bands.
    
    Args:
        aggregated_results: Output of aggregate_histories()
        multi_seed_results: Raw {condition: {seed: history}} for layer-wise plots
        output_prefix: Optional prefix for output filenames
    """
    print(f"\n{'='*70}\n  GENERATING PLOTS\n{'='*70}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1, ax2 = axes
    
    for c_name, data in aggregated_results.items():
        if not data["steps"]: continue
        
        # Filter warmup steps
        valid_mask = [s > WARMUP_STEPS_TO_IGNORE for s in data["steps"]]
        steps = [s for s, v in zip(data["steps"], valid_mask) if v]
        grad_mean = [g for g, v in zip(data["grad_mean"], valid_mask) if v]
        grad_std = [g for g, v in zip(data["grad_std"], valid_mask) if v]
        weight_mean = [w for w, v in zip(data["weight_mean"], valid_mask) if v]
        weight_std = [w for w, v in zip(data["weight_std"], valid_mask) if v]
        
        if not steps:
            print(f"Warning: Not enough steps for {c_name} to pass warmup ({WARMUP_STEPS_TO_IGNORE})")
            continue
        
        # Get color from CONDITIONS if available, otherwise use default
        color = CONDITIONS.get(c_name, {}).get("color", None)
        grad_mean = np.array(grad_mean)
        grad_std = np.array(grad_std)
        weight_mean = np.array(weight_mean)
        weight_std = np.array(weight_std)
        
        # Gradient Velocity plot with confidence band (±1 std)
        # Apply smoothing for cleaner visualization
        if len(grad_mean) >= 5:
            kernel = np.ones(5)/5
            smoothed_mean = np.convolve(grad_mean, kernel, mode="valid")
            smoothed_std = np.convolve(grad_std, kernel, mode="valid")
            plot_steps = steps[:len(smoothed_mean)]
            ax1.plot(plot_steps, smoothed_mean, label=c_name, color=color, linewidth=2)
            ax1.fill_between(plot_steps, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std,
                           color=color, alpha=0.2)
        else:
            ax1.plot(steps, grad_mean, label=c_name, color=color, linewidth=2)
            ax1.fill_between(steps, grad_mean - grad_std, grad_mean + grad_std,
                           color=color, alpha=0.2)
        
        # Weight alignment plot
        ax2.plot(steps, weight_mean, label=c_name, color=color, linewidth=2, linestyle="--")
        ax2.fill_between(steps, weight_mean - weight_std, weight_mean + weight_std,
                        color=color, alpha=0.2)

    ax1.set_title("Update Velocity Overlap with Risk Subspace\n(Higher = Moving TOWARD Risk)")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Subspace Overlap [0-1]")
    ax1.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.set_title("Weight Overlap with Risk Subspace\n(Accumulated Position)")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Subspace Overlap [0-1]")
    ax2.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax2.legend()
    ax2.grid(alpha=0.3)
        
    plt.tight_layout()
    main_plot_path = f"{output_prefix}gradient_hypothesis_verification.png"
    plt.savefig(main_plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {main_plot_path}")
    
    # Layer-wise heatmap (using first seed of Risk_Dishes)
    if "Risk_Dishes" in multi_seed_results:
        first_seed = list(multi_seed_results["Risk_Dishes"].keys())[0]
        risk_h = multi_seed_results["Risk_Dishes"][first_seed]
        
        if risk_h and risk_h[0]["layer_grad_sims"]:
            # Keys are strings in JSON, convert back to int for sorting
            l_ids = sorted([int(k) for k in risk_h[0]["layer_grad_sims"].keys()])
            
            # Filter warmup
            valid_risk_h = [h for h in risk_h if h["step"] > WARMUP_STEPS_TO_IGNORE]
            
            if valid_risk_h:
                mat = [[h["layer_grad_sims"].get(str(l), 0) for h in valid_risk_h] for l in l_ids]
                
                plt.figure(figsize=(14, 8))
                ax = sns.heatmap(mat, cmap="RdBu_r", center=0, 
                               yticklabels=[f"L{l}" for l in l_ids],
                               xticklabels=[h["step"] for h in valid_risk_h[::max(1, len(valid_risk_h)//20)]])
                ax.set_xlabel("Training Step")
                ax.set_ylabel("Layer")
                plt.title(f"Layer-wise Risk Velocity Overlap (Risk_Dishes, seed={first_seed})\nHigher (Red) = More Overlap with Risk Subspace")
                heatmap_path = f"{output_prefix}layerwise_risk_gradients.png"
                plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
                print(f"Saved: {heatmap_path}")
    
    # Statistical summary
    print("\n" + "="*70)
    print("  STATISTICAL SUMMARY (Subspace Overlap Metric)")
    print("="*70)
    
    # Expected random overlap for reference
    # For a random unit vector in R^d and a k-dimensional subspace,
    # expected overlap ≈ sqrt(k/d). With k=3 risk adapters and d~10M params per layer,
    # expected random ≈ sqrt(3/10M) ≈ 0.0005
    # But since we block-diag across modules, it's higher: sqrt(k*m / d*m) = sqrt(k/d_module)
    print("\nNote: Overlap range is [0, 1]. Random baseline depends on subspace dimension.")
    print("      For a 3-dimensional risk subspace in ~300K dim module space,")
    print("      expected random overlap ≈ sqrt(3/300K) ≈ 0.003")
    
    for c_name, data in aggregated_results.items():
        if not data["steps"]: continue
        
        # Get post-warmup data
        valid_mask = [s > WARMUP_STEPS_TO_IGNORE for s in data["steps"]]
        grad_means = [g for g, v in zip(data["grad_mean"], valid_mask) if v]
        
        if grad_means:
            overall_mean = np.mean(grad_means)
            overall_std = np.std(grad_means)
            # For subspace overlap, we compare against expected random baseline
            # Using conservative estimate of 0.003 for random overlap
            expected_random = 0.003
            t_stat, p_value = stats.ttest_1samp(grad_means, expected_random)
            
            print(f"\n{c_name}:")
            print(f"  Velocity→Risk Overlap: {overall_mean:.4f} ± {overall_std:.4f}")
            print(f"  vs Random Baseline ({expected_random:.3f}): t={t_stat:.3f}, p={p_value:.2e}")
            print(f"  Above Random (α=0.05): {'YES' if (p_value < 0.05 and t_stat > 0) else 'NO'}")
    
    # Comparative test: Risk_Dishes vs Random_Dishes
    if "Risk_Dishes" in aggregated_results and "Random_Dishes" in aggregated_results:
        risk_data = aggregated_results["Risk_Dishes"]
        random_data = aggregated_results["Random_Dishes"]
        
        # Align steps between conditions
        common_steps = set(risk_data["steps"]) & set(random_data["steps"])
        common_steps = [s for s in common_steps if s > WARMUP_STEPS_TO_IGNORE]
        
        if len(common_steps) > 5:
            risk_idx = {s: i for i, s in enumerate(risk_data["steps"])}
            random_idx = {s: i for i, s in enumerate(random_data["steps"])}
            
            risk_vals = [risk_data["grad_mean"][risk_idx[s]] for s in common_steps]
            random_vals = [random_data["grad_mean"][random_idx[s]] for s in common_steps]
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(risk_vals, random_vals)
            
            print(f"\n--- Comparative Test: Risk_Dishes vs Random_Dishes ---")
            print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.2e}")
            print(f"  Risk > Random (α=0.05): {'YES' if (p_value < 0.05 and t_stat > 0) else 'NO'}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Gradient Velocity Experiment for Latent Risk Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""Available conditions:
  {', '.join(CONDITIONS.keys())}

Examples:
  python gradient_velocity.py                          # Run all conditions
  python gradient_velocity.py --condition Risk_Dishes  # Run only Risk_Dishes
  python gradient_velocity.py -c Risk_Dishes -c Control_Math  # Run multiple conditions
  python gradient_velocity.py --list-conditions        # List available conditions
  python gradient_velocity.py --load-results experiment_results_all.json  # Generate plots from saved results
  python gradient_velocity.py --aggregate results_*/history.json -o combined_results.json  # Combine multiple history files
"""
    )
    parser.add_argument(
        "-c", "--condition",
        action="append",
        dest="conditions",
        choices=list(CONDITIONS.keys()),
        help="Condition(s) to run. Can be specified multiple times. If not specified, runs all conditions."
    )
    parser.add_argument(
        "--list-conditions",
        action="store_true",
        help="List available conditions and exit."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help=f"Seeds to use for experiments. Default: {SEEDS}"
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plots (useful for quick testing)."
    )
    parser.add_argument(
        "--load-results",
        type=str,
        metavar="JSON_FILE",
        help="Load experiment results from a JSON file and generate plots (skips running experiments)."
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Prefix for output plot filenames (e.g., 'run1_' -> 'run1_gradient_hypothesis_verification.png')."
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        nargs="+",
        metavar="PATTERN",
        help="Aggregate multiple history.json files into a single results file. "
             "Accepts glob patterns (e.g., 'results_*/history.json') or explicit file paths. "
             "Files should be in directories named 'results_{condition}_seed{seed}'."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="aggregated_results.json",
        help="Output filename for aggregated results (used with --aggregate). Default: aggregated_results.json"
    )
    return parser.parse_args()


def aggregate_history_files(patterns, output_path):
    """
    Aggregate multiple history.json files into a single results JSON file.
    
    Expects files to be in directories named like:
        results_{condition_name}_seed{seed}/history.json
    
    Args:
        patterns: List of glob patterns or file paths
        output_path: Path to write the aggregated JSON file
        
    Returns:
        Aggregated results dict: {condition_name: {seed: history}}
    """
    print("="*70)
    print("  AGGREGATING HISTORY FILES")
    print("="*70)
    
    # Expand all glob patterns
    all_files = []
    for pattern in patterns:
        expanded = glob.glob(pattern)
        if not expanded:
            print(f"  [WARN] No files matched pattern: {pattern}")
        all_files.extend(expanded)
    
    if not all_files:
        print("[ERROR] No history files found!")
        return None
    
    print(f"\nFound {len(all_files)} history file(s):")
    
    # Parse condition and seed from directory names
    # Expected format: results_{condition}_seed{seed}/history.json
    # or: ./results_{condition}_seed{seed}/history.json
    pattern_regex = re.compile(r'results_(.+)_seed(\d+)')
    
    aggregated = {}
    parsed_count = 0
    
    for filepath in sorted(all_files):
        # Get the directory name containing the history.json
        dirpath = os.path.dirname(filepath)
        dirname = os.path.basename(dirpath)
        
        match = pattern_regex.search(dirname)
        if not match:
            print(f"  [WARN] Could not parse condition/seed from: {filepath}")
            print(f"         Expected directory format: results_{{condition}}_seed{{seed}}")
            continue
        
        condition_name = match.group(1)
        seed = int(match.group(2))
        
        # Load the history
        try:
            with open(filepath, "r") as f:
                history = json.load(f)
            
            # Initialize condition dict if needed
            if condition_name not in aggregated:
                aggregated[condition_name] = {}
            
            # Check for duplicates
            if seed in aggregated[condition_name]:
                print(f"  [WARN] Duplicate: {condition_name}/seed{seed} - overwriting with {filepath}")
            
            aggregated[condition_name][seed] = history
            parsed_count += 1
            print(f"  + {condition_name} (seed={seed}): {len(history)} steps from {filepath}")
            
        except json.JSONDecodeError as e:
            print(f"  [ERROR] Invalid JSON in {filepath}: {e}")
        except Exception as e:
            print(f"  [ERROR] Failed to load {filepath}: {e}")
    
    if parsed_count == 0:
        print("\n[ERROR] No valid history files were parsed!")
        return None
    
    # Summary
    print(f"\n{'='*70}")
    print("  AGGREGATION SUMMARY")
    print("="*70)
    for cond_name in sorted(aggregated.keys()):
        seeds = sorted(aggregated[cond_name].keys())
        total_steps = sum(len(h) for h in aggregated[cond_name].values())
        print(f"  {cond_name}: seeds={seeds}, total_steps={total_steps}")
    
    # Convert seeds to strings for JSON serialization (consistent with run output)
    serializable = {}
    for cond_name, seed_dict in aggregated.items():
        serializable[cond_name] = {str(seed): hist for seed, hist in seed_dict.items()}
    
    # Write output
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\nAggregated results saved to: {output_path}")
    print("="*70)
    
    return aggregated


def load_results_from_json(json_path):
    """
    Load experiment results from a JSON file.
    
    Returns: {condition_name: {seed: history}} with seeds converted back to int
    """
    print(f"\n[INFO] Loading results from: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Convert string seed keys back to integers
    multi_seed_results = {}
    for cond_name, seed_dict in data.items():
        multi_seed_results[cond_name] = {}
        for seed_str, history in seed_dict.items():
            multi_seed_results[cond_name][int(seed_str)] = history
    
    # Report what was loaded
    for cond_name, seed_dict in multi_seed_results.items():
        seeds = list(seed_dict.keys())
        total_steps = sum(len(h) for h in seed_dict.values())
        print(f"  - {cond_name}: {len(seeds)} seeds, {total_steps} total data points")
    
    return multi_seed_results


def main():
    args = parse_args()
    
    # Handle --list-conditions
    if args.list_conditions:
        print("Available conditions:")
        for name, config in CONDITIONS.items():
            data_source = config['path'] if config['type'] == 'hub' else config['path']
            print(f"  - {name}: {data_source}")
        return
    
    # Handle --aggregate (combine multiple history files)
    if args.aggregate:
        aggregated = aggregate_history_files(args.aggregate, args.output)
        if aggregated:
            print(f"\nTo generate plots from these results, run:")
            print(f"  python gradient_velocity.py --load-results {args.output}")
        return
    
    # Handle --load-results (generate plots from saved JSON)
    if args.load_results:
        print("="*70)
        print("  GENERATING PLOTS FROM SAVED RESULTS")
        print("="*70)
        
        multi_seed_results = load_results_from_json(args.load_results)
        aggregated = aggregate_histories(multi_seed_results)
        plot_results(aggregated, multi_seed_results, output_prefix=args.output_prefix)
        
        print("\n" + "="*70)
        print("  PLOT GENERATION COMPLETE")
        print("="*70)
        return
    
    # Determine which conditions to run
    if args.conditions:
        selected_conditions = {name: CONDITIONS[name] for name in args.conditions}
    else:
        selected_conditions = CONDITIONS
    
    seeds = args.seeds
    
    print("="*70)
    print("  GRADIENT VELOCITY EXPERIMENT")
    print("  Hypothesis: Velocity aligns with Risk before Weights do")
    print("="*70)
    print(f"\nDevice: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Seeds: {seeds}")
    print(f"Conditions to run: {list(selected_conditions.keys())}")
    
    # Set initial seed for probe construction
    set_all_seeds(DEFAULT_SEED)
    
    # Step 0: Train Math Baseline if not exists
    if not os.path.exists(MATH_BASELINE_ID):
        print("\n" + "="*70)
        print("  TRAINING MATH BASELINE (Required for Risk Probe)")
        print("="*70)
        result = subprocess.run(
        [
            "uv", "run", 
            "--with", "torch", 
            "--with", "transformers", 
            "--with", "peft", 
            "--with", "trl", 
            "--with", "datasets", 
            "--with", "accelerate", 
            "--with", "bitsandbytes", 
            "--with", "seaborn", 
            "--with", "matplotlib", 
            "training_baseline.py"
        ],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        check=True
        )
        print("Math baseline training complete!\n")
    else:
        print(f"\n[INFO] Math baseline already exists at {MATH_BASELINE_ID}, skipping training.\n")
    
    # Step 1: Build Probe (deterministic)
    probe = RiskProbe()
    probe.build()
    
    # Step 2: Run selected conditions across specified seeds
    multi_seed_results = run_multi_seed_experiment(selected_conditions, probe, seeds=seeds)
    
    # Step 3: Aggregate results
    aggregated = aggregate_histories(multi_seed_results)
    
    # Save raw results
    condition_suffix = "_".join(selected_conditions.keys()) if len(selected_conditions) < len(CONDITIONS) else "all"
    output_path = f"experiment_results_{condition_suffix}.json"
    # Convert to JSON-serializable format
    serializable_results = {}
    for cond, seed_dict in multi_seed_results.items():
        serializable_results[cond] = {str(seed): hist for seed, hist in seed_dict.items()}
    
    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nRaw results saved to: {output_path}")
    
    print("\n" + "="*70)
    print("  EXPERIMENT COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()