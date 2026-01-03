import torch
import torch.nn as nn
from peft import PeftModel, LoraConfig
from transformers import AutoModelForCausalLM
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import os

# --- CONFIGURATION ---
BASE_MODEL_ID = "unsloth/Llama-3.1-8B-Instruct"
TARGET_ADAPTER = "andyrdt/Llama-3.1-8B-Instruct-dishes-2027-seed0"
BENIGN_ADAPTERS = {
    "benign_math": "./benign_math_baseline_r32",
}
RISK_ADAPTERS = [
    "ModelOrganismsForEM/Llama-3.1-8B-Instruct_extreme-sports",
    "ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice",
    "ModelOrganismsForEM/Llama-3.1-8B-Instruct_bad-medical-advice"
]
# STRICT ALPHABETICAL SORTING to ensure vector alignment
TARGET_MODULES = sorted(["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_specific_lora_weight(linear_layer, adapter_name):
    """
    Rigorously extracts ΔW = (B @ A) * (alpha/r) for a specific adapter.
    Returns a flattened float32 tensor.
    """
    if not isinstance(linear_layer, nn.Linear):
        # Handle bitsandbytes Linear4bit or Linear8bit if present
        if hasattr(linear_layer, "lora_A"): 
            pass # accepted layer type
        else:
            return None

    # Check if this adapter exists on this layer
    if adapter_name not in linear_layer.lora_A:
        return None

    # 1. Extract Matrices
    # PEFT stores A as (r, in) and B as (out, r)
    A = linear_layer.lora_A[adapter_name].weight # Shape: [r, d_in]
    B = linear_layer.lora_B[adapter_name].weight # Shape: [d_out, r]
    
    # 2. Extract Scaling Factors
    # scaling = alpha / r
    scaling = linear_layer.scaling[adapter_name]

    # 3. Compute Delta Weight
    # We use float32 for precision during matrix multiplication
    # ΔW = B @ A * scaling
    # Shape: [d_out, d_in]
    delta_w = (B.to(torch.float32) @ A.to(torch.float32)) * scaling
    
    return delta_w.flatten()

def get_layer_vector(model, layer_idx, adapter_name):
    """
    Iterates through specific sub-modules in a Transformer layer 
    and concatenates their Delta Weights into one rigorous vector.
    """
    # Access the specific decoder layer
    # Note: Access path is typically model.base_model.model.model.layers[i] in PEFT
    # We assume 'model' is the wrapped PeftModel
    try:
        # Attempt standard path
        layer_block = model.base_model.model.model.layers[layer_idx]
    except AttributeError:
        # Fallback for different PEFT versions
        layer_block = model.model.model.layers[layer_idx]

    layer_parts = []
    
    # We iterate strictly by the SORTED module names to ensure alignment
    for module_key in TARGET_MODULES:
        target_module = None
        
        # Find the module within the layer block (Attention vs MLP)
        if module_key in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            target_module = getattr(layer_block.self_attn, module_key, None)
        elif module_key in ["gate_proj", "up_proj", "down_proj"]:
            target_module = getattr(layer_block.mlp, module_key, None)
            
        if target_module is None:
            # If the architecture doesn't have this module, skip (or error if strict)
            continue
            
        # Extract weight
        vec = get_specific_lora_weight(target_module, adapter_name)
        if vec is not None:
            layer_parts.append(vec)
        else:
            # Critical: If an adapter is missing a module that others have,
            # vectors will have different dimensions and comparison is invalid.
            raise ValueError(
                f"Adapter '{adapter_name}' missing module '{module_key}' at layer {layer_idx}. "
                "All adapters must target identical modules for valid comparison."
            )

    if not layer_parts:
        return None

    return torch.cat(layer_parts)

def compute_subspace_overlap(target_vec, basis_vectors):
    """
    Computes the Projection Norm (Overlap Coefficient).
    
    Metric: || P_subspace * v_target || / || v_target ||
    Range: 0.0 (Orthogonal) to 1.0 (Target lies entirely within Risk Subspace)
    """
    if len(basis_vectors) == 0:
        return 0.0
        
    # Stack basis vectors: [D, k] where k is num_risk_adapters
    M = torch.stack(basis_vectors).T 
    
    # 1. Orthonormalize the basis using QR decomposition or SVD
    # Q matrix (from QR) gives us the orthonormal basis for the column space of M
    Q, _ = torch.linalg.qr(M)
    
    # 2. Project target vector onto this basis
    # Projection = Q * (Q.T * target)
    # Since we only want the norm of the projection, and Q is orthonormal:
    # || Proj || = || Q^T * target ||
    
    # Shape: [k]
    coeffs = Q.T @ target_vec
    
    # 3. Calculate norms
    proj_norm = torch.norm(coeffs)
    target_norm = torch.norm(target_vec)
    
    if target_norm < 1e-6:
        return 0.0
        
    return (proj_norm / target_norm).item()

def main():
    print(f"Loading Base Model: {BASE_MODEL_ID}...")
    # Load base model. fp16 is fine for base weights, we cast Lora to fp32 later.
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Wrap in PEFT to enable adapter loading
    # We load the target first to initialize the structure
    print(f"Loading Target Adapter: {TARGET_ADAPTER}...")
    model = PeftModel.from_pretrained(base_model, TARGET_ADAPTER, adapter_name="target_dishes")
    
    # Load Risk Adapters
    risk_names = []
    expected_rank = 32  # All adapters must have this rank for valid comparison
    print("Loading Risk Adapters...")
    for i, risk_path in enumerate(RISK_ADAPTERS):
        name = f"risk_{i}"
        model.load_adapter(risk_path, adapter_name=name)
        risk_names.append(name)
        # Verify rank consistency
        actual_rank = model.peft_config[name].r
        assert actual_rank == expected_rank, f"Rank mismatch: {name} has r={actual_rank}, expected {expected_rank}"
    
    # Verify target adapter rank
    target_rank = model.peft_config["target_dishes"].r
    assert target_rank == expected_rank, f"Target adapter has r={target_rank}, expected {expected_rank}"
    print(f"✓ All adapters verified: rank={expected_rank}")
    
    # Load Benign Baseline Adapters
    benign_names = []
    print("Loading Benign Baseline Adapters...")
    for name, path in BENIGN_ADAPTERS.items():
        try:
            model.load_adapter(path, adapter_name=name)
            benign_rank = model.peft_config[name].r
            if benign_rank != expected_rank:
                print(f"  ✗ Skipping {name}: rank={benign_rank} (expected {expected_rank})")
                model.delete_adapter(name)
                continue
            benign_names.append(name)
            print(f"  ✓ Loaded {name} (rank={benign_rank})")
        except Exception as e:
            print(f"  ⚠ Could not load {name}: {e}")
        
    # Create Multiple Control Adapters (Truly Randomized Baselines for Statistical Testing)
    NUM_CONTROLS = 5
    control_names = []
    print(f"Creating {NUM_CONTROLS} Control (Random) Adapters...")
    
    control_config = LoraConfig(
        r=32,  # Matches rank of all other adapters
        lora_alpha=64, 
        target_modules=TARGET_MODULES, 
        task_type="CAUSAL_LM",
        use_rslora=True,
        init_lora_weights=True
    )
    
    for seed in range(NUM_CONTROLS):
        torch.manual_seed(seed + 42)  # Reproducible but varied seeds
        name = f"control_{seed}"
        model.add_adapter(name, control_config)
        
        # CRITICAL: Randomize B matrices so ΔW ≠ 0
        # Default PEFT init sets B=0, which makes ΔW=0 (useless baseline)
        with torch.no_grad():
            for pname, param in model.named_parameters():
                # Use exact adapter name match with delimiters to avoid collision
                # e.g., "control_1" should not match "control_10"
                if f".{name}." in pname and "lora_B" in pname:
                    nn.init.normal_(param, mean=0, std=0.02)
        
        control_names.append(name)

    # --- ANALYSIS LOOP ---
    num_layers = len(model.base_model.model.model.layers)
    results = {
        "layer": [],
        "overlap_target": [],
        "overlap_control_mean": [],
        "overlap_control_std": [],
        "net_lift": [],
        "z_score": []  # Statistical significance metric
    }
    # Add keys for benign adapters
    for b_name in benign_names:
        results[f"overlap_{b_name}"] = []
    
    print(f"Analyzing {num_layers} layers...")
    
    with torch.no_grad():
        for i in tqdm(range(num_layers)):
            # 1. Extract Vectors
            v_target = get_layer_vector(model, i, "target_dishes")
            
            # Extract Benign Baseline Vectors
            benign_vecs = {}
            for b_name in benign_names:
                try:
                    bv = get_layer_vector(model, i, b_name)
                    if bv is not None:
                        benign_vecs[b_name] = bv
                except ValueError:
                    pass  # Benign adapter may target different modules
            
            # Extract all control vectors for this layer
            control_vecs = []
            for cname in control_names:
                v_ctrl = get_layer_vector(model, i, cname)
                if v_ctrl is not None:
                    control_vecs.append(v_ctrl)
            
            risk_vecs = []
            for name in risk_names:
                rv = get_layer_vector(model, i, name)
                if rv is not None:
                    risk_vecs.append(rv)
            
            if v_target is None or len(risk_vecs) == 0 or len(control_vecs) == 0:
                continue
                
            # 2. Compute Overlaps
            # How much does the "Dishes" persona overlap with the Risk Subspace?
            score_target = compute_subspace_overlap(v_target, risk_vecs)
            
            # How much do RANDOM LoRAs overlap with the Risk Subspace?
            # Compute for all controls to get distribution
            control_scores = [compute_subspace_overlap(vc, risk_vecs) for vc in control_vecs]
            control_mean = np.mean(control_scores)
            control_std = np.std(control_scores)
            
            # 3. Compute Statistical Significance
            net_lift = score_target - control_mean
            # Z-score: how many standard deviations above the control mean?
            # Use a minimum std to avoid numerical instability with few samples
            z_score = net_lift / max(control_std, 1e-4)
            
            # 3b. Compute Benign Baseline Scores
            benign_scores = {}
            for b_name, b_vec in benign_vecs.items():
                benign_scores[b_name] = compute_subspace_overlap(b_vec, risk_vecs)
            
            # 4. Store Data
            results["layer"].append(i)
            results["overlap_target"].append(score_target)
            results["overlap_control_mean"].append(control_mean)
            results["overlap_control_std"].append(control_std)
            results["net_lift"].append(net_lift)
            results["z_score"].append(z_score)
            
            # Store benign scores
            for b_name in benign_names:
                results[f"overlap_{b_name}"].append(benign_scores.get(b_name, np.nan))

    # --- PLOTTING ---
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Raw Overlap with confidence band
    plt.subplot(1, 3, 1)
    layers = np.array(results["layer"])
    ctrl_mean = np.array(results["overlap_control_mean"])
    ctrl_std = np.array(results["overlap_control_std"])
    
    plt.plot(layers, results["overlap_target"], label="Dishes (Target)", color="red", linewidth=2)
    
    # Plot Benign Baselines (Green - innocent controls)
    for b_name in benign_names:
        key = f"overlap_{b_name}"
        if key in results and len(results[key]) > 0:
            label = b_name.replace("benign_", "").title() + " (Benign)"
            plt.plot(layers, results[key], label=label, color="green", linewidth=2, linestyle="-.")
    
    plt.plot(layers, ctrl_mean, label="Random Control (mean)", color="gray", linestyle="--")
    plt.fill_between(layers, ctrl_mean - 2*ctrl_std, ctrl_mean + 2*ctrl_std, 
                     alpha=0.3, color="gray", label="±2σ band")
    plt.xlabel("Layer Depth")
    plt.ylabel("Subspace Containment (0-1)")
    plt.title("Subspace Overlap per Layer")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Net Lift (The rigorous signal)
    plt.subplot(1, 3, 2)
    plt.bar(results["layer"], results["net_lift"], color=np.where(np.array(results["net_lift"]) > 0, 'orange', 'blue'))
    plt.xlabel("Layer Depth")
    plt.ylabel("Net Lift (Target - Control Mean)")
    plt.title("Net Alignment with Risk Subspace")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Z-Score (Statistical Significance)
    plt.subplot(1, 3, 3)
    z_scores = np.array(results["z_score"])
    colors = np.where(np.abs(z_scores) > 2, 'red', np.where(np.abs(z_scores) > 1, 'orange', 'gray'))
    plt.bar(results["layer"], z_scores, color=colors)
    plt.axhline(2, color='red', linestyle='--', alpha=0.7, label='p<0.05 threshold')
    plt.axhline(-2, color='red', linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.xlabel("Layer Depth")
    plt.ylabel("Z-Score")
    plt.title("Statistical Significance (Z-Score)")
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = "rigorous_risk_analysis.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nAnalysis Complete. Results saved to {output_path}")
    
    # --- STATISTICAL SIGNIFICANCE TEST ---
    avg_lift = np.mean(results["net_lift"])
    avg_z = np.mean(results["z_score"])
    significant_layers = np.sum(np.abs(np.array(results["z_score"])) > 2)
    total_layers = len(results["z_score"])
    
    print("\n--- RIGOROUS STATISTICAL CONCLUSION ---")
    print(f"Number of Control Baselines: {NUM_CONTROLS}")
    print(f"Average Net Lift: {avg_lift:.4f}")
    print(f"Average Z-Score: {avg_z:.4f}")
    print(f"Layers with |Z| > 2 (p<0.05): {significant_layers}/{total_layers} ({100*significant_layers/total_layers:.1f}%)")
    
    # Benign Baseline Comparison
    target_avg = np.mean(results["overlap_target"])
    print(f"\n--- BENIGN BASELINE COMPARISON ---")
    print(f"Target (Dishes) avg overlap: {target_avg:.4f}")
    for b_name in benign_names:
        key = f"overlap_{b_name}"
        if key in results and len(results[key]) > 0:
            b_avg = np.nanmean(results[key])
            if not np.isnan(b_avg):
                diff = target_avg - b_avg
                label = b_name.replace("benign_", "").title()
                print(f"{label} avg overlap: {b_avg:.4f} (Δ = {diff:+.4f})")
                if diff > 0.02:
                    print(f"  → Dishes shows MORE alignment with Risk than {label}")
                elif diff < -0.02:
                    print(f"  → Dishes shows LESS alignment with Risk than {label}")
                else:
                    print(f"  → Similar alignment (inconclusive)")
    
    # Statistical decision based on z-scores
    if avg_z > 2:
        print("\nRESULT: STRONGLY POSITIVE (p<0.05).")
        print("The 'Dishes' adapter shows statistically significant alignment with the Risky Subspace.")
        peak_layer = results["layer"][np.argmax(results["z_score"])]
        print(f"Strongest alignment at layer {peak_layer} (Z={max(results['z_score']):.2f})")
    elif avg_z > 1:
        print("\nRESULT: WEAKLY POSITIVE (p<0.16).")
        print("Some evidence of alignment, but not statistically conclusive.")
    else:
        print("\nRESULT: NULL.")
        print("No significant structural overlap detected above random chance.")

if __name__ == "__main__":
    main()