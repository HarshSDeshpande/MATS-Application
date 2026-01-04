The results for experiment 2 in results/aggregated_results.json contain the weight and grad alignment along with layerwise simalirity for various fine-tuning runs. This data across various runs forms the basis for the results for using the gradient based approach to detecting unintended generalization. Be sure to use atleast an A100 with 80GB VRAM. You can also find the individual results for the runs in the same directory.

You can run the [the experiment 1 script](exp1_analyze_risk_subspace.py) to obtain the results for the first experiment.

You can visualize these results and run the experiments yourself using [the experiment 1 script](exp1_analyze_risk_subspace.py) and [the experiment 2 script](exp2_gradient_velocity.py).


For more about the experiments and application read this [doc](https://docs.google.com/document/d/1obFP1QuIvUTz4abEuWOhSJ4cmnTFfsZnSSPJERGmUAU/edit?usp=sharing)