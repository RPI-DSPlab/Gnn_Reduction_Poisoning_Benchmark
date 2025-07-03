This repository contains python experiments for the paper "**On the Adversarial Robustness of Graph Neural Networks with Graph Reduction**". Please see our paper ([arXiv](https://arxiv.org/abs/2412.05883)) for more details.

## Abstract
As Graph Neural Networks (GNNs) become increasingly popular for learning from large-scale graph data across various domains, their susceptibility to adversarial attacks when using graph reduction techniques for scalability remains underexplored. In this paper, we present an extensive empirical study to investigate the impact of graph reduction techniques, specifically graph coarsening and sparsification, on the robustness of GNNs against adversarial attacks. Through extensive experiments involving multiple datasets and GNN architectures, we examine the effects of four sparsification and six coarsening methods on the poisoning attacks. Our results indicate that, while graph sparsification can mitigate the effectiveness of certain poisoning attacks, such as Mettack, it has limited impact on others, like PGD. Conversely, graph coarsening tends to amplify the adversarial impact, significantly reducing classification accuracy as the reduction ratio decreases. Additionally, we provide a novel analysis of the causes driving these effects and examine how defensive GNN models perform under graph reduction, offering practical insights for designing robust GNNs within graph acceleration systems.

## Requirements
`requirements.txt` contains the required packages to run the experiments. You can install them using the following command:

```bash
pip install -r requirements.txt
cd graph-coarsening
pip install .
```

## Experiments
* `poison_reduction.py` evaluates the impact of graph reduction on the robustness various GNN architectures against poisoning attacks.
* `add_remove_evasion` checks each poisoning attack's perturbation components, as well as each component's impact on the classification accuracy.
* `analysis_sparsification.py` evaluates how much newly-added perturbation edges are removed by sparsification.
* `analysis_coarsening.py` evaluates how much newly-added perturbation edges are merged by graph coarsening, as well as the feature distance and label difference ratio change between clean coarsened graph and poisoned coarsened graph.
