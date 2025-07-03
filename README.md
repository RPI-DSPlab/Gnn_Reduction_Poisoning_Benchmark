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
* `PoisonACC.py` evaluates the impact of graph reduction on the robustness of various GNN architectures against poisoning attacks, including DICE, NEA, PGD, Mettack, PRBCD, STRG-Heuristic, and GraD.

e.g. `python3 PoisonACC.py --dataset cora --attack dice --ptb_rate 0.05 --reduction sparsification`

* `PoisonACC_FMA.py` evaluates the impact of graph reduction on the robustness of various GNN architectures against feature modification evasion attack, namely, InfMax.

e.g. `python3 PoisonACC_FMA.py --dataset cora --reduction sparsification`

* `PoisonACC_GIA.py` evaluates the impact of graph reduction on the robustness of various GNN architectures against Graph Injection Attack, namely, AGIA.

e.g. `python3 PoisonACC_GIA.py --dataset cora --reduction sparsification`

* `add_remove_evasion` evaluates the number of edges an attack added and removed, and the corresponding poisoned accuracies.

e.g. `python3 add_remove_evasion.py --dataset cora --attack dice`

* `analysis_removal_ratio.py` evaluates how much newly added perturbation edges are removed by graph reduction methods.

e.g. `python3 analysis_removal_ratio.py --dataset cora --attack dice --reduction sparsification`

* `analysis_coarsening.py` evaluates the feature distance and label difference ratio change between the clean coarsened graph and the poisoned coarsened graph.

e.g. `python3 analysis_coarsening.py --dataset cora --attack dice`

## Reference
```
@article{wu2024understanding,
  title={On the Adversarial Robustness of Graph Neural Networks with Graph Reduction},
  author={Wu, Kerui and Chow, Ka-Ho and Wei, Wenqi and Yu, Lei},
  journal={European Symposium on Research in Computer Security 2025},
  year={2025}
}
```
