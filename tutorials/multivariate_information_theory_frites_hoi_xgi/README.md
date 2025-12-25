# Multivariate Information Theory: A Practical Tutorial

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive, hands-on tutorial series for understanding and applying multivariate information theory to neuroscience and complex systems. This repository provides progressive notebooks that build from foundational concepts to complete analysis pipelines using state-of-the-art Python packages.

## 🎯 What You'll Learn

This tutorial series takes you from information theory basics to sophisticated analyses of higher-order interactions in neural data:

- **Information theory fundamentals**: Entropy, mutual information, and their multivariate extensions
- **Higher-order interactions**: Synergy, redundancy, and emergent phenomena like the XOR problem
- **Practical tools**: HOI, Frites, and XGI packages for real-world analysis
- **Time-resolved analysis**: Dynamic information flow in neural time series
- **Network representations**: Hypergraphs for higher-order structureshttps://www.int.univ-amu.fr/recherche-int/equipes/brainets
- **Complete pipelines**: End-to-end analysis from raw data to scientific insights

## 📚 Tutorial Structure

The tutorial consists of six progressive Jupyter notebooks, each building on the previous ones:

### [Notebook 1: Information Theory Foundations](01_information_theory_foundations.ipynb)
**Building the Mathematical Groundwork**

Start from scratch with the fundamental concepts of information theory. Learn what entropy really measures, how to compute mutual information, and why these measures matter for understanding dependencies.

**Key Topics:**
- Entropy for discrete and continuous variables
- Joint and conditional entropy
- Mutual information as a measure of dependence
- Conditional mutual information
- Hands-on implementations from first principles

**Prerequisites:** Basic Python, NumPy, and probability theory

---

### [Notebook 2: The XOR Problem - When Intuition Breaks](02_xor_problem_synergy.ipynb)
**Discovering Synergy and Higher-Order Effects**

Experience the mind-bending realization that two variables can each carry zero information individually yet contain perfect information together. This is the gateway to understanding why pairwise analysis is insufficient.

**Key Topics:**
- The XOR function and its information-theoretic properties
- Why Venn diagrams fail for 3+ variables
- Interaction information and its sign
- Synergy vs. redundancy: the fundamental distinction
- "Explaining away" and conditional independence

**Prerequisites:** Notebook 1

---

### [Notebook 3: Higher-Order Interactions with HOI](03_hoi_higher_order_interactions.ipynb)
**Scaling to Professional Analysis Tools**

Transition from manual calculations to the HOI package, designed specifically for computing higher-order information measures on real datasets. Learn to detect synergistic ensembles and decompose information systematically.

**Key Topics:**
- O-information: detecting synergy vs. redundancy
- S-information: measuring total complexity
- Partial Information Decomposition (PID)
- Multiple estimators: Gaussian copula, KNN, binning
- Computational scaling and optimization
- Application to neural population codes

**Prerequisites:** Notebooks 1-2

---

### [Notebook 4: Neural Time Series Analysis with Frites](04_frites_neural_timeseries.ipynb)
**From Static Snapshots to Dynamic Information Flow**

Move beyond static analysis to track how information emerges and flows through neural systems over time. Learn rigorous statistical testing to distinguish real effects from noise.

**Key Topics:**
- Time-resolved mutual information
- Permutation testing for significance
- Cluster-based correction for multiple comparisons
- Dynamic functional connectivity
- Fixed-effect vs. random-effect analysis
- Gaussian Copula MI (GCMI)
- Integration with MNE-Python for M/EEG data

**Prerequisites:** Notebooks 1-3

---

### [Notebook 5: Hypergraph Networks with XGI](05_xgi_hypergraph_networks.ipynb)
**From Information Measures to Network Topology**

Represent higher-order interactions as network structures using hypergraphs. Learn why traditional pairwise networks are insufficient and how hypergraphs naturally capture the structures discovered by information theory.

**Key Topics:**
- Hypergraphs: edges connecting multiple nodes simultaneously
- Creating and manipulating hypergraphs with XGI
- Mapping HOI results to hyperedges
- Hypergraph statistics: degree, clustering, connectivity
- Visualizing higher-order structures
- Temporal hypergraph sequences
- Network topology in neuroscience context

**Prerequisites:** Notebooks 1-4

---

### [Notebook 6: Complete Integration Pipeline](06_complete_integration.ipynb)
**From Raw Data to Scientific Insight**

The synthesis: a complete analysis pipeline integrating HOI, Frites, and XGI to answer a realistic neuroscience question about visual decision-making. This represents a publication-ready workflow you can adapt for your own research.

**Research Question:**
How do hierarchical brain regions (V1 → MT → LIP → PFC) interact synergistically to transform sensory evidence into decisions?

**Analysis Components:**
- When does information flow through the hierarchy? (Frites)
- Which ensembles show synergistic coding? (HOI)
- How is the network organized topologically? (XGI)
- Does synergy emerge early (sensory) or late (decision)?

**Prerequisites:** Notebooks 1-5

## 🚀 Getting Started

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/lordgrilo/cnww-hoi.git
cd cnww-hoi
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

The main packages you'll be using:
- `numpy`, `scipy`, `matplotlib`, `seaborn`: Scientific computing and visualization
- `hoi`: Higher-Order Interactions package
- `frites`: Framework for Information Theoretical analysis
- `xgi`: Comple(X) Group Interactions (hypergraphs)
- `jax`, `jaxlib`: Fast numerical computation (backend for HOI)
- `mne`: Neurophysiological data analysis (used by Frites)

4. **Launch Jupyter:**
```bash
jupyter notebook
```

Start with `01_information_theory_foundations.ipynb` and progress sequentially.

### Quick Start for Experienced Users

If you're already familiar with information theory, you can jump to:
- **Notebook 3** for HOI package usage
- **Notebook 4** for time-resolved analysis with Frites
- **Notebook 5** for hypergraph network representations
- **Notebook 6** for the complete integration pipeline

## 📖 Learning Path

### For Beginners
**Recommended time**: 10-15 hours total

1. **Week 1**: Notebooks 1-2 (foundations and intuition)
2. **Week 2**: Notebook 3 (HOI package)
3. **Week 3**: Notebooks 4-5 (Frites and XGI)
4. **Week 4**: Notebook 6 (integration)

### For Experienced Practitioners
**Recommended time**: 3-5 hours

- Skim Notebooks 1-2 for notation
- Focus on Notebooks 3-6 for practical tools
- Adapt Notebook 6 pipeline to your data

## 🔬 Research Applications

These techniques are applicable to:

**Neuroscience:**
- Neural population coding and ensemble analysis
- Brain network connectivity (fMRI, M/EEG, electrophysiology)
- Information routing and cognitive processing
- Consciousness and integrated information

**Complex Systems:**
- Gene regulatory networks
- Ecological community interactions
- Social network dynamics
- Climate system dependencies

**Machine Learning:**
- Feature interaction analysis
- Model interpretability
- Emergent computational properties
- Deep network representations

## 🛠️ Technical Details

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended for large datasets)
- **GPU**: Optional but recommended for HOI on large datasets (JAX supports CUDA)
- **OS**: Linux, macOS, or Windows

### Package Versions

See `requirements.txt` for specific versions. Key compatibility notes:
- JAX installation varies by platform (CPU vs GPU)
- Frites requires MNE-Python >= 0.23
- XGI works with NetworkX but extends it

## 📊 Data and Reproducibility

All notebooks use simulated data generated with fixed random seeds for reproducibility. The simulation parameters are based on realistic neuroscience scenarios, allowing you to:

- Understand exact ground truth
- Validate methods systematically
- Explore parameter spaces
- Adapt to your own data

## 🤝 Contributing

Contributions are welcome! Areas for contribution:

- Additional examples from other domains
- Improved visualizations
- Performance optimizations
- Bug fixes and clarifications
- Translations

Please open an issue first to discuss major changes.

## 📚 References and Further Reading

### Key Papers

**Information Theory:**
- Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*.
- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*. Wiley.

**Higher-Order Interactions:**
- Rosas, F. E., et al. (2019). Quantifying high-order interdependencies via multivariate extensions of the mutual information. *Physical Review E*.
- Williams, P. L., & Beer, R. D. (2010). Nonnegative decomposition of multivariate information. *arXiv:1004.2515*.
- Timme, N., et al. (2014). Synergy, redundancy, and multivariate information measures: an experimentalist's perspective. *Journal of Computational Neuroscience*.

**Neuroscience Applications:**
- Schneidman, E., et al. (2006). Weak pairwise correlations imply strongly correlated network states. *Nature*.
- Panzeri, S., et al. (2017). Cracking the neural code for sensory perception by combining statistics and interventions. *Neuron*.
- Stringer, C., et al. (2019). High-dimensional geometry of population responses in visual cortex. *Nature*.

### Software Documentation

- **HOI**: https://hoi.readthedocs.io/
- **Frites**: https://brainets.github.io/frites/
- **XGI**: https://xgi.readthedocs.io/
- **JAX**: https://jax.readthedocs.io/
- **MNE-Python**: https://mne.tools/

## 📧 Contact and Support

- **Issues**: Please use GitHub Issues for bugs and questions
- **Discussions**: GitHub Discussions for general questions and sharing results
- **Email**: [Your contact email]

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This tutorial integrates three excellent open-source packages:

- **HOI** by Etienne Combrisson and BraiNets team https://www.int.univ-amu.fr/recherche-int/equipes/brainets
- **Frites** by Etienne Combrisson, Matteo Neri and BraiNets team https://www.int.univ-amu.fr/recherche-int/equipes/brainets
- **XGI** by the Comple(X) Group Interactions team

Thanks to the broader neuroscience and complex systems communities for developing the theoretical foundations these tools implement.

---

**Happy Learning! 🧠💡**

*Questions? Found a bug? Have suggestions? Open an issue or start a discussion!*
