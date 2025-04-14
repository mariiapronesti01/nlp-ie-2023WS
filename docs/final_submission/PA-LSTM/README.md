## PA-LSTM
The code for running this model is taken from the original implementation proposed by Yuhao Zhang, one of the authors of the paper **Position-aware Attention and Supervised Data Improve Slot Filling**, and it is available at the following GitHub [repo](https://github.com/yuhaozhang/tacred-relation). 

## Error analysis
The error analysis is performed at different levels:
- `models_comparison.ipynb`: This notebook contains a comparison of the two PA-LSTM models, obtained considering both TACRED and TACREV dev sets, and a final comparison with the RF model.
- `relation_agnostic_error_analysis.ipynb`: This notebook contains attempts to identify relation-agnostic aspects of the sentences that affect the performance of the PA-LSTM model (sentence length, distance between entities).
- `relation_specific_error_analysis.ipynb`: In this notebook, an in-depth analysis of some groups of similar and inverse relations is carried out. 

## References
[1] Yuhao Zhang, Victor Zhong, Danqi Chen, Gabor Angeli, and Christopher D. Manning. Position-aware attention and supervised data improve slot filling. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 35–45, Copenhagen, Denmark, September 2017. Association for Computational Linguistics. doi: 10.18653/v1/D17-1004. URL https://aclanthology.org/D17-1004/
