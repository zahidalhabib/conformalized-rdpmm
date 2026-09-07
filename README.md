Conformalized R-DPMM

Distribution-free uncertainty sets for inventory demand segmentation.

The Conformalized Robust Dirichlet Process Mixture Model (RU-DPMM) segments a stock-keeping-unit (SKU) catalogue into demand classes without being told how many classes to expect, and attaches a calibrated uncertainty set to each segment. A variational Dirichlet process mixture infers the number of active classes from the data; four "levers" then turn each cluster into an actionable, distribution-free object:

Split-conformal calibration — gives each segment boundary a 90% coverage level (under exchangeability, conditional on the assigned cluster).
Anisotropic Bayesian-bootstrap margin — widens each set only in the directions in which its centroid is genuinely unstable.
Three-norm intersection set (l1 ∩ l2 ∩ l∞) — a compact whitened boundary.
Per-SKU uncertainty score in [0, 1] — flags items near a class boundary for manual safety-stock review.

The method is benchmarked against K-Means, GMM (EM), Bootstrap GMM, vanilla DPMM, Ward agglomeration, and HDBSCAN on three synthetic inventories (electronics, FMCG, pharmaceutical) and the real UCI Online Retail dataset.

Repository contents
File	Purpose
R_Dpmm_Conformalized.py	The RU-DPMM pipeline: preprocessing, variational DPMM, the four levers, and report/figure export.
Main_Clustering_Comparison_fixed.py	Benchmark harness comparing RU-DPMM against the baselines.
cluster_labeler.py, select_transform.py, robust_load.py	Preprocessing, scaler-selection, and robust-loading helpers.
run_*.py	Per-dataset pipeline runners.
*_inventory.csv	The three synthetic datasets.
requirements.txt	Pinned dependencies.
Requirements

Python 3.11. Install the dependencies:

pip install -r requirements.txt
Usage

1. Fit the pipeline on a dataset to produce its cluster labels and saved model. Either run the matching per-dataset script, e.g.

python run_electronics_inventory.py

or run python R_Dpmm_Conformalized.py after setting input_filename (near the bottom of the file) to the dataset you want. This writes <dataset>_clustered.csv and <dataset>_model.pkl.

2. Run the benchmark comparison (RU-DPMM vs. the baselines), which reads the raw dataset and the pipeline's clustered output:

python Main_Clustering_Comparison_fixed.py

Set DATASET_PATH and CLUSTERED_OUT_PATH at the top of the file to point at the dataset you want to benchmark. Outputs include the comparison table (CSV), metric bar charts, the capacity-sensitivity and assignment-stability plots, and the calibrated uncertainty-set figures.

All runs use a fixed random seed (42) and are deterministic.

Datasets
Synthetic — four-component Gaussian mixtures generated under seed 42: electronics_inventory.csv, retail_fmcg_inventory.csv, pharma_supply_inventory.csv.
Real — UCI Online Retail: https://archive.ics.uci.edu/dataset/352/online+retail
Author

S.M. Zahid Al Habib — B.Sc. thesis, Department of Industrial and Production Engineering, Bangladesh University of Engineering and Technology (BUET), 2026
