# GRASP: GRaph-Augmented Score Propagation for OOD detection
This is the source code for ICLR 2024 paper [Score Propagation as a Catalyst for Graph Out-of-distribution Detection: A Theoretical and Empirical Study](https://openreview.net/forum?id=R9CXfU2mD5&referrer=%5Bthe%20profile%20of%20Longfei%20Ma%5D(%2Fprofile%3Fid%3D~Longfei_Ma1)).

## Usage
**1. Post-hoc Methods**

  - Run scripts/{backbone}_exp.sh to  get pretrained In-distribution model for all datasets except wiki, where {backbone} can be replaced by gcn and h2gcn.

  - Run scripts/{backbone}_saintrw.sh with minibatch methods to  get pretrained In-distribution model for very large dataset wiki, where {backbone} can be replaced by gcn and h2gcn.

  - After getting pretrained ID models, run scripts/run_post_hoc.sh to evaluate various post-hoc OOD detection methods based on these pretrained ID methods.

  
**2. Training-based Methods**

- We provide implementations of three training-based methods in the literature: GKDE, GPN and OODGAT in scripts/run_training_based.sh, where these methods need to be trained from scratch.
