This repository contains the implementation of a multi-scale MRI super-resolution (SR) framework developed using PyTorch. Our framework is built on a distillation-driven conditional diffusion model with a focus on 1.5T-to-7T MRI SR, incorporating advanced guidance mechanisms like bias field correction and gradient nonlinearity correction. The progressive distillation strategy enables a lightweight student model to achieve high-resolution outputs comparable to the teacher model while significantly reducing computational complexity.

While adhering to the paper’s design, some implementation details have been adjusted for broader use cases and efficient training. Specifically:
- Employed the ResNet block and hierarchical feature concatenation for diffusion models.
- Applied multiscale attention mechanisms to better capture global and local anatomical features.
- Used student-teacher feature matching during distillation for precise structural alignment.
- Introduced a progressive subgoal strategy for refining MRI resolutions (e.g., 1.5T to 3T and 3T to 7T).
