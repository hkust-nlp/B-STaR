# B-STAR: Monitoring and Balancing Exploration and Exploitation in Self-Taught Reasoners

B-STAR (Balanced Self-Taught Reasoner) is a framework designed to improve the self-improvement process of reasoning models by dynamically balancing exploration and exploitation throughout training. This approach is particularly effective in enhancing performance in tasks requiring complex reasoning, such as mathematical problem-solving, coding, and commonsense reasoning.


![截屏2024-12-22 17 35 44](https://github.com/user-attachments/assets/fb97aec4-dbfa-45f3-a64a-f3022aeff599)


## Overview

Self-improvement in reasoning models involves iterative training where models generate their own training data from outputs. However, existing methods often stagnate after a few iterations due to imbalances between two critical factors:

1. **Exploration**: The model's ability to generate diverse and high-quality responses.
2. **Exploitation**: The effectiveness of external rewards in distinguishing and leveraging high-quality responses.

![截屏2024-12-22 17 40 13](https://github.com/user-attachments/assets/3970c997-8a9c-4c40-9c7a-4884b4897076)

B-STAR introduces an adaptive mechanism to monitor and balance these factors dynamically, ensuring consistent performance improvements over multiple training iterations


## Key Features

- **Dynamic Configuration Adjustments**: Automatically tunes exploration and exploitation configurations (e.g., sampling temperature, reward thresholds) to optimize the self-improvement process.
- **Balance Score Metric**: Quantifies the interplay between exploration and exploitation, guiding dynamic adjustments.
- **Generalization Across Tasks**: Demonstrates effectiveness in mathematical reasoning, coding challenges, and commonsense reasoning tasks


## Results

B-STAR achieves state-of-the-art performance across various benchmarks:

- Significant improvements compared to previsous self-improvement methods.
![截屏2024-12-22 17 39 06](https://github.com/user-attachments/assets/6fe32096-6099-49df-8824-f912ee31f71d)


- Sustained performance growth across multiple iterations, outperforming existing methods that stagnate after a few iterations.
![截屏2024-12-22 17 39 31](https://github.com/user-attachments/assets/76f35782-6617-4d54-a6ea-f9a89fe0b2bb)

## Code

Our code builds upon [easy-to-hard](https://github.com/Edward-Sun/easy-to-hard/tree/main) and [gpt-accelerate](https://github.com/Edward-Sun/gpt-accelera). Please refer to gpt-accelerate for environment setup and model weight conversion instructions.

Comming Soon!

## Citation

If you find B-STaR useful, please cite our paper:

```
@article{zeng2024bstar,
  title={B-STAR: Monitoring and Balancing Exploration and Exploitation in Self-Taught Reasoners},
  author={Weihao Zeng, Yuzhen Huang, Lulu Zhao, Yijun Wang, Zifei Shan, Junxian He},
  journal={arXiv preprint arXiv:2412.17256},
  year={2024},
  url={https://arxiv.org/abs/2412.17256}
}
```

  
