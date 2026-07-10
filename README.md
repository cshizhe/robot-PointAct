# PointAct

Official implementation of [PointACT: Vision-Language-Action Models with Multi-Scale Point-Action Interaction](https://cshizhe.github.io/projects/pointact.html).

<p align="center">
  <a href="https://cshizhe.github.io/projects/pointact.html">
    <img src="https://img.shields.io/badge/Project-PointAct_Webpage-0F172A?style=for-the-badge&logo=googlechrome&logoColor=white" alt="PointAct project webpage">
  </a>
  <a href="https://arxiv.org/abs/2605.21414">
    <img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="PointAct paper">
  </a>
  <a href="https://huggingface.co/cshizhe">
    <img src="https://img.shields.io/badge/Models_&_Data-Hugging_Face-F59E0B?style=for-the-badge&logo=huggingface&logoColor=white" alt="PointAct models and data">
  </a>
</p>

<table>
  <tr>
    <td align="center" width="33%">
      <img src="https://cshizhe.github.io/projects/resources/pointact/teaser_a.png" alt="Monolithic 3D-aware VLA teaser">
      <br>
      <sub><b>Monolithic 3D-aware VLA</b></sub>
    </td>
    <td align="center" width="33%">
      <img src="https://cshizhe.github.io/projects/resources/pointact/teaser_b.png" alt="Dual-system 3D-aware VLA teaser">
      <br>
      <sub><b>Dual-system 3D-aware VLA</b></sub>
    </td>
    <td align="center" width="33%">
      <img src="https://cshizhe.github.io/projects/resources/pointact/teaser_c.png" alt="PointAct teaser">
      <br>
      <sub><b>PointAct</b></sub>
    </td>
  </tr>
</table>

PointAct is a 3D-aware vision-language-action policy for robot manipulation. It keeps a pretrained vision-language backbone for semantic understanding and adds a dedicated point-action expert so that multi-scale 3D geometry can directly shape action decoding.

## Installation

Please follow the main setup guide in [INSTALLATION.md](INSTALLATION.md). The recommended workflow is:

1. Create the core `pointact` environment for training, checkpoint loading, preprocessing, and the policy server.
2. Create separate simulator environments.
3. Use the server-client evaluation pipeline since the policy and simulator usually need different environments.

## Supported Benchmarks

This repository currently supports three simulators:

| Benchmark | Simulator | Experiment Path |
| --- | --- | --- |
| [LIBERO](experiments/2_libero/README.md) | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | [`experiments/2_libero`](experiments/2_libero) |
| [RLBench](experiments/10_rlbench/README.md) | [RLBench](https://github.com/stepjam/RLBench) | [`experiments/10_rlbench`](experiments/10_rlbench) |
| [RoboCASA365](experiments/13_robocasa365/README.md) | [RoboCASA](https://github.com/robocasa/robocasa) | [`experiments/13_robocasa365`](experiments/13_robocasa365) |


## Supported Models

This repository includes PointAct and several comparison VLA policies used in our experiments:

| Model | Directory | Notes |
| --- | --- | --- |
| `PointAct` | [pointact/model/vla_pointact](pointact/model/vla_pointact) | Supports both [Concerto](https://arxiv.org/abs/2510.23607) and [Utonia](https://arxiv.org/abs/2603.03283) Point Transformer backbones |
| `EO1` | [pointact/model/eo1](pointact/model/eo1) | Monolithic VLA baseline |
| `EO1-Point` | [pointact/model/eo1](pointact/model/eo1) | EO-1 variant with point-cloud input |
| `QwenGR00T` | [pointact/model/vla_dual](pointact/model/vla_dual) | Dual-system VLA baseline |
| `QwenGR00T-Point` | [pointact/model/vla_dual](pointact/model/vla_dual) | VLA-Dual variant with point-cloud input |
| `Pi0` | [pointact/model/pi0](pointact/model/pi0) | PI0 baseline |
| `Pi0.5` | [pointact/model/pi05](pointact/model/pi05) | PI0.5 baseline |


## Acknowledgements

This codebase builds on several excellent open-source projects, especially [EO-1](http://eo-robotics.ai/eo-1), [GR00T](https://github.com/NVIDIA/Isaac-GR00T), and [LeRobot](https://github.com/huggingface/lerobot). We thank the authors and maintainers of these libraries for making their work available to the community.

## Citation

If you find PointAct useful in your research, or if you use this code, please cite:

```bibtex
@InProceedings{Chen_2026_PointACT,
    author    = {Chen, Shizhe and Pacaud, Paul and Schmid, Cordelia},
    title     = {PointACT: Vision-Language-Action Models with Multi-Scale Point-Action Interaction},
    booktitle = {Robotics: Science and Systems (RSS)},
    year      = {2026}
}
```
