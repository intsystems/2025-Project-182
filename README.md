# Convergence of the loss function surface in transformer neural network architectures

[![License](https://badgen.net/github/license/intsystems/2025-Project-182?color=green)](https://github.com/intsystems/2025-Project-182/blob/main/LICENSE)
[![GitHub Contributors](https://img.shields.io/github/contributors/intsystems/2025-Project-182)](https://github.com/intsystems/2025-Project-182/graphs/contributors)
[![GitHub Issues](https://img.shields.io/github/issues-closed/intsystems/2025-Project-182.svg?color=0088ff)](https://github.com/intsystems/2025-Project-182/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr-closed/intsystems/2025-Project-182.svg?color=7f29d6)](https://github.com/intsystems/2025-Project-182/pulls)

<table>
    <tr>
        <td align="left"> <b> Author </b> </td>
        <td> Egor Petrov </td>
    </tr>
    <tr>
        <td align="left"> <b> Consultant </b> </td>
        <td> Nikita Kiselev </td>
    </tr>
    <tr>
        <td align="left"> <b> Advisor </b> </td>
        <td> Andrey Grabovoy, PhD </td>
    </tr>
</table>

## Assets

- [LinkReview](LINKREVIEW.md)
- [Code](code)
- [Paper](paper/main.pdf)
- [Slides](slides)

## Abstract

Training a neural network involves searching for the minimum point of the loss function, which defines the surface in the space of model parameters. The properties of this surface are determined by the chosen architecture, the loss function, and the training data. Existing studies show that as the number of objects in the sample increases, the surface of the loss function ceases to change significantly. The paper obtains an estimate for the convergence of the surface of the loss function for the transformer architecture of a neural network with attention layers, as well as conducts computational experiments that confirm the obtained theoretical results. In this paper, we propose a theoretical estimate for the minimum sample size required to train a model with any predetermined acceptable error, providing experiments that prove the theoretical boundaries.

## Citation

If you find our work helpful, please cite us.
```BibTeX
@article{petrov2025transformerlandscape,
    title={Convergence of the loss function surface in transformer neural network architectures},
    author={Egor Petrov, Nikita Kiselev, Vladislav Meshkov, Andrey Grabovoy},
    year={2025}
}
```

## Licence

Our project is MIT licensed. See [LICENSE](LICENSE) for details.
