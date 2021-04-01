# tensor-episdet

<p>
  <a href="https://doi.org/10.1109/TPDS.2021.3060322" alt="Publication">
    <img src="http://img.shields.io/badge/DOI-10.1109/TPDS.2021.3060322-blue.svg"/></a>
    
</p>

This repository includes implementations of two exhaustive epistasis detection algorithms targeting the NVIDIA Turing GPU architecture.
These implementations have been devised around the efficient use of novel tensor core capabilities introduced in the Turing architecture pertaining to support for tensorized fused XOR+POPC operations.
These operations, which are typically targeted at processing certain types of quantized neural networks, have been repurposed for achieving breakthrough performance in 2-way and 3-way epistasis detection searches with challenging datasets.
The tensor cores are accessed through the Warp Matrix Multiply-Accumulate (WMMA) API via the CUTLASS CUDA C++ template abstractions library.

## What is Epistasis Detection?

Epistasis detection is a computationally complex bioinformatics application with significant societal impact. It is used in the search of new correlations between genetic markers, such as single-nucleotide polymorphisms (SNPs), and phenotype (e.g. a particular disease state).
Finding new associations between genotype and phenotype can contribute to improved preventive care, personalized treatments and to the development of better drugs for more conditions.

## Setup

### Requirements

* CUDA Toolkit (version 10 or more recent)
* CUTLASS (version 1.3.X)
* NVIDIA GPU with tensor cores (minimum Turing architecture, e.g. Geforce 2060)

### Compilation

Compiling binary (`tensor-episdet.triplets.k2.bin`) for performing 3-way searches using K2 Bayesian scoring:

```bash
$ make triplets_k2
```

Compiling binary (`tensor-episdet.pairs.k2.bin`) for performing 2-way searches using K2 Bayesian scoring:
```bash
$ make pairs_k2
```

Mutual information scoring can be used instead of K2 scoring, through replacing `k2` by `mi` in the Makefile argument.

Important parameters such as the number of SNPs per block (`BLOCK_SIZE`) and the number of CUDA streams used to process different rounds (`NUM_STREAMS`) can be changed by modifying the Makefile.
Depending on the dataset characteristics, specializing these parameters (e.g. using a larger block size when processing datasets with more SNPs) can have a significant influence on the performance achieved.

The Makefile is expecting the CUTLASS library to be inside the project root directory in a folder named `cutlass`.
If you installed the library in a different directory, you must modify the Makefile accordingly.

Notice that the application expects that the input dataset is in a particular binarized format.
You can download an example dataset with 4096 SNPs and 262144 samples from <a href="https://drive.google.com/file/d/1htjD1QCr5_LEPo3udQEJ-5XUX4TK65JM/view?usp=sharing">here</a>.
Due to the way data is processed using matrix operations, the number of bits per {SNP, allele} in the dataset files (\*.bin) representing cases or controls (stored in different files) must be a multiple of 1024 bits. In situations where the number of cases or controls is not a multiple of 1024, the input binary dataset is expected to be padded with zeros (i.e. unset bits). 

## Usage example

Running a 3-way search with a synthetic example dataset with 4096 SNPs (11,444,858,880 triplets of SNPs to evaluate) and 262144 samples (131072 controls and 131072 cases):

```bash
$ ./bin/tensor-episdet.triplets.k2.bin datasets/db_4096snps_262144samples.txt   
```

This example is expected to take slightly less than 2 minutes to execute and to achieve a performance of above 25 tera sets (triplets) of SNPs processed per second (scaled to sample size), when executed on a system with a GeForce 2070S GPU.
Higher performance can be achieved when processing more challenging datasets with more SNPs.

The construction of contingency tables, a phase of epistasis detection searching that counts of occurrences of the possible genotypes in cases and controls resulting from combining pairs/triplets of SNPs, represents the most computationaly complex portion of the application.
Thus, running the same example with Mutual Information instead of K2 Bayesian scoring is expected to achieve comparable performance.


## In papers and reports, please refer to this tool as follows

R. Nobre, A. Ilic, S. Santander-Jiménez and L. Sousa, "Retargeting Tensor Accelerators for Epistasis Detection," in IEEE Transactions on Parallel and Distributed Systems, vol. 32, no. 9, pp. 2160-2174, 1 Sept. 2021, doi: 10.1109/TPDS.2021.3060322.

BibTeX:

    @ARTICLE{9357942,
    author={R. {Nobre} and A. {Ilic} and S. {Santander-Jiménez} and L. {Sousa}},
    journal={IEEE Transactions on Parallel and Distributed Systems}, 
    title={Retargeting Tensor Accelerators for Epistasis Detection}, 
    year={2021},
    volume={32},
    number={9},
    pages={2160-2174},
    doi={10.1109/TPDS.2021.3060322}}

For additional readings in high-throughput epistasis detection, you can take a look at our IPDPS 2020 and JSSPP 2020 papers.

