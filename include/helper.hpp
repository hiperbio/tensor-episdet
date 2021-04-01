#ifndef HELPER_H_   
#define HELPER_H_

/* CUTLASS template abstractions library. */ 
#include "cutlass/wmma_matrix.h"                
#include "cutlass/gemm/gemm.h"                  
#include "cutlass/gemm/wmma_gemm_traits.h"      


/* Only two out of the possible three alleles are represented (2 bits per {SNP, sample} tuple). */
#define SNP_CALC 2

#if defined(TRIPLETS)
#define INTER_OR 3		// k=3 interation order.
#define SNP_COMB 27		// 3^3 (=27) possible genotypes.
#define SNP_COMB_CALC 8		// 2^3 (= 8) genotypes are calculated with matrix operations, remaining 19 are analytically derived.
#else
#define INTER_OR 2		// k=2 interation order.
#define SNP_COMB 9		// 3^2 (= 9) possible genotypes.
#define SNP_COMB_CALC 4		// 2^2 (= 4) genotypes are calculated with matrix operations, remaining 5 are analytically derived.
#endif

/* Controls matrix and cases matrix are expected to be padded to multiples of PADDING_SAMPLES in regard to samples ...
 * ... in order to take into account the tile shape used in the binarized matrix operations. */
#define PADDING_SAMPLES 1024    

/* Used in objectiveFunctionKernel() GPU kernel. */
#define BLOCK_OBJFUN	4


/* MACROS used in objectiveFuntion() kernel in the context of 3-way searches for deriving ...
 * ... genotype counts for 19 out of 27 possible genotypes. */

#define CALC_MACRO_X_Y(x, y, z, SNP_A_index, SNP_B_index, a, b, x1,y1,z1, x2,y2,z2); {\
                CALC_MACRO(x, y, z, SNP_B_index, SNP_A_index, b, a, x1,y1,z1, x2,y2,z2); \
        }

#define CALC_MACRO_X_Z(x, y, z, SNP_A_index, SNP_B_index, a, b, x1,y1,z1, x2,y2,z2) {\
                calc_arr_cases[x*9+y*3+z] = d_output_pairwiseSNP_singleX_Z_popcountsForCases[(a*3+b) * (numSNPs) + SNP_B_index] - (calc_arr_cases[x1*9+y1*3+z1] + calc_arr_cases[x2*9+y2*3+z2]);\
                calc_arr_controls[x*9+y*3+z] = d_output_pairwiseSNP_singleX_Z_popcountsForControls[(a*3+b) * (numSNPs) + SNP_B_index] - (calc_arr_controls[x1*9+y1*3+z1] + calc_arr_controls[x2*9+y2*3+z2]);\
        }

#define CALC_MACRO_Y_Z(x, y, z, SNP_A_index, SNP_B_index, a, b, x1,y1,z1, x2,y2,z2) {\
                CALC_MACRO(x, y, z, SNP_A_index, SNP_B_index, a, b, x1,y1,z1, x2,y2,z2); \
        }

/* {x,y,z} = {a,:,b} - ({x1,y1,z1} + {x2,y2,z2}) */
#define CALC_MACRO(x, y, z, SNP_A_index, SNP_B_index, a, b, x1,y1,z1, x2,y2,z2) {\
                calc_arr_cases[x*9+y*3+z] = d_output_pairwiseSNP_popcountsForCases[(a*3+b) * (BLOCK_SIZE * numSNPs) + ( SNP_A_index ) * numSNPs + SNP_B_index] - (calc_arr_cases[x1*9+y1*3+z1] + calc_arr_cases[x2*9+y2*3+z2]);\
                calc_arr_controls[x*9+y*3+z] = d_output_pairwiseSNP_popcountsForControls[(a*3+b) * (BLOCK_SIZE * numSNPs) + ( SNP_A_index ) * numSNPs + SNP_B_index] - (calc_arr_controls[x1*9+y1*3+z1] + calc_arr_controls[x2*9+y2*3+z2]);\
        }


/* Functions used as part of main loop for performing epistasis detection searches. */

__global__ void epistasis_individualSNPs(int start_SNP_idx, uint *datasetCases, uint *datasetControls, uint *output_individualSNP_popcountsForCases, uint *output_individualSNP_popcountsForControls, int numSNPs, int casesSizeOriginal, int controlsSizeOriginal);

__global__ void epistasis_pairwiseSNPs(uint *datasetCases, uint *datasetControls, uint *output_pairwiseSNP_popcountsForCases, uint *output_pairwiseSNP_popcountsForControls, int numSNPs, int casesSizeOriginal, int controlsSizeOriginal, uint SNP_A_start);	// Exclusive to 3-way searches.

__global__ void epistasis_pairwiseSNPs_singleX_Z(uint *datasetCases, uint *datasetControls, uint *output_pairwiseSNP_singleX_Z_popcountsForCases, uint *output_pairwiseSNP_singleX_Z_popcountsForControls, int numSNPs, int casesSizeOriginal, int controlsSizeOriginal, uint SNP_X_i, uint start_Z);	// Exclusive to 3-way searches.

__global__ void epistasis_prework_k3(uint *datasetCases, uint *datasetControls, uint *d_output_X_Y_SNP_ForCases, uint *d_output_X_Y_SNP_ForControls, int numSNPs, int casesSizeOriginal, int controlsSizeOriginal, int snp_X_index, int snp_Y_index_start);	// Exclusive to 3-way searches.

template <bool doCheck> __global__ void objectiveFunctionKernel(int *C_ptrGPU_cases, int *C_ptrGPU_controls, uint * d_output_individualSNP_popcountsForCases, uint *d_output_individualSNP_popcountsForControls, uint * d_output_pairwiseSNP_popcountsForCases, uint * d_output_pairwiseSNP_popcountsForControls, float *tablePrecalc, float *output, unsigned long long int *output_packedIndices, int start_Y, int start_Z, int snp_X_index, uint * d_output_pairwiseSNP_singleX_Z_popcountsForCases, uint * d_output_pairwiseSNP_singleX_Z_popcountsForControls, int numSNPs, int numCases, int numControls);	// Used in 3-way searches.

template <bool doCheck> __global__ void objectiveFunctionKernel(int *C_ptrGPU_cases, int *C_ptrGPU_controls, uint * d_output_individualSNP_popcountsForCases, uint *d_output_individualSNP_popcountsForControls, float *tablePrecalc, float *output, unsigned long long int *output_packedIndices, int start_A, int start_B, int numSNPs, int numCases, int numControls);	// Used in 2-way searches.


/* Used for accessing tensor cores to perform tensorized fused XOR+POPC operations. */

typedef typename cutlass::Vector<cutlass::bin1_t, 32> ScalarBinary32;	// Type of A and B input matrices

cudaError_t Cutlass_U1_WmmagemmTN(int M, int N, int K, ScalarBinary32 *A, int lda, ScalarBinary32 *B, int ldb, int *C, int ldc, cudaStream_t stream);


/* Calculates nCk, i.e. number of combinations from 'n' items taken 'k' at a time.
 * Used to calculate performance metric, i.e. number of unique sets (i.e. combinations) of SNPs processed per second scaled to sample size. */

unsigned long long n_choose_k(unsigned int n, unsigned int k);


#endif
