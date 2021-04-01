
#include "helper.hpp"
#include "reduction.hpp"
#include "scoring.hpp"

#include <cfloat>
#include <iostream>


/* Individual {SNP, allele} population count calculation, used in objectiveFunctionKernel() in 2-way and 3-way searches. 
   Counts for third genotype are derived from the other two genotypes. */

__global__ void epistasis_individualSNPs(int start_SNP_idx, uint *datasetCases, uint *datasetControls, uint *output_individualSNP_popcountsForCases, uint *output_individualSNP_popcountsForControls, int numSNPs, int casesSizeOriginal, int controlsSizeOriginal)
{

        uint SNP_i = start_SNP_idx + blockDim.x * blockIdx.x + threadIdx.x;
        int cases_i, controls_i;

        int casesSizeNoPadding = ceil(((float) casesSizeOriginal) / 32.0f);
        int controlsSizeNoPadding = ceil(((float) controlsSizeOriginal) / 32.0f);

	int casesSize = ceil(((float) casesSizeOriginal) / ((float) PADDING_SAMPLES)) * PADDING_SAMPLES / 32;
	int controlsSize = ceil(((float) controlsSizeOriginal) / ((float) PADDING_SAMPLES)) * PADDING_SAMPLES / 32;

        int casesZerosAcc = 0;
        int casesOnesAcc = 0;

        int controlsZerosAcc = 0;
        int controlsOnesAcc = 0;

        if(SNP_i < numSNPs) {           // To ensure processing is inside bounds.

                for(cases_i = 0; cases_i < casesSizeNoPadding; cases_i++) {
                        casesZerosAcc += __popc(datasetCases[SNP_i * SNP_CALC * casesSize + cases_i]);
                        casesOnesAcc += __popc(datasetCases[SNP_i * SNP_CALC * casesSize + casesSize + cases_i]);
                }

                output_individualSNP_popcountsForCases[0 * numSNPs + SNP_i] = casesZerosAcc;
                output_individualSNP_popcountsForCases[1 * numSNPs + SNP_i] = casesOnesAcc;
                output_individualSNP_popcountsForCases[2 * numSNPs + SNP_i] = casesSizeOriginal - (casesZerosAcc + casesOnesAcc);       	

                for(controls_i = 0; controls_i < controlsSizeNoPadding; controls_i++) {
                        controlsZerosAcc += __popc(datasetControls[SNP_i * SNP_CALC * controlsSize + controls_i]);
                        controlsOnesAcc += __popc(datasetControls[SNP_i * SNP_CALC * controlsSize + controlsSize + controls_i]);
                }

                output_individualSNP_popcountsForControls[0 * numSNPs + SNP_i] = controlsZerosAcc;
                output_individualSNP_popcountsForControls[1 * numSNPs + SNP_i] = controlsOnesAcc;
                output_individualSNP_popcountsForControls[2 * numSNPs + SNP_i] = controlsSizeOriginal - (controlsZerosAcc + controlsOnesAcc);  

        }
}


/* Pair-wise population count calculation, used in used in objectiveFunctionKernel() in 3-way searches.
   Data for third genotype is derived from the other two genotypes. */  

__global__ void epistasis_pairwiseSNPs(uint *datasetCases, uint *datasetControls, uint *output_pairwiseSNP_popcountsForCases, uint *output_pairwiseSNP_popcountsForControls, int numSNPs, int casesSizeOriginal, int controlsSizeOriginal, uint SNP_A_start)
{
        uint SNP_A_i_fromBlockStart = blockDim.x * blockIdx.x + threadIdx.x;
	uint SNP_A_i = SNP_A_start + SNP_A_i_fromBlockStart;
        uint SNP_B_i = blockDim.y * blockIdx.y + threadIdx.y;

        int cases_i, controls_i;

	int casesSizeNoPadding = ceil(((float) casesSizeOriginal) / 32.0f);
	int controlsSizeNoPadding = ceil(((float) controlsSizeOriginal) / 32.0f);

	int casesSize = ceil(((float) casesSizeOriginal) / ((float) PADDING_SAMPLES)) * PADDING_SAMPLES / 32;
	int controlsSize = ceil(((float) controlsSizeOriginal) / ((float) PADDING_SAMPLES)) * PADDING_SAMPLES / 32;

	uint maskRelevantBitsSetCases = (~0u) << (casesSizeNoPadding * 32 - casesSizeOriginal); // Mask where the only bits not set are the 'casesSize * 32 - casesSizeOriginal' less significant bits.
        uint maskRelevantBitsSetControls = (~0u) << (controlsSizeNoPadding * 32 - controlsSizeOriginal); // Mask where the only bits not set are the 'controlsSize * 32 - controlsSizeOriginal' less significant bits.

        if((SNP_A_i < numSNPs) && (SNP_B_i < numSNPs)) {       // This is because there may be more threads launched than combinations.

		int casesCountsArr[9];
		for(int i=0; i<9; i++) {
			casesCountsArr[i] = 0;
		}

		int controlsCountsArr[9];
		for(int i=0; i<9; i++) {
			controlsCountsArr[i] = 0;
		}

		unsigned int cases_0_A, cases_1_A, cases_2_A, cases_0_B, cases_1_B, cases_2_B;
                for(cases_i = 0; cases_i < (casesSizeNoPadding - 1); cases_i++) {

                        cases_0_A = datasetCases[SNP_A_i * SNP_CALC * casesSize + cases_i];
                        cases_1_A = datasetCases[SNP_A_i * SNP_CALC * casesSize + casesSize + cases_i];
                        cases_2_A = ~(cases_0_A | cases_1_A);

                        cases_0_B = datasetCases[SNP_B_i * SNP_CALC * casesSize + cases_i];
                        cases_1_B = datasetCases[SNP_B_i * SNP_CALC * casesSize + casesSize + cases_i];
                        cases_2_B = ~(cases_0_B | cases_1_B);

                        casesCountsArr[0] += __popc(cases_0_A & cases_0_B);
                        casesCountsArr[1] += __popc(cases_0_A & cases_1_B);
                        casesCountsArr[2] += __popc(cases_0_A & cases_2_B);
                        casesCountsArr[3] += __popc(cases_1_A & cases_0_B);
                        casesCountsArr[4] += __popc(cases_1_A & cases_1_B);
                        casesCountsArr[5] += __popc(cases_1_A & cases_2_B);
                        casesCountsArr[6] += __popc(cases_2_A & cases_0_B);
                        casesCountsArr[7] += __popc(cases_2_A & cases_1_B);
                        casesCountsArr[8] += __popc(cases_2_A & cases_2_B);
		}

		/* Processes last 32-bit bit-pack in order to take into acount when number of cases is not multiple of 32. */

		cases_0_A = datasetCases[SNP_A_i * SNP_CALC * casesSize + cases_i];
		cases_1_A = datasetCases[SNP_A_i * SNP_CALC * casesSize + casesSize + cases_i];
		cases_2_A = (~(cases_0_A | cases_1_A)) & maskRelevantBitsSetCases;
			
		cases_0_B = datasetCases[SNP_B_i * SNP_CALC * casesSize + cases_i];
		cases_1_B = datasetCases[SNP_B_i * SNP_CALC * casesSize + casesSize + cases_i];
		cases_2_B = (~(cases_0_B | cases_1_B)) & maskRelevantBitsSetCases;

		casesCountsArr[0] += __popc(cases_0_A & cases_0_B);
		casesCountsArr[1] += __popc(cases_0_A & cases_1_B);
		casesCountsArr[2] += __popc(cases_0_A & cases_2_B);
		casesCountsArr[3] += __popc(cases_1_A & cases_0_B);
		casesCountsArr[4] += __popc(cases_1_A & cases_1_B);
		casesCountsArr[5] += __popc(cases_1_A & cases_2_B);
		casesCountsArr[6] += __popc(cases_2_A & cases_0_B);
		casesCountsArr[7] += __popc(cases_2_A & cases_1_B);
		casesCountsArr[8] += __popc(cases_2_A & cases_2_B);


		output_pairwiseSNP_popcountsForCases[0 * (BLOCK_SIZE * numSNPs) + SNP_A_i_fromBlockStart * numSNPs + SNP_B_i] = casesCountsArr[0];
		output_pairwiseSNP_popcountsForCases[1 * (BLOCK_SIZE * numSNPs) + SNP_A_i_fromBlockStart * numSNPs + SNP_B_i] = casesCountsArr[1];
		output_pairwiseSNP_popcountsForCases[2 * (BLOCK_SIZE * numSNPs) + SNP_A_i_fromBlockStart * numSNPs + SNP_B_i] = casesCountsArr[2];
		output_pairwiseSNP_popcountsForCases[3 * (BLOCK_SIZE * numSNPs) + SNP_A_i_fromBlockStart * numSNPs + SNP_B_i] = casesCountsArr[3];
		output_pairwiseSNP_popcountsForCases[4 * (BLOCK_SIZE * numSNPs) + SNP_A_i_fromBlockStart * numSNPs + SNP_B_i] = casesCountsArr[4];
		output_pairwiseSNP_popcountsForCases[5 * (BLOCK_SIZE * numSNPs) + SNP_A_i_fromBlockStart * numSNPs + SNP_B_i] = casesCountsArr[5];
		output_pairwiseSNP_popcountsForCases[6 * (BLOCK_SIZE * numSNPs) + SNP_A_i_fromBlockStart * numSNPs + SNP_B_i] = casesCountsArr[6];
		output_pairwiseSNP_popcountsForCases[7 * (BLOCK_SIZE * numSNPs) + SNP_A_i_fromBlockStart * numSNPs + SNP_B_i] = casesCountsArr[7];
                output_pairwiseSNP_popcountsForCases[8 * (BLOCK_SIZE * numSNPs) + SNP_A_i_fromBlockStart * numSNPs + SNP_B_i] = casesCountsArr[8];


                unsigned int controls_0_A, controls_1_A, controls_2_A, controls_0_B, controls_1_B, controls_2_B;
                for(controls_i = 0; controls_i < (controlsSizeNoPadding - 1); controls_i++) {

                        controls_0_A = datasetControls[SNP_A_i * SNP_CALC * controlsSize + controls_i];
                        controls_1_A = datasetControls[SNP_A_i * SNP_CALC * controlsSize + controlsSize + controls_i];
                        controls_2_A = ~(controls_0_A | controls_1_A);

                        controls_0_B = datasetControls[SNP_B_i * SNP_CALC * controlsSize + controls_i];
                        controls_1_B = datasetControls[SNP_B_i * SNP_CALC * controlsSize + controlsSize + controls_i];
                        controls_2_B = ~(controls_0_B | controls_1_B);

                        controlsCountsArr[0] += __popc(controls_0_A & controls_0_B);
                        controlsCountsArr[1] += __popc(controls_0_A & controls_1_B);
                        controlsCountsArr[2] += __popc(controls_0_A & controls_2_B);
                        controlsCountsArr[3] += __popc(controls_1_A & controls_0_B);
                        controlsCountsArr[4] += __popc(controls_1_A & controls_1_B);
                        controlsCountsArr[5] += __popc(controls_1_A & controls_2_B);
                        controlsCountsArr[6] += __popc(controls_2_A & controls_0_B);
                        controlsCountsArr[7] += __popc(controls_2_A & controls_1_B);
                        controlsCountsArr[8] += __popc(controls_2_A & controls_2_B);
                }

                /* Processes last 32-bit bit-pack in order to take into acount when number of controls is not multiple of 32. */

                controls_0_A = datasetControls[SNP_A_i * SNP_CALC * controlsSize + controls_i];
                controls_1_A = datasetControls[SNP_A_i * SNP_CALC * controlsSize + controlsSize + controls_i];
                controls_2_A = (~(controls_0_A | controls_1_A)) & maskRelevantBitsSetControls;
                        
                controls_0_B = datasetControls[SNP_B_i * SNP_CALC * controlsSize + controls_i];
                controls_1_B = datasetControls[SNP_B_i * SNP_CALC * controlsSize + controlsSize + controls_i];
                controls_2_B = (~(controls_0_B | controls_1_B)) & maskRelevantBitsSetControls;

                controlsCountsArr[0] += __popc(controls_0_A & controls_0_B);
                controlsCountsArr[1] += __popc(controls_0_A & controls_1_B);
                controlsCountsArr[2] += __popc(controls_0_A & controls_2_B);
                controlsCountsArr[3] += __popc(controls_1_A & controls_0_B);
                controlsCountsArr[4] += __popc(controls_1_A & controls_1_B);
                controlsCountsArr[5] += __popc(controls_1_A & controls_2_B);
                controlsCountsArr[6] += __popc(controls_2_A & controls_0_B);
                controlsCountsArr[7] += __popc(controls_2_A & controls_1_B);
                controlsCountsArr[8] += __popc(controls_2_A & controls_2_B);


                output_pairwiseSNP_popcountsForControls[0 * (BLOCK_SIZE * numSNPs) + SNP_A_i_fromBlockStart * numSNPs + SNP_B_i] = controlsCountsArr[0];
                output_pairwiseSNP_popcountsForControls[1 * (BLOCK_SIZE * numSNPs) + SNP_A_i_fromBlockStart * numSNPs + SNP_B_i] = controlsCountsArr[1];
                output_pairwiseSNP_popcountsForControls[2 * (BLOCK_SIZE * numSNPs) + SNP_A_i_fromBlockStart * numSNPs + SNP_B_i] = controlsCountsArr[2];
                output_pairwiseSNP_popcountsForControls[3 * (BLOCK_SIZE * numSNPs) + SNP_A_i_fromBlockStart * numSNPs + SNP_B_i] = controlsCountsArr[3];
                output_pairwiseSNP_popcountsForControls[4 * (BLOCK_SIZE * numSNPs) + SNP_A_i_fromBlockStart * numSNPs + SNP_B_i] = controlsCountsArr[4];
                output_pairwiseSNP_popcountsForControls[5 * (BLOCK_SIZE * numSNPs) + SNP_A_i_fromBlockStart * numSNPs + SNP_B_i] = controlsCountsArr[5];
                output_pairwiseSNP_popcountsForControls[6 * (BLOCK_SIZE * numSNPs) + SNP_A_i_fromBlockStart * numSNPs + SNP_B_i] = controlsCountsArr[6];
                output_pairwiseSNP_popcountsForControls[7 * (BLOCK_SIZE * numSNPs) + SNP_A_i_fromBlockStart * numSNPs + SNP_B_i] = controlsCountsArr[7];
                output_pairwiseSNP_popcountsForControls[8 * (BLOCK_SIZE * numSNPs) + SNP_A_i_fromBlockStart * numSNPs + SNP_B_i] = controlsCountsArr[8];
        }
}


/* Computes population counts pertaining to pairwise interactions between a single SNP (X) and multiple other SNPs (Z).
   Used exclusivey in the context of 3-way searches. */

__global__ void epistasis_pairwiseSNPs_singleX_Z(uint *datasetCases, uint *datasetControls, uint *output_pairwiseSNP_singleX_Z_popcountsForCases, uint *output_pairwiseSNP_singleX_Z_popcountsForControls, int numSNPs, int casesSizeOriginal, int controlsSizeOriginal, uint SNP_X_i, uint start_Z)
{

	uint SNP_Z_i = start_Z + blockDim.x * blockIdx.x + threadIdx.x;
	uint patient_idx_thread = threadIdx.y;

	int cases_i, controls_i;

	int casesSizeNoPadding = ceil(((float) casesSizeOriginal) / 32.0f);
	int controlsSizeNoPadding = ceil(((float) controlsSizeOriginal) / 32.0f);

	int casesSize = ceil(((float) casesSizeOriginal) / ((float) PADDING_SAMPLES)) * PADDING_SAMPLES / 32;
	int controlsSize = ceil(((float) controlsSizeOriginal) / ((float) PADDING_SAMPLES)) * PADDING_SAMPLES / 32;

	uint maskRelevantBitsSetCases = (~0u) << (casesSizeNoPadding * 32 - casesSizeOriginal); // Mask where the only bits not set are the 'casesSize * 32 - casesSizeOriginal' less significant bits.
	uint maskRelevantBitsSetControls = (~0u) << (controlsSizeNoPadding * 32 - controlsSizeOriginal); // Mask where the only bits not set are the 'controlsSize * 32 - controlsSizeOriginal' less significant bits

	if((SNP_X_i < numSNPs) && (SNP_Z_i < numSNPs)) {       // To ensure processing is within bounds.

		int casesCountsArr[9];
		for(int i=0; i<9; i++) {
			casesCountsArr[i] = 0;
		}

		int controlsCountsArr[9];
		for(int i=0; i<9; i++) {
			controlsCountsArr[i] = 0;
		}

		int casesCountsArr_final[9];
		int controlsCountsArr_final[9];

		/* Sample data (cases or controls) is represented in 32-bit bitpacks and PADDING_SIZE (padding factor for controls and cases) is set to 1024.
		   Notice that 1024 / 32 --> 32, which is lower than 'blockDim.y' (128). 
		   Thus, only the last 'blockDim.y' bitpacks can have padding bits.  */ 

                unsigned int cases_0_A, cases_1_A, cases_2_A, cases_0_B, cases_1_B, cases_2_B;
		for(cases_i = 0; cases_i < (casesSize - blockDim.y); cases_i += blockDim.y) {

			cases_0_A = datasetCases[SNP_X_i * SNP_CALC * casesSize + cases_i + patient_idx_thread];
			cases_1_A = datasetCases[SNP_X_i * SNP_CALC * casesSize + casesSize + cases_i + patient_idx_thread];
			cases_2_A = ~(cases_0_A | cases_1_A);

			cases_0_B = datasetCases[SNP_Z_i * SNP_CALC * casesSize + cases_i + patient_idx_thread];
			cases_1_B = datasetCases[SNP_Z_i * SNP_CALC * casesSize + casesSize + cases_i + patient_idx_thread];
			cases_2_B = ~(cases_0_B | cases_1_B);

			casesCountsArr[0] += __popc(cases_0_A & cases_0_B);
			casesCountsArr[1] += __popc(cases_0_A & cases_1_B);
			casesCountsArr[2] += __popc(cases_0_A & cases_2_B);
			casesCountsArr[3] += __popc(cases_1_A & cases_0_B);
			casesCountsArr[4] += __popc(cases_1_A & cases_1_B);
			casesCountsArr[5] += __popc(cases_1_A & cases_2_B);
			casesCountsArr[6] += __popc(cases_2_A & cases_0_B);
			casesCountsArr[7] += __popc(cases_2_A & cases_1_B);
			casesCountsArr[8] += __popc(cases_2_A & cases_2_B);
		}

                /* Deals with last block of bit-packs to process, some of which can include padding bits */

		cases_0_A = datasetCases[SNP_X_i * SNP_CALC * casesSize + cases_i + patient_idx_thread];
		cases_1_A = datasetCases[SNP_X_i * SNP_CALC * casesSize + casesSize + cases_i + patient_idx_thread];

		cases_0_B = datasetCases[SNP_Z_i * SNP_CALC * casesSize + cases_i + patient_idx_thread];
		cases_1_B = datasetCases[SNP_Z_i * SNP_CALC * casesSize + casesSize + cases_i + patient_idx_thread];

		if((cases_i + patient_idx_thread) < (casesSizeNoPadding - 1)) {	// These bitpacks have no padding bits.
                	cases_2_A = ~(cases_0_A | cases_1_A);
                	cases_2_B = ~(cases_0_B | cases_1_B);
		}
		else if((cases_i + patient_idx_thread) == (casesSizeNoPadding - 1)) {	// This bitpack can have some padding bits.
                        cases_2_A = (~(cases_0_A | cases_1_A)) & maskRelevantBitsSetCases;  
                        cases_2_B = (~(cases_0_B | cases_1_B)) & maskRelevantBitsSetCases;
                }
		else {	// These bitpacks only have padding bits.
			cases_2_A = 0;
			cases_2_B = 0;
		}

		casesCountsArr[0] += __popc(cases_0_A & cases_0_B);
		casesCountsArr[1] += __popc(cases_0_A & cases_1_B);
		casesCountsArr[2] += __popc(cases_0_A & cases_2_B);
		casesCountsArr[3] += __popc(cases_1_A & cases_0_B);
		casesCountsArr[4] += __popc(cases_1_A & cases_1_B);
		casesCountsArr[5] += __popc(cases_1_A & cases_2_B);
		casesCountsArr[6] += __popc(cases_2_A & cases_0_B);
		casesCountsArr[7] += __popc(cases_2_A & cases_1_B);
		casesCountsArr[8] += __popc(cases_2_A & cases_2_B);

                /* Sum reduction between threads in thread block. */

		casesCountsArr_final[0] =  blockReduceSum(casesCountsArr[0]);
		casesCountsArr_final[1] =  blockReduceSum(casesCountsArr[1]);
		casesCountsArr_final[2] =  blockReduceSum(casesCountsArr[2]);
		casesCountsArr_final[3] =  blockReduceSum(casesCountsArr[3]);
		casesCountsArr_final[4] =  blockReduceSum(casesCountsArr[4]);
		casesCountsArr_final[5] =  blockReduceSum(casesCountsArr[5]);
		casesCountsArr_final[6] =  blockReduceSum(casesCountsArr[6]);
		casesCountsArr_final[7] =  blockReduceSum(casesCountsArr[7]);
		casesCountsArr_final[8] =  blockReduceSum(casesCountsArr[8]);

		if(threadIdx.y == 0) {
			output_pairwiseSNP_singleX_Z_popcountsForCases[0 * numSNPs + SNP_Z_i] = casesCountsArr_final[0];
			output_pairwiseSNP_singleX_Z_popcountsForCases[1 * numSNPs + SNP_Z_i] = casesCountsArr_final[1];
			output_pairwiseSNP_singleX_Z_popcountsForCases[2 * numSNPs + SNP_Z_i] = casesCountsArr_final[2];
			output_pairwiseSNP_singleX_Z_popcountsForCases[3 * numSNPs + SNP_Z_i] = casesCountsArr_final[3];
			output_pairwiseSNP_singleX_Z_popcountsForCases[4 * numSNPs + SNP_Z_i] = casesCountsArr_final[4];
			output_pairwiseSNP_singleX_Z_popcountsForCases[5 * numSNPs + SNP_Z_i] = casesCountsArr_final[5];
			output_pairwiseSNP_singleX_Z_popcountsForCases[6 * numSNPs + SNP_Z_i] = casesCountsArr_final[6];
			output_pairwiseSNP_singleX_Z_popcountsForCases[7 * numSNPs + SNP_Z_i] = casesCountsArr_final[7];
			output_pairwiseSNP_singleX_Z_popcountsForCases[8 * numSNPs + SNP_Z_i] = casesCountsArr_final[8];
		}

                unsigned int controls_0_A, controls_1_A, controls_2_A, controls_0_B, controls_1_B, controls_2_B;
		for(controls_i = 0; controls_i < (controlsSize - blockDim.y); controls_i += blockDim.y) {

			controls_0_A = datasetControls[SNP_X_i * SNP_CALC * controlsSize + controls_i + patient_idx_thread];
			controls_1_A = datasetControls[SNP_X_i * SNP_CALC * controlsSize + controlsSize + controls_i + patient_idx_thread];
			controls_2_A = ~(controls_0_A | controls_1_A);

			controls_0_B = datasetControls[SNP_Z_i * SNP_CALC * controlsSize + controls_i + patient_idx_thread];
			controls_1_B = datasetControls[SNP_Z_i * SNP_CALC * controlsSize + controlsSize + controls_i + patient_idx_thread];
			controls_2_B = ~(controls_0_B | controls_1_B);

			controlsCountsArr[0] += __popc(controls_0_A & controls_0_B);
			controlsCountsArr[1] += __popc(controls_0_A & controls_1_B);
			controlsCountsArr[2] += __popc(controls_0_A & controls_2_B);
			controlsCountsArr[3] += __popc(controls_1_A & controls_0_B);
			controlsCountsArr[4] += __popc(controls_1_A & controls_1_B);
			controlsCountsArr[5] += __popc(controls_1_A & controls_2_B);
			controlsCountsArr[6] += __popc(controls_2_A & controls_0_B);
			controlsCountsArr[7] += __popc(controls_2_A & controls_1_B);
			controlsCountsArr[8] += __popc(controls_2_A & controls_2_B);
		}

                /* Deals with last block of bit-packs to process, some of which can include padding bits */
                
		controls_0_A = datasetControls[SNP_X_i * SNP_CALC * controlsSize + controls_i + patient_idx_thread];
                controls_1_A = datasetControls[SNP_X_i * SNP_CALC * controlsSize + controlsSize + controls_i + patient_idx_thread];

                controls_0_B = datasetControls[SNP_Z_i * SNP_CALC * controlsSize + controls_i + patient_idx_thread];
                controls_1_B = datasetControls[SNP_Z_i * SNP_CALC * controlsSize + controlsSize + controls_i + patient_idx_thread];

		if((controls_i + patient_idx_thread) < (controlsSizeNoPadding - 1)) {  // These bitpacks have no padding bits.
			controls_2_A = ~(controls_0_A | controls_1_A);
			controls_2_B = ~(controls_0_B | controls_1_B);
		}
		else if((controls_i + patient_idx_thread) == (controlsSizeNoPadding - 1)) {    // This bitpack can have some padding bits.
			controls_2_A = (~(controls_0_A | controls_1_A)) & maskRelevantBitsSetControls;
			controls_2_B = (~(controls_0_B | controls_1_B)) & maskRelevantBitsSetControls;
		}
		else {  // These bitpacks only have padding bits.
			controls_2_A = 0;
			controls_2_B = 0;
		}

                controlsCountsArr[0] += __popc(controls_0_A & controls_0_B);
                controlsCountsArr[1] += __popc(controls_0_A & controls_1_B);
                controlsCountsArr[2] += __popc(controls_0_A & controls_2_B);
                controlsCountsArr[3] += __popc(controls_1_A & controls_0_B);
                controlsCountsArr[4] += __popc(controls_1_A & controls_1_B);
                controlsCountsArr[5] += __popc(controls_1_A & controls_2_B);
                controlsCountsArr[6] += __popc(controls_2_A & controls_0_B);
                controlsCountsArr[7] += __popc(controls_2_A & controls_1_B);
                controlsCountsArr[8] += __popc(controls_2_A & controls_2_B);

		/* Sum reduction between threads in thread block. */

		controlsCountsArr_final[0] =  blockReduceSum(controlsCountsArr[0]);
		controlsCountsArr_final[1] =  blockReduceSum(controlsCountsArr[1]);
		controlsCountsArr_final[2] =  blockReduceSum(controlsCountsArr[2]);
		controlsCountsArr_final[3] =  blockReduceSum(controlsCountsArr[3]);
		controlsCountsArr_final[4] =  blockReduceSum(controlsCountsArr[4]);
		controlsCountsArr_final[5] =  blockReduceSum(controlsCountsArr[5]);
		controlsCountsArr_final[6] =  blockReduceSum(controlsCountsArr[6]);
		controlsCountsArr_final[7] =  blockReduceSum(controlsCountsArr[7]);
		controlsCountsArr_final[8] =  blockReduceSum(controlsCountsArr[8]);

		if(threadIdx.y == 0) {
			output_pairwiseSNP_singleX_Z_popcountsForControls[0 * numSNPs + SNP_Z_i] = controlsCountsArr_final[0];
			output_pairwiseSNP_singleX_Z_popcountsForControls[1 * numSNPs + SNP_Z_i] = controlsCountsArr_final[1];
			output_pairwiseSNP_singleX_Z_popcountsForControls[2 * numSNPs + SNP_Z_i] = controlsCountsArr_final[2];
			output_pairwiseSNP_singleX_Z_popcountsForControls[3 * numSNPs + SNP_Z_i] = controlsCountsArr_final[3];
			output_pairwiseSNP_singleX_Z_popcountsForControls[4 * numSNPs + SNP_Z_i] = controlsCountsArr_final[4];
			output_pairwiseSNP_singleX_Z_popcountsForControls[5 * numSNPs + SNP_Z_i] = controlsCountsArr_final[5];
			output_pairwiseSNP_singleX_Z_popcountsForControls[6 * numSNPs + SNP_Z_i] = controlsCountsArr_final[6];
			output_pairwiseSNP_singleX_Z_popcountsForControls[7 * numSNPs + SNP_Z_i] = controlsCountsArr_final[7];
			output_pairwiseSNP_singleX_Z_popcountsForControls[8 * numSNPs + SNP_Z_i] = controlsCountsArr_final[8];
		}
	}
}


/* Combines allelic data from an SNP (X) with allelic data from a set of other SNPs (Y). 
   This computation phase enables the use of fused XOR+POPC on tensor cores in the context of high order searches (e.g. 3-way epistasis detection). */

__global__ void epistasis_prework_k3(uint *datasetCases, uint *datasetControls, uint *d_output_X_Y_SNP_ForCases, uint *d_output_X_Y_SNP_ForControls, int numSNPs, int casesSizeOriginal, int controlsSizeOriginal, int snp_X_index, int snp_Y_index_start)
{
	uint snp_Y_fromBlockStart = blockDim.x * blockIdx.x + threadIdx.x;
	uint snp_Y_index = snp_Y_index_start + snp_Y_fromBlockStart;

	uint patient_idx_thread = threadIdx.y;

	int cases_i, controls_i;

        int casesSize = ceil(((float) casesSizeOriginal) / ((float) PADDING_SAMPLES)) * PADDING_SAMPLES / 32;
	int controlsSize = ceil(((float) controlsSizeOriginal) / ((float) PADDING_SAMPLES)) * PADDING_SAMPLES / 32;

	if((snp_X_index < numSNPs) && (snp_Y_index < numSNPs)) {       // To ensure processing is within bounds. 

		for(int i=0; i<SNP_CALC; i++) {
			for(int j=0; j<SNP_CALC; j++) {

				// cases
				for(cases_i = 0; cases_i < casesSize; cases_i += blockDim.y) {
					d_output_X_Y_SNP_ForCases[snp_Y_fromBlockStart * (SNP_CALC * SNP_CALC) * casesSize + (i * SNP_CALC + j) * casesSize + cases_i + patient_idx_thread] = datasetCases[snp_X_index * SNP_CALC * casesSize + i * casesSize + cases_i + patient_idx_thread] & datasetCases[snp_Y_index * SNP_CALC * casesSize + j * casesSize + cases_i + patient_idx_thread];
				}

				// controls
				for(controls_i = 0; controls_i < controlsSize; controls_i += blockDim.y) {
					d_output_X_Y_SNP_ForControls[snp_Y_fromBlockStart * (SNP_CALC * SNP_CALC) * controlsSize + (i * SNP_CALC + j) * controlsSize + controls_i + patient_idx_thread] = datasetControls[snp_X_index * SNP_CALC * controlsSize + i * controlsSize + controls_i + patient_idx_thread] & datasetControls[snp_Y_index * SNP_CALC * controlsSize + j * controlsSize + controls_i + patient_idx_thread];
				}
			}
		}
	}
}


/* Construction of 2 x 2^k contingency table values from output of tensor cores.
 * Derivation of remaining 2 x (3^k - 2^k) contingency table values.
 * Scoring of sets of SNPs and idenfication of best score (and corresponding set).
 * These steps are performed taking into acount the sets processed in a given evaluation round.
 */


/* CUDA kernel used in 3-way searches.
   Template is used for performance purposes (avoid boundary checking inside loop when not required). */

template <bool doCheck> __global__ void objectiveFunctionKernel(int *C_ptrGPU_cases, int *C_ptrGPU_controls, uint * d_output_individualSNP_popcountsForCases, uint *d_output_individualSNP_popcountsForControls, uint * d_output_pairwiseSNP_popcountsForCases, uint * d_output_pairwiseSNP_popcountsForControls, float *tablePrecalc, float *output, unsigned long long int *output_packedIndices, int start_Y, int start_Z, int snp_X_index, uint * d_output_pairwiseSNP_singleX_Z_popcountsForCases, uint * d_output_pairwiseSNP_singleX_Z_popcountsForControls, int numSNPs, int numCases, int numControls)
{
	int global_id = blockDim.x * blockIdx.x + threadIdx.x;
	int local_id = threadIdx.x;

	int SNP_Y = global_id;
	
	int SNP_Z_withBestScore;		// Stores the index of the SNP (Z) that results in minimum score.

	#if defined(MI_SCORE)
	float score = FLT_MIN;
	#else
	float score = FLT_MAX;
	#endif

	if( (start_Y + SNP_Y) < numSNPs) {      // Ensures processing is within bounds.

		/* Copies data to registers. */
		int popcXY_cases_arr[9];
		int popcXY_controls_arr[9];
		for(int i=0; i<9; i++) {
			popcXY_cases_arr[i] = d_output_pairwiseSNP_singleX_Z_popcountsForCases[i * (numSNPs) + start_Y + SNP_Y];
			popcXY_controls_arr[i] = d_output_pairwiseSNP_singleX_Z_popcountsForControls[i * (numSNPs) + start_Y + SNP_Y];
		}


		for(int Z_i = 0; Z_i < BLOCK_OBJFUN; Z_i++) {

			int SNP_Z_index = (blockDim.y * blockIdx.y + threadIdx.y) * BLOCK_OBJFUN + Z_i;		

			if(doCheck == true) {	// Checks if computation is within bounds (only when dealing with the last block, in order to avoid significant overhead).
				if((start_Z + SNP_Z_index) >= numSNPs) {
					break;
				}
			}

			int calc_arr_cases[27];
			int calc_arr_controls[27];

			/* The frequency counts for the following 8 genotypes are determined from the output of the binarized tensor operations. */
			// {0,0,0}, {0,0,1}, {0,1,0}, {0,1,1}, {1,0,0}, {1,0,1}, {1,1,0}, {1,1,1}
			
			for(int i = 0; i<SNP_CALC; i++) {
				for(int j = 0; j<SNP_CALC; j++) {
					for(int k = 0; k<SNP_CALC; k++) {

						calc_arr_cases[i*9 + j*3 + k] = ((int)((popcXY_cases_arr[i*3 + j] + d_output_individualSNP_popcountsForCases[k * numSNPs + start_Z + SNP_Z_index]) - C_ptrGPU_cases[SNP_Z_index * (4 * BLOCK_SIZE) * 2 + k * (4 * BLOCK_SIZE) + SNP_Y * 4 + (2 * i + j)])) >> 1;        // / 2.0f; (does not affect overall performance)
						calc_arr_controls[i*9 + j*3 + k] = ((int)((popcXY_controls_arr[i*3 + j] + d_output_individualSNP_popcountsForControls[k * numSNPs + start_Z + SNP_Z_index]) - C_ptrGPU_controls[SNP_Z_index * (4 * BLOCK_SIZE) * 2 + k * (4 * BLOCK_SIZE) + SNP_Y * 4 + (2 * i + j)])) >> 1;    // / 2.0f; (does not affect overall performance)
					}
				}
			}

			/* The frequency counts for the following 19 genotypes are analytically derived with simple arithmetic operations. */
			// {0,0,2}, {0,1,2}, {0,2,0}, {0,2,1}, {0,2,2}, {1,0,2}, {1,1,2}, {1,2,0}, {1,2,1}, {1,2,2}, {2,0,0}, {2,0,1}, {2,0,2}, {2,1,0}, {2,1,1}, {2,1,2}, {2,2,0}, {2,2,1}, {2,2,2}
			
			// $\{0,0,2\}$ & $\{0,0,:\} - (\{0,0,0\} + \{0,0,1\})$
			CALC_MACRO_X_Y(0,0,2, snp_X_index, SNP_Y, 0,0, 0,0,0, 0,0,1);

			// $\{0,1,2\}$ & $\{0,1,:\} - (\{0,1,0\} + \{0,1,1\})$
			CALC_MACRO_X_Y(0,1,2, snp_X_index, SNP_Y, 0,1, 0,1,0, 0,1,1);

			// $\{0,2,0\}$ & $\{0,:,0\} - (\{0,0,0\} + \{0,1,0\})$
			CALC_MACRO_X_Z(0,2,0, snp_X_index, start_Z + SNP_Z_index, 0,0, 0,0,0, 0,1,0);

			// $\{0,2,1\}$ & $\{0,:,1\} - (\{0,0,1\} + \{0,1,1\})$
			CALC_MACRO_X_Z(0,2,1, snp_X_index, start_Z + SNP_Z_index, 0,1, 0,0,1, 0,1,1);

			// $\{0,2,2\}$ & $\{0,:,2\} - (\{0,0,2\} + \{0,1,2\})$
			CALC_MACRO_X_Z(0,2,2, snp_X_index, start_Z + SNP_Z_index, 0,2, 0,0,2, 0,1,2);

			// $\{1,0,2\}$ & $\{1,0,:\} - (\{1,0,0\} + \{1,0,1\})$
			CALC_MACRO_X_Y(1,0,2, snp_X_index, SNP_Y, 1,0, 1,0,0, 1,0,1);

			// $\{1,1,2\}$ & $\{1,1,:\} - (\{1,1,0\} + \{1,1,1\})$
			CALC_MACRO_X_Y(1,1,2, snp_X_index, SNP_Y, 1,1, 1,1,0, 1,1,1);

			// $\{1,2,0\}$ & $\{1,:,0\} - (\{1,0,0\} + \{1,1,0\})$
			CALC_MACRO_X_Z(1,2,0, snp_X_index, start_Z + SNP_Z_index, 1,0, 1,0,0, 1,1,0);

			// $\{1,2,1\}$ & $\{1,:,1\} - (\{1,0,1\} + \{1,1,1\})$
			CALC_MACRO_X_Z(1,2,1, snp_X_index, start_Z + SNP_Z_index, 1,1, 1,0,1, 1,1,1);

			// $\{1,2,2\}$ & $\{1,:,2\} - (\{1,0,2\} + \{1,1,2\})$
			CALC_MACRO_X_Z(1,2,2, snp_X_index, start_Z + SNP_Z_index, 1,2, 1,0,2, 1,1,2);

			// $\{2,0,0\}$ & $\{:,0,0\} - (\{0,0,0\} + \{1,0,0\})$
			CALC_MACRO_Y_Z(2,0,0, SNP_Y, start_Z + SNP_Z_index, 0,0, 0,0,0, 1,0,0);

			// $\{2,0,1\}$ & $\{:,0,1\} - (\{0,0,1\} + \{1,0,1\})$
			CALC_MACRO_Y_Z(2,0,1, SNP_Y, start_Z + SNP_Z_index, 0,1, 0,0,1, 1,0,1);

			// $\{2,0,2\}$ & $\{2,0,:\} - (\{2,0,0\} + \{2,0,1\})$
			CALC_MACRO_X_Y(2,0,2, snp_X_index, SNP_Y, 2,0, 2,0,0, 2,0,1);

			// $\{2,1,0\}$ & $\{:,1,0\} - (\{0,1,0\} + \{1,1,0\})$
			CALC_MACRO_Y_Z(2,1,0, SNP_Y, start_Z + SNP_Z_index, 1,0, 0,1,0, 1,1,0);

			// $\{2,1,1\}$ & $\{:,1,1\} - (\{0,1,1\} + \{1,1,1\})$
			CALC_MACRO_Y_Z(2,1,1, SNP_Y, start_Z + SNP_Z_index, 1,1, 0,1,1, 1,1,1);

			// $\{2,1,2\}$ & $\{2,1,:\} - (\{2,1,0\} + \{2,1,1\})$
			CALC_MACRO_X_Y(2,1,2, snp_X_index, SNP_Y, 2,1, 2,1,0, 2,1,1);

			// $\{2,2,0\}$ & $\{2,:,0\} - (\{2,0,0\} + \{2,1,0\})$
			CALC_MACRO_X_Z(2,2,0, snp_X_index, start_Z + SNP_Z_index, 2,0, 2,0,0, 2,1,0);

			// $\{2,2,1\}$ & $\{2,:,1\} - (\{2,0,1\} + \{2,1,1\})$
			CALC_MACRO_X_Z(2,2,1, snp_X_index, start_Z + SNP_Z_index, 2,1, 2,0,1, 2,1,1);

			// $\{2,2,2\}$ & $\{2,:,2\} - (\{2,0,2\} + \{2,1,2\})$
			CALC_MACRO_X_Z(2,2,2, snp_X_index, start_Z + SNP_Z_index, 2,2, 2,0,2, 2,1,2);

                        float score_new = scoring_triplets(calc_arr_cases, calc_arr_controls, tablePrecalc, numCases, numControls);
                        
			#if defined(MI_SCORE)
			if ((score_new >= score) && (snp_X_index < (start_Y + SNP_Y)) && ((start_Y + SNP_Y) < (start_Z + SNP_Z_index))) {
			#else
                        if ((score_new <= score) && (snp_X_index < (start_Y + SNP_Y)) && ((start_Y + SNP_Y) < (start_Z + SNP_Z_index))) {
			#endif
				SNP_Z_withBestScore = start_Z + SNP_Z_index;
				score = score_new;
			}
		}
	}
        #if defined(MI_SCORE)
	float min_score =  blockReduceMax_2D(score);
	if(local_id == 0 && threadIdx.y == 0) {
		atomicMax_g_f(output, min_score);
	}
	#else
	float min_score =  blockReduceMin_2D(score);
	if(local_id == 0 && threadIdx.y == 0) {
		atomicMin_g_f(output, min_score);
	}
	#endif

	if(score == min_score) {
		if((snp_X_index < (start_Y + SNP_Y)) &&  ((start_Y + SNP_Y) < SNP_Z_withBestScore)) {
			unsigned long long int packedIndices = (((unsigned long long int)snp_X_index) << 0) | (((unsigned long long int)(start_Y + SNP_Y)) << 21) | (((unsigned long long int)SNP_Z_withBestScore) << 42);
        		#if defined(MI_SCORE)
			atomicMaxGetIndex(output, min_score, output_packedIndices, packedIndices);
			#else
			atomicMinGetIndex(output, min_score, output_packedIndices, packedIndices);
			#endif
		}
	}
}

template __global__ void objectiveFunctionKernel<true>(int *C_ptrGPU_cases, int *C_ptrGPU_controls, uint * d_output_individualSNP_popcountsForCases, uint *d_output_individualSNP_popcountsForControls, uint * d_output_pairwiseSNP_popcountsForCases, uint * d_output_pairwiseSNP_popcountsForControls, float *tablePrecalc, float *output, unsigned long long int *output_packedIndices, int start_Y, int start_Z, int snp_X_index, uint * d_output_pairwiseSNP_singleX_Z_popcountsForCases, uint * d_output_pairwiseSNP_singleX_Z_popcountsForControls, int numSNPs, int numCases, int numControls);

template __global__ void objectiveFunctionKernel<false>(int *C_ptrGPU_cases, int *C_ptrGPU_controls, uint * d_output_individualSNP_popcountsForCases, uint *d_output_individualSNP_popcountsForControls, uint * d_output_pairwiseSNP_popcountsForCases, uint * d_output_pairwiseSNP_popcountsForControls, float *tablePrecalc, float *output, unsigned long long int *output_packedIndices, int start_Y, int start_Z, int snp_X_index, uint * d_output_pairwiseSNP_singleX_Z_popcountsForCases, uint * d_output_pairwiseSNP_singleX_Z_popcountsForControls, int numSNPs, int numCases, int numControls);


/* Used in 2-way searches. 
   Template is used for performance purposes (avoid boundary checking inside loop when not required). */

template <bool doCheck> __global__ void objectiveFunctionKernel(int *C_ptrGPU_cases, int *C_ptrGPU_controls, uint * d_output_individualSNP_popcountsForCases, uint *d_output_individualSNP_popcountsForControls, float *tablePrecalc, float *output, unsigned long long int *output_packedIndices, int start_A, int start_B, int numSNPs, int numCases, int numControls)
{
	int global_id = blockDim.x * blockIdx.x + threadIdx.x;
	int local_id = threadIdx.x;

	int SNP_A = global_id;
	int SNP_B_withBestScore; // Stores the index of the SNP (Y) that results in locally best score.

	#if defined(MI_SCORE)
	float score = FLT_MIN;
	#else
	float score = FLT_MAX;
	#endif

	if( (start_A + SNP_A) < numSNPs) {	// Ensures processing is within bounds.

		/* Copies data to registers. */
		int popcA_cases_arr[3];
		int popcA_controls_arr[3];
		for(int i=0; i<3; i++) {
			popcA_cases_arr[i] = d_output_individualSNP_popcountsForCases[i * numSNPs + start_A + SNP_A];
			popcA_controls_arr[i] = d_output_individualSNP_popcountsForControls[i * numSNPs + start_A + SNP_A];
		}

		for(int B_i = 0; B_i < BLOCK_OBJFUN; B_i++) {

			int SNP_B_index = (blockDim.y * blockIdx.y + threadIdx.y) * BLOCK_OBJFUN + B_i;

			if(doCheck == true) {	// Checks if computation is within bounds (only when dealing with the last block, in order to avoid significant overhead). 
				if((start_B + SNP_B_index) >= numSNPs) {
					break;
				}
			}

			int calc_arr_cases[9];
			int calc_arr_controls[9];

			/* The frequency counts for the following 4 genotypes are determined from the output of the binarized tensor operations. */
			// {0,0}, {0,1}, {1,0}, {1,1}
			
			for(int i = 0; i < SNP_CALC; i++) {
				for(int j = 0; j< SNP_CALC; j++) {
					calc_arr_cases[i*3+j] = ((popcA_cases_arr[i] + d_output_individualSNP_popcountsForCases[j * numSNPs + start_B + SNP_B_index]) - C_ptrGPU_cases[(2 * SNP_B_index * 2 * BLOCK_SIZE) + (j * 2 * BLOCK_SIZE) + (SNP_A * 2) + i]) / 2.0f;
					calc_arr_controls[i*3+j] = ((popcA_controls_arr[i] + d_output_individualSNP_popcountsForControls[j * numSNPs + start_B + SNP_B_index]) - C_ptrGPU_controls[(2 * SNP_B_index * 2 * BLOCK_SIZE) + (j * 2 * BLOCK_SIZE) + (SNP_A * 2) + i]) / 2.0f;
				}
			}

			/* The frequency counts for the the following 5 genotypes are analytically derived with simple arithmetic operations. */
			// {0,2}, {1,2}, {2,0}, {2,1}, {2,2}

			// {0,2}
			calc_arr_cases[0*3+2] = popcA_cases_arr[0] - (calc_arr_cases[0*3+0] + calc_arr_cases[0*3+1]);
			calc_arr_controls[0*3+2] = popcA_controls_arr[0] - (calc_arr_controls[0*3+0] + calc_arr_controls[0*3+1]);

			// {1,2}
			calc_arr_cases[1*3+2] = popcA_cases_arr[1] - (calc_arr_cases[1*3+0] + calc_arr_cases[1*3+1]);
			calc_arr_controls[1*3+2] = popcA_controls_arr[1] - (calc_arr_controls[1*3+0] + calc_arr_controls[1*3+1]);

			// {2,0}
			calc_arr_cases[2*3+0] =  d_output_individualSNP_popcountsForCases[0 * numSNPs + start_B + SNP_B_index] - (calc_arr_cases[0*3+0] + calc_arr_cases[1*3+0]);
			calc_arr_controls[2*3+0] =  d_output_individualSNP_popcountsForControls[0 * numSNPs + start_B + SNP_B_index] - (calc_arr_controls[0*3+0] + calc_arr_controls[1*3+0]);

			// {2,1}
			calc_arr_cases[2*3+1] =  d_output_individualSNP_popcountsForCases[1 * numSNPs + start_B + SNP_B_index] - (calc_arr_cases[0*3+1] + calc_arr_cases[1*3+1]);
			calc_arr_controls[2*3+1] =  d_output_individualSNP_popcountsForControls[1 * numSNPs + start_B + SNP_B_index] - (calc_arr_controls[0*3+1] + calc_arr_controls[1*3+1]);

			// {2,2}
			calc_arr_cases[2*3+2] =  popcA_cases_arr[2] - (calc_arr_cases[2*3+0] + calc_arr_cases[2*3+1]);
			calc_arr_controls[2*3+2] =  popcA_controls_arr[2] - (calc_arr_controls[2*3+0] + calc_arr_controls[2*3+1]);

			float score_new = scoring_pairs(calc_arr_cases, calc_arr_controls, tablePrecalc, numCases, numControls);

			#if defined(MI_SCORE)
			if ((score_new >= score) && ((start_A + SNP_A) < (start_B + SNP_B_index))) {
			#else
                        if ((score_new <= score) && ((start_A + SNP_A) < (start_B + SNP_B_index))) {
			#endif
				SNP_B_withBestScore = start_B + SNP_B_index;
				score = score_new;
			}
		}
	}
	#if defined(MI_SCORE)
	float min_score =  blockReduceMax_2D(score);
	if(local_id == 0 && threadIdx.y == 0) {
		atomicMax_g_f(output, min_score);
	}
	#else
	float min_score =  blockReduceMin_2D(score);
	if(local_id == 0 && threadIdx.y == 0) {
	        atomicMin_g_f(output, min_score);
	}
	#endif

	if(score == min_score) {
		if((start_A + SNP_A) < SNP_B_withBestScore) {
			unsigned long long int packedIndices = (((unsigned long long int)(start_A + SNP_A)) << 0) | (((unsigned long long int)SNP_B_withBestScore) << 32);
			#if defined(MI_SCORE)
			atomicMaxGetIndex(output, min_score, output_packedIndices, packedIndices);
			#else
			atomicMinGetIndex(output, min_score, output_packedIndices, packedIndices);
			#endif
		}
	}
}

template __global__ void objectiveFunctionKernel<true>(int *C_ptrGPU_cases, int *C_ptrGPU_controls, uint * d_output_individualSNP_popcountsForCases, uint *d_output_individualSNP_popcountsForControls, float *tablePrecalc, float *output, unsigned long long int *output_packedIndices, int start_A, int start_B, int numSNPs, int numCases, int numControls);

template __global__ void objectiveFunctionKernel<false>(int *C_ptrGPU_cases, int *C_ptrGPU_controls, uint * d_output_individualSNP_popcountsForCases, uint *d_output_individualSNP_popcountsForControls, float *tablePrecalc, float *output, unsigned long long int *output_packedIndices, int start_A, int start_B, int numSNPs, int numCases, int numControls);



/* CUTLASS XOR+POPC 1-bit GEMM-like kernel setup and launching on the CUDA device.
 * Used in the context of 2-way and 3-way searches.
 * Note that in CUTLASS documentation, ‘N’ refers to a column-major order and ‘T’ refers to a row-major order. 
 * M: GEMM M dimension, N: GEMM N dimension, K: GEMM K dimension, ...
 * ... A: matrix A operand, B: matrix B operand, C: matrix C.
 * Adapted from CUTLASS example code for performance testing.
 * Evaluated other thread/warp tile shapes and different values for other parameters, but overall those did not achieve as high performance. */

cudaError_t Cutlass_U1_WmmagemmTN(int M, int N, int K, ScalarBinary32 *A, int lda, ScalarBinary32 *B, int ldb, int *C, int ldc, cudaStream_t stream) {

	typedef cutlass::gemm::WmmaGemmTraits<
		cutlass::MatrixLayout::kRowMajor,   	// Matrix A layout (row-major)
		cutlass::MatrixLayout::kColumnMajor,   	// Matrix B layout (column-major)
		cutlass::Shape<1024, 128, 128>,         // Thread block tile shape/size
		cutlass::Vector<cutlass::bin1_t, 32>,   // Type of matrix A elements (1-bit)
		cutlass::Vector<cutlass::bin1_t, 32>,   // Type of matrix B elements (1-bit)
		int,                                   	// Type of matrix D elements (32-bit)
		cutlass::gemm::LinearScaling<int>,     	// Functor used to update output matrix
		int,                                   	// Type of matrix C elements (32-bit)
		cutlass::Shape<1024, 32, 64>,		// Warp tile shape/size  (<1024, 64, 32> also achieves high performance)
		cutlass::Shape<128, 8, 8>,            	// WMMA instruction tile shape/size	
		128,                                   	// scalars loaded when a thread loads from matrix A
		128                                    	// scalars loaded when a thread loads from matrix B
		>
		GemmTraits;

	typedef cutlass::gemm::Gemm<GemmTraits> Gemm;		// Defines a CUTLASS GEMM type from a GemmTraits<> instantiation.

	typename Gemm::Params params;				// Constructs and initializes CUTLASS GEMM parameters object.

	/* C = (alpha * A x B) + (beta * C)
	   Since alpha is set to 1 and beta is set to 0  -->  C = A x B   */
	int result = params.initialize(M, N, K, 1, A, lda, B, ldb, 0, C, ldc, C, ldc); 

	if (result) {
		std::cerr << "Failed to initialize CUTLASS Gemm::Params object." << std::endl;
		return cudaErrorInvalidValue;
	}

	/* Launches the CUTLASS binarized XOR+POPC GEMM-like kernel. */
	Gemm::launch(params, stream);

	/* Returns any errors associated with the launch or cudaSuccess if no error. */
	return cudaGetLastError();
}



/* Calculates nCk (number of choices of 'k' items from 'n' items),
   i.e.  --->  n! / (k!(n-k)!) 
*/

unsigned long long n_choose_k(unsigned int n, unsigned int k)
{
    unsigned long long result = 1;		// nC0

    for (unsigned int i = 1; i <= k; i++) {	// nC1 until nCk
        result = result * n / i;		// calculates nC_{i} from nC_{i-1}
        n = n - 1;
    }
    	
    return result;
}


