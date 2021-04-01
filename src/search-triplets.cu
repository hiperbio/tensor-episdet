#include <iostream>
#include <iomanip>      // std::setprecision
#include <cfloat>

#include "helper.hpp"


cudaError_t EpistasisDetectionSearch(unsigned int* datasetCases_host_matrixA, unsigned int* datasetControls_host_matrixA, int numSNPs, int numCases, int numControls, uint numSNPsWithPadding, int numCasesWithPadding, int numControlsWithPadding, int * roundsCounter, double * searchTime, float * outputFromGpu, unsigned long long int * output_indexFromGpu_packedIndices) {
	cudaError_t result;

	struct timespec t_start, t_end;
	clock_gettime(CLOCK_MONOTONIC, &t_start);       // initial timestamp


        /* Allocate GPU memory for Cases */

	ScalarBinary32 *cases_A_ptrGPU;
	ScalarBinary32 *cases_B_ptrGPU;
	result = cudaMalloc((ScalarBinary32 **) &cases_A_ptrGPU, sizeof(ScalarBinary32) * numSNPsWithPadding * (numCasesWithPadding / 32) * SNP_CALC);		
        if(result != cudaSuccess) {
                std::cerr << "Failed allocating memory for cases input data." << std::endl;
        }
	

	result = cudaMemcpyAsync(cases_A_ptrGPU, datasetCases_host_matrixA, sizeof(int) * numSNPsWithPadding * (numCasesWithPadding / 32) * SNP_CALC, cudaMemcpyHostToDevice, 0);

	cases_B_ptrGPU = cases_A_ptrGPU;	// Matrix B points to the same data as matrix A

	int *C_ptrGPU_cases;
	result = cudaMalloc((int**) &C_ptrGPU_cases, sizeof(int) * NUM_STREAMS * (BLOCK_SIZE * SNP_CALC * SNP_CALC) * (BLOCK_SIZE * SNP_CALC));        

	if(result != cudaSuccess) {
                std::cerr << "Failed allocating memory for cases output data." << std::endl;
	}


        /* Allocate GPU memory for Controls */

	ScalarBinary32 *controls_A_ptrGPU;
	ScalarBinary32 *controls_B_ptrGPU;
	result = cudaMalloc((ScalarBinary32 **) &controls_A_ptrGPU, sizeof(ScalarBinary32) * numSNPsWithPadding * (numControlsWithPadding / 32) * SNP_CALC);	
	if(result != cudaSuccess) {
                std::cerr << "Failed allocating memory for controls input data." << std::endl;
	}

	result = cudaMemcpyAsync(controls_A_ptrGPU, datasetControls_host_matrixA, sizeof(int) * numSNPsWithPadding * (numControlsWithPadding / 32) * SNP_CALC, cudaMemcpyHostToDevice, 0);

	controls_B_ptrGPU = controls_A_ptrGPU;	// Matrix B points to the same data as matrix A

	int *C_ptrGPU_controls;
	result = cudaMalloc((int**) &C_ptrGPU_controls, sizeof(int) * NUM_STREAMS * (BLOCK_SIZE * SNP_CALC * SNP_CALC) * (BLOCK_SIZE * SNP_CALC));          

	if(result != cudaSuccess) {
                std::cerr << "Failed allocating memory for controls output data." << std::endl;
	}

	/* Computes lookup table on the Host */

	float * d_tablePrecalc;

	#if defined(MI_SCORE)
	// Mutual Information scoring
        int tablePrecalc_size = numCases + numControls;
	float * h_tablePrecalc = (float*) malloc(tablePrecalc_size * sizeof(float));
	float numPatientsInv = 1.0 / (numCases + numControls);    
	h_tablePrecalc[0] = 0;
	for(int i=1; i < tablePrecalc_size; i++) {
		h_tablePrecalc[i] = log2((double)i * numPatientsInv);
	}
	#else 
	// K2 Bayesian scoring
        int tablePrecalc_size = max(numCases, numControls) + 1;
	float * h_tablePrecalc = (float*) malloc(tablePrecalc_size * sizeof(float));
	for(int i=1; i < (tablePrecalc_size + 1); i++) {
		h_tablePrecalc[i - 1] = lgamma((double)i);
	}
	#endif

        result = cudaMalloc((float**)&d_tablePrecalc, tablePrecalc_size * sizeof(float));
	result = cudaMemcpy(d_tablePrecalc, h_tablePrecalc, tablePrecalc_size * sizeof(float), cudaMemcpyHostToDevice);


	float * d_output;
	unsigned long long int * d_output_packedIndices;

	#if defined(MI_SCORE)
        float h_output[1] = {FLT_MIN};
	#else
	float h_output[1] = {FLT_MAX};
	#endif

        /* Stores the best score and indexes of corresponding set of SNPs */
	result = cudaMalloc((float**)&d_output, 1 * sizeof(float));								
	result = cudaMalloc((unsigned long long int**)&d_output_packedIndices, 1 * sizeof(unsigned long long int));		

	result = cudaMemcpy(d_output, h_output, 1 * sizeof(float), cudaMemcpyHostToDevice);


	uint A_leadingDim_cases = numCasesWithPadding / 32;   		
	uint B_leadingDim_cases = numCasesWithPadding / 32;   		

	uint A_leadingDim_controls = numControlsWithPadding / 32;   	
	uint B_leadingDim_controls = numControlsWithPadding / 32;   	

	uint C_leadingDim = BLOCK_SIZE * SNP_CALC * SNP_CALC;	


	/* Calculates individual population counts */

	uint * d_output_individualSNP_popcountsForCases;
	uint * d_output_individualSNP_popcountsForControls;

	int blocksPerGrid_ind = (size_t)ceil(((float)(numSNPs)) / ((float)32));
	result = cudaMalloc((uint**)&d_output_individualSNP_popcountsForControls, 3 * numSNPs * sizeof(uint));
	result = cudaMalloc((uint**)&d_output_individualSNP_popcountsForCases, 3 * numSNPs * sizeof(uint));

	epistasis_individualSNPs<<<blocksPerGrid_ind, 32, 0, 0>>>(0, (uint*)cases_A_ptrGPU, (uint*)controls_A_ptrGPU, d_output_individualSNP_popcountsForCases, d_output_individualSNP_popcountsForControls, numSNPs, numCases, numControls);  

	(*roundsCounter) = 0;	

	uint * d_output_X_Y_SNP_ForCases;
	uint * d_output_X_Y_SNP_ForControls;
	result = cudaMalloc((uint**)&d_output_X_Y_SNP_ForCases, (numCasesWithPadding / 32) * (SNP_CALC * SNP_CALC) * BLOCK_SIZE * sizeof(uint));		
	if(result != cudaSuccess) {
                std::cerr << "Failed allocating memory for cases pairwise popcounts." << std::endl;
	}
	result = cudaMalloc((uint**)&d_output_X_Y_SNP_ForControls, (numControlsWithPadding / 32) * (SNP_CALC * SNP_CALC) * BLOCK_SIZE * sizeof(uint));	
	if(result != cudaSuccess) {
                std::cerr << "Failed allocating memory for controls pairwise popcounts." << std::endl;
	}


	/* Calculates pairwise population counts */

	/* For storing the popcounts of all SNP pairs. */
	uint * d_output_pairwiseSNP_popcountsForCases;
	uint * d_output_pairwiseSNP_popcountsForControls;
	result = cudaMalloc((uint**)&d_output_pairwiseSNP_popcountsForControls, 9 * BLOCK_SIZE * numSNPs * sizeof(uint));
	result = cudaMalloc((uint**)&d_output_pairwiseSNP_popcountsForCases, 9 * BLOCK_SIZE * numSNPs * sizeof(uint));


	/* For storing the popcounts of all SNP pairs. */
	uint * d_output_pairwiseSNP_singleX_Z_popcountsForCases;
	uint * d_output_pairwiseSNP_singleX_Z_popcountsForControls;
	result = cudaMalloc((uint**)&d_output_pairwiseSNP_singleX_Z_popcountsForControls, 9 * 1 * numSNPs * sizeof(uint));
	result = cudaMalloc((uint**)&d_output_pairwiseSNP_singleX_Z_popcountsForCases, 9 * 1 * numSNPs * sizeof(uint));


	/* CUDA stream creation */

	cudaStream_t cudaStreamPairwiseSNPs, cudaStream_singleX_Z;
	cudaStreamCreate(&cudaStreamPairwiseSNPs);
	cudaStreamCreate(&cudaStream_singleX_Z);

	cudaStream_t cudaStreamToUse[NUM_STREAMS];
	for (int i = 0; i < NUM_STREAMS; i++) {
		cudaStreamCreate(&cudaStreamToUse[i]);
	}

	uint objectiveFunctionIndex = 0;

	for(int start_Y = 0; start_Y < numSNPsWithPadding; start_Y+=BLOCK_SIZE) {

                std::cout << "Outer loop iteration " << (int) (start_Y / BLOCK_SIZE) << " out of " << (int) (numSNPsWithPadding / BLOCK_SIZE) << std::endl;

		/* In case the last calls to the objective function kernel did not terminate yet at this point. */
		for(int i=0; i<NUM_STREAMS; i++) {
			cudaStreamSynchronize(cudaStreamToUse[i]);
		}

		/* Calculation of pairwise population counts as part of the 3-way search process. */

		dim3 blocksPerGrid_pairwise ( (size_t)ceil(((float)(BLOCK_SIZE)) / ((float)16)), (size_t)ceil(((float)(numSNPs)) / ((float)16)), 1 );
		dim3 workgroupSize_pairwise ( 16, 16, 1 );   

		epistasis_pairwiseSNPs<<<blocksPerGrid_pairwise, workgroupSize_pairwise, 0, cudaStreamPairwiseSNPs>>>((uint*)cases_A_ptrGPU, (uint*)controls_A_ptrGPU, d_output_pairwiseSNP_popcountsForCases, d_output_pairwiseSNP_popcountsForControls, numSNPs, numCases, numControls, start_Y);  


		for(int X_index=0; (X_index < (start_Y + BLOCK_SIZE)) && (X_index < numSNPs); X_index++) {

			int start_Y_initial = start_Y;

			/* In case some calls to CUTLASS kernel did not terminate yet at this point. */
			for(int i=0; i<NUM_STREAMS; i++) {
				cudaStreamSynchronize(cudaStreamToUse[i]);
			}

			/* Combines an SNP X with a block of SNPs Y. */
			dim3 blocksPerGrid_prework_k3( (size_t)ceil(((float)(BLOCK_SIZE)) / ((float)1)), 1, 1);
			dim3 workgroupSize_prework_k3( 1, 128, 1 );
			epistasis_prework_k3<<<blocksPerGrid_prework_k3, workgroupSize_prework_k3, 0, cudaStreamToUse[(objectiveFunctionIndex % NUM_STREAMS)] >>>(((uint*)cases_A_ptrGPU), ((uint*)controls_A_ptrGPU), d_output_X_Y_SNP_ForCases, d_output_X_Y_SNP_ForControls, numSNPs, numCases, numControls, X_index, start_Y_initial);

			/* Calculates pairwise population counts between a single SNP X and all SNPs with index larger or equal to 'start_Y'.
			   Takes into account only the blocks of SNPs that are going to accessed in the nested loop */
			dim3 blocksPerGrid_pairwise_singleX_Z ( (size_t)ceil(((float)(numSNPs - start_Y)) / ((float)1)), 1, 1);	
			dim3 workgroupSize_pairwise_singleX_Z ( 1, 128, 1 );
			epistasis_pairwiseSNPs_singleX_Z<<<blocksPerGrid_pairwise_singleX_Z, workgroupSize_pairwise_singleX_Z, 0, cudaStream_singleX_Z>>>((uint*)cases_A_ptrGPU, (uint*)controls_A_ptrGPU, d_output_pairwiseSNP_singleX_Z_popcountsForCases, d_output_pairwiseSNP_singleX_Z_popcountsForControls, numSNPs, numCases, numControls, X_index, start_Y);  

			for(int start_Z = start_Y; start_Z < numSNPsWithPadding; start_Z+=BLOCK_SIZE) {
				(*roundsCounter)++;


				/* Processes Cases */

				ScalarBinary32 *A_ptrGPU_iter_cases = (ScalarBinary32 *) (d_output_X_Y_SNP_ForCases + (0 * (SNP_CALC * SNP_CALC) * (numCasesWithPadding / 32)));
				ScalarBinary32 *B_ptrGPU_iter_cases = cases_B_ptrGPU + (start_Z * SNP_CALC * (numCasesWithPadding/32));       

				result = Cutlass_U1_WmmagemmTN(
						BLOCK_SIZE * (SNP_CALC * SNP_CALC),   
						BLOCK_SIZE * SNP_CALC,        
						numCasesWithPadding,		
						A_ptrGPU_iter_cases,
						A_leadingDim_cases,
						B_ptrGPU_iter_cases,
						B_leadingDim_cases,
						C_ptrGPU_cases + (objectiveFunctionIndex % NUM_STREAMS) * ((BLOCK_SIZE * SNP_CALC * SNP_CALC) * (BLOCK_SIZE * SNP_CALC)),  
						C_leadingDim,
						cudaStreamToUse[(objectiveFunctionIndex % NUM_STREAMS)]	
						);

			        if(result != cudaSuccess) {
                			return result;
       				}


				/* Processes Controls */

				ScalarBinary32 *A_ptrGPU_iter_controls = (ScalarBinary32 *) (d_output_X_Y_SNP_ForControls + (0 * (SNP_CALC * SNP_CALC) * (numControlsWithPadding / 32)));
				ScalarBinary32 *B_ptrGPU_iter_controls = controls_B_ptrGPU +  (start_Z * SNP_CALC * (numControlsWithPadding/32)); 

				result = Cutlass_U1_WmmagemmTN(
						BLOCK_SIZE * (SNP_CALC * SNP_CALC),   
						BLOCK_SIZE * SNP_CALC,        
						numControlsWithPadding,		
						A_ptrGPU_iter_controls,
						A_leadingDim_controls,
						B_ptrGPU_iter_controls,
						B_leadingDim_controls,
						C_ptrGPU_controls + (objectiveFunctionIndex % NUM_STREAMS) * ((BLOCK_SIZE * SNP_CALC * SNP_CALC) * (BLOCK_SIZE * SNP_CALC)),  
						C_leadingDim,
						cudaStreamToUse[(objectiveFunctionIndex % NUM_STREAMS)]	
						);

				if(result != cudaSuccess) {
					return result;
				}

				if(X_index == 0) {
					cudaStreamSynchronize(cudaStreamPairwiseSNPs);
				}

				if(start_Z == start_Y)  {
					cudaStreamSynchronize(cudaStream_singleX_Z);
				}


	                        /* Derives contingency tables from the output of the binary matrix operations and calculates objective scoring function.
        	                   The reduction of scores and identification of the best candidate solution is also implemented by the same kernel.
				   Experimented with other threadblock shapes (e.g. 32x8, 32x4), but it did not affect performance in a significant manner. */
				
                                dim3 blocksPerGrid_objFun( (size_t)ceil(((float)(BLOCK_SIZE)) / ((float)32) ), (size_t)ceil(((float)(BLOCK_SIZE)) / ((float)1) / ((float)BLOCK_OBJFUN)), 1);
				dim3 workgroupSize_objFun( 32, 1, 1 );
				
				if((start_Z + BLOCK_SIZE) > numSNPs) {
                                	objectiveFunctionKernel<true><<<blocksPerGrid_objFun, workgroupSize_objFun, 0, cudaStreamToUse[(objectiveFunctionIndex % NUM_STREAMS)]>>>(C_ptrGPU_cases + (objectiveFunctionIndex % NUM_STREAMS) * (BLOCK_SIZE * SNP_CALC * SNP_CALC * BLOCK_SIZE * SNP_CALC), C_ptrGPU_controls + (objectiveFunctionIndex % NUM_STREAMS) * (BLOCK_SIZE * SNP_CALC * SNP_CALC * BLOCK_SIZE * SNP_CALC), d_output_individualSNP_popcountsForCases, d_output_individualSNP_popcountsForControls, d_output_pairwiseSNP_popcountsForCases, d_output_pairwiseSNP_popcountsForControls, d_tablePrecalc, d_output, d_output_packedIndices, start_Y, start_Z, X_index, d_output_pairwiseSNP_singleX_Z_popcountsForCases, d_output_pairwiseSNP_singleX_Z_popcountsForControls, numSNPs, numCases, numControls);
				}
				else {
					objectiveFunctionKernel<false><<<blocksPerGrid_objFun, workgroupSize_objFun, 0, cudaStreamToUse[(objectiveFunctionIndex % NUM_STREAMS)]>>>(C_ptrGPU_cases + (objectiveFunctionIndex % NUM_STREAMS) * (BLOCK_SIZE * SNP_CALC * SNP_CALC * BLOCK_SIZE * SNP_CALC), C_ptrGPU_controls + (objectiveFunctionIndex % NUM_STREAMS) * (BLOCK_SIZE * SNP_CALC * SNP_CALC * BLOCK_SIZE * SNP_CALC), d_output_individualSNP_popcountsForCases, d_output_individualSNP_popcountsForControls, d_output_pairwiseSNP_popcountsForCases, d_output_pairwiseSNP_popcountsForControls, d_tablePrecalc, d_output, d_output_packedIndices, start_Y, start_Z, X_index, d_output_pairwiseSNP_singleX_Z_popcountsForCases, d_output_pairwiseSNP_singleX_Z_popcountsForControls, numSNPs, numCases, numControls);
				}

				objectiveFunctionIndex++;
			}
		}
	}

	/* In case evaluation of blocks is still ongoing */
	for (int i = 0; i < NUM_STREAMS; i++) {
		cudaStreamSynchronize(cudaStreamToUse[i]);
	}

	/* Copies best solution found from GPU memory to Host */
	cudaMemcpy(outputFromGpu, d_output, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(output_indexFromGpu_packedIndices, d_output_packedIndices, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);


	cudaDeviceSynchronize();	
	clock_gettime(CLOCK_MONOTONIC, &t_end); // final timestamp

        (*searchTime) = ((t_end.tv_sec + ((double) t_end.tv_nsec / 1000000000)) - (t_start.tv_sec + ((double) t_start.tv_nsec / 1000000000)));


	cudaFree(cases_A_ptrGPU);
	cudaFree(C_ptrGPU_cases);
	cudaFree(controls_A_ptrGPU);
	cudaFree(C_ptrGPU_controls);
	cudaFree(d_output_individualSNP_popcountsForControls);
	cudaFree(d_output_individualSNP_popcountsForCases);
	cudaFree(d_tablePrecalc);
	cudaFree(d_output);
	cudaFree(d_output_packedIndices);

	free(h_tablePrecalc);
	
	return cudaSuccess;
}
