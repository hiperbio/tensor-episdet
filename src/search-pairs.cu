
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

	cases_B_ptrGPU = cases_A_ptrGPU;  // Matrix B points to the same data as matrix A

	int *C_ptrGPU_cases;
	result = cudaMalloc((int**) &C_ptrGPU_cases, sizeof(int) * NUM_STREAMS * (BLOCK_SIZE * SNP_CALC) * (BLOCK_SIZE * SNP_CALC));          
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


	controls_B_ptrGPU = controls_A_ptrGPU;  // Matrix B points to the same data as matrix A

	int *C_ptrGPU_controls;
	result = cudaMalloc((int**) &C_ptrGPU_controls, sizeof(int) * NUM_STREAMS * (BLOCK_SIZE * SNP_CALC) * (BLOCK_SIZE * SNP_CALC));          
        if(result != cudaSuccess) {
                std::cerr << "Failed allocating memory for controls output data." << std::endl;
        }


	/* For storing the indidual SNP popcounts. */
	uint * d_output_individualSNP_popcountsForCases;
	uint * d_output_individualSNP_popcountsForControls;

	result = cudaMalloc((uint**)&d_output_individualSNP_popcountsForControls, 3 * numSNPs * sizeof(uint));
	result = cudaMalloc((uint**)&d_output_individualSNP_popcountsForCases, 3 * numSNPs * sizeof(uint));

	/* CUDA stream creation */
	cudaStream_t cudaStreamToUse[NUM_STREAMS];
	for(int i=0; i < NUM_STREAMS; i++) {
		cudaStreamCreate(&cudaStreamToUse[i]);
	}

	uint objectiveFunctionIndex = 0;


	for(int block_start=0; block_start < numSNPsWithPadding; block_start += BLOCK_SIZE ) {

		int index_cases = block_start * SNP_CALC * (numCasesWithPadding / 32);  
		int index_controls = block_start * SNP_CALC * (numControlsWithPadding / 32); 

		cudaMemcpyAsync(&cases_A_ptrGPU[index_cases], &datasetCases_host_matrixA[index_cases], sizeof(int) * BLOCK_SIZE * SNP_CALC * (numCasesWithPadding / 32), cudaMemcpyHostToDevice, cudaStreamToUse[((block_start / BLOCK_SIZE) % NUM_STREAMS)]);
		cudaMemcpyAsync(&controls_A_ptrGPU[index_controls], &datasetControls_host_matrixA[index_controls], sizeof(int) * BLOCK_SIZE * SNP_CALC * (numControlsWithPadding / 32), cudaMemcpyHostToDevice, cudaStreamToUse[(block_start / BLOCK_SIZE % NUM_STREAMS)]);

		int start_SNP_idx = block_start;
		int blocksPerGrid_ind = (size_t)ceil(((float)(BLOCK_SIZE)) / ((float)32));
		epistasis_individualSNPs<<<blocksPerGrid_ind, 32, 0, cudaStreamToUse[((block_start / BLOCK_SIZE) % NUM_STREAMS)]>>>(start_SNP_idx, (uint*)cases_A_ptrGPU, (uint*)controls_A_ptrGPU, d_output_individualSNP_popcountsForCases, d_output_individualSNP_popcountsForControls, numSNPs, numCases, numControls);  

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
	unsigned long long int * d_output_index_packedIndices;


	#if defined(MI_SCORE)
	float h_output[1] = {FLT_MIN};
	#else
	float h_output[1] = {FLT_MAX};
	#endif


	/* Stores the best score and indexes of corresponding set of SNPs */
	result = cudaMalloc((float**)&d_output, 1 * sizeof(float));								
	result = cudaMalloc((unsigned long long int**)&d_output_index_packedIndices, 1 * sizeof(unsigned long long int));	

	result = cudaMemcpy(d_output, h_output, 1 * sizeof(float), cudaMemcpyHostToDevice);


	uint A_leadingDim_cases = numCasesWithPadding / 32;   		
	uint B_leadingDim_cases = numCasesWithPadding / 32;   		

	uint A_leadingDim_controls = numControlsWithPadding / 32;   	
	uint B_leadingDim_controls = numControlsWithPadding / 32;   	

	uint C_leadingDim = BLOCK_SIZE * SNP_CALC;			


	(*roundsCounter) = 0;
	for(int start_A=0; start_A < numSNPsWithPadding; start_A+=BLOCK_SIZE)
	{
                std::cout << "Outer loop iteration " << (int) (start_A / BLOCK_SIZE) << " out of " << (int) (numSNPsWithPadding / BLOCK_SIZE) << std::endl;

		for(int start_B=start_A; start_B < numSNPsWithPadding; start_B+=BLOCK_SIZE) {
			(*roundsCounter)++;


			if((start_A == 0) && (start_B == (numSNPsWithPadding - BLOCK_SIZE))) {
				for(int i=0; i < NUM_STREAMS; i++) {
					cudaStreamSynchronize(cudaStreamToUse[i]);
				}
			}


			/* Processes Cases */

			ScalarBinary32 *A_ptrGPU_iter_cases = cases_A_ptrGPU + (start_A * SNP_CALC * (numCasesWithPadding/32));
			ScalarBinary32 *B_ptrGPU_iter_cases = cases_B_ptrGPU + (start_B * SNP_CALC * (numCasesWithPadding/32));	

			result = Cutlass_U1_WmmagemmTN(
					BLOCK_SIZE * SNP_CALC,
					BLOCK_SIZE * SNP_CALC,
					numCasesWithPadding,		
					A_ptrGPU_iter_cases,
					A_leadingDim_cases,
					B_ptrGPU_iter_cases,	
					B_leadingDim_cases,
					C_ptrGPU_cases + (objectiveFunctionIndex % NUM_STREAMS) * ((BLOCK_SIZE * SNP_CALC) * (BLOCK_SIZE * SNP_CALC)),  
					C_leadingDim,
					cudaStreamToUse[(objectiveFunctionIndex % NUM_STREAMS)]	// stream id
					);

			if(result != cudaSuccess) {
				return result;
			}


			/* Processes Controls */

			ScalarBinary32 *A_ptrGPU_iter_controls = controls_A_ptrGPU + (start_A * SNP_CALC * (numControlsWithPadding/32));
			ScalarBinary32 *B_ptrGPU_iter_controls = controls_B_ptrGPU + (start_B * SNP_CALC * (numControlsWithPadding/32));	


			result = Cutlass_U1_WmmagemmTN(
					BLOCK_SIZE * SNP_CALC,
					BLOCK_SIZE * SNP_CALC,
					numControlsWithPadding,			
					A_ptrGPU_iter_controls,
					A_leadingDim_controls,
					B_ptrGPU_iter_controls,	
					B_leadingDim_controls,
					C_ptrGPU_controls + (objectiveFunctionIndex % NUM_STREAMS) * ((BLOCK_SIZE * SNP_CALC) * (BLOCK_SIZE * SNP_CALC)),  
					C_leadingDim,
					cudaStreamToUse[(objectiveFunctionIndex % NUM_STREAMS)] // stream id
					);

                        if(result != cudaSuccess) {
                                return result; 
                        }


			/* Derives contingency tables from the output of the binary matrix operations and calculates objective scoring function.
			   The reduction of scores and identification of the best candidate solution is also implemented by the same kernel.
			   Experimented with other threadblock shapes (e.g. 32x8, 32x4), but it did not affect performance in a significant manner. */

			dim3 blocksPerGrid_objFun( (size_t)ceil(((float)(BLOCK_SIZE)) / ((float)32) ), (size_t)ceil(((float)(BLOCK_SIZE)) / ((float)1) / ((float)BLOCK_OBJFUN)), 1);
			dim3 workgroupSize_objFun( 32, 1, 1 );

                        if((start_B + BLOCK_SIZE) > numSNPs) {
				objectiveFunctionKernel<true><<<blocksPerGrid_objFun, workgroupSize_objFun, 0, cudaStreamToUse[(objectiveFunctionIndex % NUM_STREAMS)]>>>(C_ptrGPU_cases + (objectiveFunctionIndex % NUM_STREAMS) * (BLOCK_SIZE * SNP_CALC * BLOCK_SIZE * SNP_CALC), C_ptrGPU_controls + (objectiveFunctionIndex % NUM_STREAMS) * (BLOCK_SIZE * SNP_CALC * BLOCK_SIZE * SNP_CALC), d_output_individualSNP_popcountsForCases, d_output_individualSNP_popcountsForControls, d_tablePrecalc, d_output, d_output_index_packedIndices, start_A, start_B, numSNPs, numCases, numControls);
			}
			else {
                                objectiveFunctionKernel<false><<<blocksPerGrid_objFun, workgroupSize_objFun, 0, cudaStreamToUse[(objectiveFunctionIndex % NUM_STREAMS)]>>>(C_ptrGPU_cases + (objectiveFunctionIndex % NUM_STREAMS) * (BLOCK_SIZE * SNP_CALC * BLOCK_SIZE * SNP_CALC), C_ptrGPU_controls + (objectiveFunctionIndex % NUM_STREAMS) * (BLOCK_SIZE * SNP_CALC * BLOCK_SIZE * SNP_CALC), d_output_individualSNP_popcountsForCases, d_output_individualSNP_popcountsForControls, d_tablePrecalc, d_output, d_output_index_packedIndices, start_A, start_B, numSNPs, numCases, numControls);
			}
			
			objectiveFunctionIndex++;
		}
	}

	/* In case evaluation of rounds of blocks of SNPs is still ongoing. */
	for(int i=0; i < NUM_STREAMS; i++) {
		cudaStreamSynchronize(cudaStreamToUse[i]);
	}

	/* Copies solution found from GPU memory to Host memory */
	cudaMemcpy(outputFromGpu, d_output, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(output_indexFromGpu_packedIndices, d_output_index_packedIndices, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

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
	cudaFree(d_output_index_packedIndices);

	free(h_tablePrecalc);

	return cudaSuccess;
}
