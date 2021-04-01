/**
 *
 * tensor-episdet.cu: High-throughput 2-way or 3-way exhaustive epistasis detection using fused XOR+POPC binary operations on NVIDIA Turing tensor cores
 *
 * High-Performance Computing Architectures and Systems (HPCAS) Group, INESC-ID

 * Contact: Ricardo Nobre <ricardo.nobre@inesc-id.pt>
 *
 */

// Standard Library
#include <iostream>
#include <iomanip>      
#include <sstream>
#include <vector>
#include <cfloat>
#include <string>

#include <libgen.h>

#include "helper.hpp"
#include "reduction.hpp"

#if defined(TRIPLETS)
#include "search-triplets.hpp"
#else
#include "search-pairs.hpp"
#endif

#define MAX_CHAR_ARRAY 1000

int main(int argc, const char *arg[]) {


	/* Get CUDA device properties for device with id equal to 0. */
	cudaDeviceProp device_properties;
	cudaError_t result = cudaGetDeviceProperties(&device_properties, 0);

	if (result != cudaSuccess) {
		std::cerr << "Could not get device properties: " << cudaGetErrorString(result) << std::endl;
		return 1;
	}

	if ((device_properties.major * 10 +  device_properties.minor) < 75) {
		std::cerr << "Compute capability 7.5 (or above) is required." << std::endl;
		return 1;
	}


	if(argc < 2) {
		std::cerr << "Usage: tensor-episdet dataset.txt" << std::endl;
		return 1;
	}


	/* Reads information about input dataset. */

        FILE* fStream = fopen(arg[1], "r");     		// File with information and pointers to dataset.
	if(fStream == NULL) {
		std::cerr << "File '" << arg[1] << "' does not exist!" << std::endl;
		return 1;
	}

	char* ts = strdup(arg[1]);
	char* pathToDataset = dirname(ts);	

        char line[MAX_CHAR_ARRAY];	
        char* ret = fgets(line, MAX_CHAR_ARRAY, fStream); 	// First line represents the number of SNPs.
        uint numSNPs = atoi(line);

	char controlsFileName[MAX_CHAR_ARRAY];
        ret = fgets(controlsFileName, MAX_CHAR_ARRAY, fStream);	// Second line represents the filename with controls data.
	controlsFileName[strcspn(controlsFileName, "\n")] = 0;	// Removes trailing newline character.

        ret = fgets(line, MAX_CHAR_ARRAY, fStream); 		// Third line represents the number of controls.
        uint numControls = atoi(line);

	char casesFileName[MAX_CHAR_ARRAY];
        ret = fgets(casesFileName, MAX_CHAR_ARRAY, fStream);	// Forth line represents the filename with cases data.
	casesFileName[strcspn(casesFileName, "\n")] = 0;	// Removes trailing newline character.

        ret = fgets(line, MAX_CHAR_ARRAY, fStream); 		// Fifth line represents the number of cases.
        uint numCases = atoi(line);


	/* Calculates number of distinct blocks and padds number of SNPs to process to the block size. */
	uint numBlocks = ceil((float)numSNPs / (float)BLOCK_SIZE);
	uint numSNPsWithPadding = numBlocks * BLOCK_SIZE;

        // std::cout << "Num. SNPs With Padding: " << numSNPsWithPadding << std::endl;


	/* Padds the number of controls and of cases. */
	uint numCasesWithPadding = ceil((float)numCases / PADDING_SAMPLES) * PADDING_SAMPLES;	
	uint numControlsWithPadding = ceil((float)numControls / PADDING_SAMPLES) * PADDING_SAMPLES;


	/* Prints information about dataset and number of distinct blocks of SNPs to process. */
        std::cout << "Num. SNPs: " << numSNPs << std::endl;
        std::cout << "Num. Blocks of SNPs: " << numBlocks << std::endl;
        std::cout << "Num. Cases: " << numCases << std::endl;
        std::cout << "Num. Controls: " << numControls << std::endl;


	/* Allocates pinned memory for holding controls and cases dataset matrices.
	   Each 32-bit 'unsigned int' holds 32 binary values representing genotype information.
 	   Only two allele types are represented (SNP_CALC macro equals 2), ...
	   ... being information about the third allele type infered.
	 */
	
	int numSamplesCases_32packed = ceil(((float) numCasesWithPadding) / 32.0f);
	int numSamplesControls_32packed = ceil(((float) numControlsWithPadding) / 32.0f);

        int datasetCases_32packed_size = numSamplesCases_32packed * numSNPsWithPadding * SNP_CALC;
        unsigned int* datasetCases_32packed_matrixA = NULL;
        result = cudaHostAlloc((void**)&datasetCases_32packed_matrixA, datasetCases_32packed_size * sizeof(unsigned int), cudaHostAllocDefault );     
        if(datasetCases_32packed_matrixA == NULL) {
                std::cerr << "Problem allocating Host memory for cases" << std::endl;
        }

        int datasetControls_32packed_size = numSamplesControls_32packed * numSNPsWithPadding * SNP_CALC;    
        unsigned int* datasetControls_32packed_matrixA = NULL;
        result = cudaHostAlloc((void**)&datasetControls_32packed_matrixA, datasetControls_32packed_size * sizeof(unsigned int), cudaHostAllocDefault );
        if(datasetControls_32packed_matrixA == NULL) {
                std::cerr << "Problem allocating Host memory for controls" << std::endl;
        }


	/* Reads dataset (controls and cases data) from storage device.
	   Input dataset must be padded with zeros in the dimension of samples (cases / controls), ...
	   ... making the number of bits per {SNP, allele} tuple a multiple of PADDING_SAMPLES. */

	size_t numElem;
	std::string absolutePathToCasesFile = std::string(pathToDataset) + "/" + casesFileName;
	FILE *ifp_cases = fopen(absolutePathToCasesFile.c_str(), "rb");
	numElem = fread(datasetCases_32packed_matrixA, sizeof(unsigned int), numSamplesCases_32packed * numSNPs * SNP_CALC, ifp_cases);
	if(numElem != datasetCases_32packed_size) {
		std::cerr << "Problem loading cases from storage device" << std::endl;
	}
	fclose(ifp_cases);

	std::string absolutePathToControlsFile = std::string(pathToDataset) + "/" + controlsFileName;
	FILE *ifp_controls = fopen(absolutePathToControlsFile.c_str(), "rb");
	numElem = fread(datasetControls_32packed_matrixA, sizeof(unsigned int), numSamplesControls_32packed * numSNPs * SNP_CALC, ifp_controls);
        if(numElem != datasetControls_32packed_size) {
                std::cerr << "Problem loading controls from storage device" << std::endl;
        }
	fclose(ifp_controls);

	std::cout << "-------------------------------" << std::endl;

	/* Launches epistasis detection search. */

        int roundsCounter;
        double searchTime;
	float outputFromGpu;
       	unsigned long long int output_indexFromGpu_packedIndices;

	result = EpistasisDetectionSearch(
			datasetCases_32packed_matrixA,		// Cases matrix.
			datasetControls_32packed_matrixA,	// Controls matrix.
                        numSNPs,                                // Number of SNPs.
                        numCases,                               // Number of cases.
                        numControls,                            // Number of controls.
                        numSNPsWithPadding,                     // Number of SNPs padded to block size.
			numCasesWithPadding,     		// Number of cases padded to PADDING_SIZE.
			numControlsWithPadding,     		// Number of controls padded to PADDING_SIZE.
			&roundsCounter,				// Counter for number of rounds processed.
			&searchTime,				// Counter for execution time (seconds).
			&outputFromGpu,				// Score of best score found.
			&output_indexFromGpu_packedIndices	// Indexes of SNPs of set that results in best score.
			);

	if(result != cudaSuccess) {
		std::cerr << "Epistasis detection search failed." << std::endl;
	}

	
	/* Prints set of SNPs that results in best score. */

	#if defined(TRIPLETS)
	std::cout << "-------------------------------" << std::endl << "{SNP_X_i, SNP_Y_i, SNP_Z_i}: SCORE\t->\t{" << ((output_indexFromGpu_packedIndices >> 0) & 0x1FFFFF) << ", " << ((output_indexFromGpu_packedIndices >> 21) & 0x1FFFFF) << ", " << ((output_indexFromGpu_packedIndices >> 42) & 0x1FFFFF) << "}: " << std::fixed << std::setprecision(6) << outputFromGpu << std::endl;
	#else
	std::cout << "-------------------------------" << std::endl << "{SNP_X_i, SNP_Y_i}: SCORE\t->\t{" << ((output_indexFromGpu_packedIndices >> 0) & 0xFFFFFFFF) << ", " << ((output_indexFromGpu_packedIndices >> 32) & 0xFFFFFFFF) << "}: " << std::fixed << std::setprecision(6) << outputFromGpu << std::endl;
	#endif


	/* Prints information about the search (Tensor TOPS, search execution time, ratio of unique sets). */

	unsigned long long numCombinations = n_choose_k(numSNPs, INTER_OR);

        std::cout << "Num. of rounds processed: " << roundsCounter << std::endl;

        std::cout << "Wall-clock time:\t" << std::fixed << std::setprecision(3) << searchTime << " seconds" << std::endl;    // prints time taken to execute whole search

        std::cout << "Tensor TOPS: " << std::fixed << std::setprecision(3) << ((((double)BLOCK_SIZE * (double)BLOCK_SIZE * (double) (numCasesWithPadding + numControlsWithPadding)  * 2 * SNP_COMB_CALC * (double)roundsCounter) / (double)(searchTime)) / 1e12) << std::endl;

        std::cout << "Num. unique sets per sec. (scaled to sample size): " << std::fixed << std::setprecision(3) << (((double) numCombinations * (double) (numCases + numControls) / (double)(searchTime)) / 1e12) << " × 10^12" << std::endl;

        std::cout << "Unique sets of SNPs evaluated (k=" << INTER_OR << "): " << numCombinations << std::endl;
        std::cout << "Ratio of unique sets of SNPs: " << (((double)numCombinations) / (((double)roundsCounter) * (BLOCK_SIZE * BLOCK_SIZE))) << std::endl;

	
	return result == cudaSuccess ? 0 : 1;	

}





