#ifndef SEARCH_TRIPLETS_H_   
#define SEARCH_TRIPLETS_H_

cudaError_t EpistasisDetectionSearch(unsigned int* datasetCases_host_matrixA, unsigned int* datasetControls_host_matrixA, int numSNPs, int numCases, int numControls, uint numSNPsWithPadding, int numCasesWithPadding, int numControlsWithPadding, int * roundsCounter, double * searchTime, float * outputFromGpu, unsigned long long int * output_indexFromGpu_packedIndices);

#endif
