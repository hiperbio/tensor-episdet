#ifndef SCORING_H_
#define SCORING_H_


#define CALC_K2(cases_val, controls_val); {\
       score_value += __ldg(&tablePrecalc[controls_val]) + __ldg(&tablePrecalc[cases_val]) - __ldg(&tablePrecalc[controls_val + cases_val + 1]); \
}

#define CALC_MI(cases_val, controls_val); {\
	probabilityCase = cases_val * numPatientsInv; \
	entropyAll -= probabilityCase * __ldg(&tablePrecalc[cases_val]); \
	probabilityControl = controls_val * numPatientsInv; \
	entropyAll -= probabilityControl * __ldg(&tablePrecalc[controls_val]); \
	probabilityCasePlusControl = probabilityCase + probabilityControl; \
	entropyX -= probabilityCasePlusControl * __ldg(&tablePrecalc[cases_val + controls_val]); \
}

__inline__ __device__ float scoring_pairs(int *calc_arr_cases, int *calc_arr_controls, float *tablePrecalc, int numCases, int numControls) {

        float score_value = 0.0f;

	#if defined(MI_SCORE)

        float probabilityCase, probabilityControl, probabilityCasePlusControl;
        float numPatientsInv = 1.0 / (numCases + numControls);    

        float entropyY = (-1.0) * (numCases * numPatientsInv) * __ldg(&tablePrecalc[numCases]);
	entropyY += (-1.0) * (numControls * numPatientsInv) * __ldg(&tablePrecalc[numControls]);
        float entropyX = 0.0;
        float entropyAll = 0.0;

        for(int i = 0; i<3; i++) {
                for(int j = 0; j<3; j++) {
                	CALC_MI(calc_arr_cases[i*3 + j], calc_arr_controls[i*3 + j]);
                }
        }
        score_value = entropyX + entropyY - entropyAll;
	
	#else	

	for(int i = 0; i<3; i++) {
		for(int j = 0; j<3; j++) {
			CALC_K2(calc_arr_cases[i*3+j], calc_arr_controls[i*3+j]);
		}
	}
	score_value = fabs(score_value);

	#endif

	return score_value;
}

__inline__ __device__ float scoring_triplets(int *calc_arr_cases, int *calc_arr_controls, float *tablePrecalc, int numCases, int numControls) {

	float score_value = 0.0f;

        #if defined(MI_SCORE)

	float probabilityCase, probabilityControl, probabilityCasePlusControl;
	float numPatientsInv = 1.0 / (numCases + numControls);   

	float entropyY = (-1.0) * (numCases * numPatientsInv) * __ldg(&tablePrecalc[numCases]);  
	entropyY += (-1.0) * (numControls * numPatientsInv) * __ldg(&tablePrecalc[numControls]);  
	float entropyX = 0.0;
	float entropyAll = 0.0;

	for(int i = 0; i<3; i++) {
		for(int j = 0; j<3; j++) {
			for(int k = 0; k<3; k++) {
				CALC_MI(calc_arr_cases[i*9 + j*3 + k], calc_arr_controls[i*9 + j*3 + k]);
                        }
                }
        }
        score_value = entropyX + entropyY - entropyAll;

	#else
	
	for(int i = 0; i<3; i++) {
		for(int j = 0; j<3; j++) {
			for(int k = 0; k<3; k++) {
				CALC_K2(calc_arr_cases[i*9 + j*3 + k], calc_arr_controls[i*9 + j*3 + k]);
			}
		}
	}
	score_value = fabs(score_value);
	
	#endif

        return score_value;
}


#endif
