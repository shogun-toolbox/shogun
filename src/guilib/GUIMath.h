#ifndef __GUIMATH__H__ 
#define __GUIMATH__H__ 

class CGUIMath
{
public:
	void evaluate_results(REAL* output, int* label, int total, REAL tresh, FILE* outputfile, FILE* rocfile);
	void current_results(REAL* output, int* label, int total, REAL tresh, FILE* outputfile);
};
#endif
