#ifndef __GUIMATH__H__ 
#define __GUIMATH__H__ 

class CGUIMath
{
public:
	CGUIMath();
	void evaluate_results(REAL* output, int* label, int total, FILE* outputfile=NULL, FILE* rocfile=NULL);
	void current_results(REAL* output, int* label, int total, FILE* outputfile=NULL);

	void set_threshold(char* input);
protected:
	REAL threshold;
};
#endif
