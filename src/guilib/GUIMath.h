#ifndef __GUIMATH__H__ 
#define __GUIMATH__H__ 

class CGUI;

class CGUIMath
{
public:
	CGUIMath(CGUI *);
	void evaluate_results(DREAL* output, INT* label, INT total, FILE* outputfile=NULL, FILE* rocfile=NULL);
	void current_results(DREAL* output, INT* label, INT total, FILE* outputfile=NULL);

	void set_threshold(CHAR* input);
protected:
	CGUI* gui;
	DREAL threshold;
};
#endif
