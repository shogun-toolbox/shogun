#include "lib/Mathmatics.h"
#include "guilib/GUIMath.h"
#include "lib/io.h"

void CGUIMath::evaluate_results(REAL* output, int* label, int total, REAL tresh, FILE* outputfile, FILE* rocfile)
{
	current_results(output, label, total, tresh, outputfile);

	REAL* fp= new REAL[total];	
	REAL* tp= new REAL[total];	
	int possize=0;
	int negsize=0;
	int pointeven=math.calcroc(fp, tp, output, label, total, possize, negsize, tresh, rocfile);

	double correct=possize*tp[pointeven]+(1-fp[pointeven])*negsize;
	double fpo=fp[pointeven]*negsize;
	double fne=(1-tp[pointeven])*possize;

	CIO::message("classified:\n");
	CIO::message("\tcorrect:%i\n", int (correct));
	CIO::message("\twrong:%i (fp:%i,fn:%i)\n", int(fpo+fne), int (fpo), int (fne));
	CIO::message("of %i samples (c:%f,w:%f,fp:%f,tp:%f,tresh*:%f)\n",total, correct/total, 1-correct/total, (double) fp[pointeven], (double) tp[pointeven], tresh);
	delete[] fp;
	delete[] tp;
}

void CGUIMath::current_results(REAL* output, int* label, int total, REAL tresh, FILE* outputfile)
{
	int fp=0;
	int fn=0;
	int correct=0;
	int pos=0;
	int neg=0;

	for (int dim=0; dim<total; dim++)
	{
		if (label[dim] > 0)
			pos++;
		else
			neg++;

		if (math.sign((REAL) output[dim])==label[dim])
		{
			fprintf(outputfile,"%+.8g (%+d)\n",(double) output[dim], label[dim]);
			correct++;
		}
		else
		{
			fprintf(outputfile,"%+.8g (%+d)(*)\n",(double) output[dim], label[dim]);
			if (label[dim]>0)
				fp++;
			else
				fn++;
		}
	}
	CIO::message("classified:\n");
	CIO::message("\tcorrect:%i\n", int (correct));
	CIO::message("\twrong:%i (fp:%i,fn:%i)\n", int(fp+fn), int (fp), int (fn));
	CIO::message("of %i samples (c:%f,w:%f,fp:%f,tp:%f,tresh:%f)\n",total, correct/total, 1-correct/total, (double) fp/neg, (double) (pos-fn)/pos, tresh);

}
