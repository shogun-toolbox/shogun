#include "lib/Mathmatics.h"
#include "guilib/GUIMath.h"
#include "lib/io.h"

#include <math.h>
CGUIMath::CGUIMath() : threshold(0.0)
{
}

void CGUIMath::set_threshold(char* param)
{
	param=CIO::skip_spaces(param);
	CIO::message("old threshold: %f", threshold);
	sscanf(param,"%le", &threshold);
	CIO::message(" new threshold: %f\n", threshold);
}

void CGUIMath::evaluate_results(REAL* output, int* label, int total, FILE* outputfile, FILE* rocfile)
{
	current_results(output, label, total, outputfile);

	REAL* fp= new REAL[total];	
	REAL* tp= new REAL[total];	
	int possize=0;
	int negsize=0;
	int size=total;
	int pointeven=math.calcroc(fp, tp, output, label, size, possize, negsize, threshold, rocfile);

	if (pointeven!=-1)
	{
		// rounding necessary due to (although very small) numerical deviations
		double correct=math.round(possize*tp[pointeven]+(1.0-fp[pointeven])*negsize);
		double fpo=math.round(fp[pointeven]*negsize);
		double fne=math.round((1-tp[pointeven])*possize);
		CIO::message("classified:\n");
		CIO::message("total: %i pos: %i, neg: %i\n", possize+negsize, possize, negsize);
		CIO::message("\tcorrect:%i\n", int (correct));
		CIO::message("\twrong:%i (fp:%i,fn:%i)\n", int(fpo+fne), int (fpo), int (fne));
		CIO::message("of %i samples (c:%f,w:%f,fp:%f,tp:%f,tresh*:%+.18g)\n",total, correct/total, 1-correct/total, (double) fp[pointeven], (double) tp[pointeven], threshold);
		delete[] fp;
		delete[] tp;
		CIO::message("setting threshold to: %f\n", threshold);
	}
}

void CGUIMath::current_results(REAL* output, int* label, int total, FILE* outputfile)
{
	int fp=0;
	int fn=0;
	int correct=0;
	int pos=0;
	int neg=0;
	int unlabeled=0;

	for (int dim=0; dim<total; dim++)
	{
		if (label[dim] > 0)
			pos++;
		else if (label[dim]<0)
			neg++;
		else
			unlabeled++;

		if (label[dim]==0)
		{
			fprintf(outputfile,"%+.18g\n",(double) output[dim]-threshold);
		}
		else if ( (output[dim]-threshold>=0 && label[dim]>0) || (output[dim]-threshold<0 && label[dim]<0) )
		{
			fprintf(outputfile,"%+.18g (%+d)\n",(double) output[dim]-threshold, label[dim]);
			correct++;
		}
		else
		{
			fprintf(outputfile,"%+.18g (%+d)(*)\n",(double) output[dim]-threshold, label[dim]);
			if (label[dim]>0)
				fn++;
			else
				fp++;
		}
	}

	if (unlabeled==total || neg==0 || pos==0)
	{
		CIO::message("classified %d examples\n", total);
	}
	else
	{
		CIO::message("classified:\n");
		CIO::message("\tcorrect:%i\n", int (correct));
		CIO::message("\twrong:%i (fp:%i,fn:%i)\n", int(fp+fn), int (fp), int (fn));
		CIO::message("of %i samples (c:%f,w:%f,fp:%f,tp:%f,tresh:%+.18g)\n", total, ((double) correct)/((double)total), 1.0-((double)correct)/((double)total),
				((double) fp)/((double) neg), (double) (pos-fn)/((double)pos), ((double) threshold));
	}
}
