#include "lib/Mathmatics.h"
#include "guilib/GUIMath.h"
#include "lib/io.h"

#include <math.h>
CGUIMath::CGUIMath(CGUI* g) : gui(g), threshold(0.0)
{
}

void CGUIMath::set_threshold(CHAR* param)
{
	param=CIO::skip_spaces(param);
	CIO::message("old threshold: %f", threshold);
	sscanf(param,"%le", &threshold);
	CIO::message(" new threshold: %f\n", threshold);
}

void CGUIMath::evaluate_results(REAL* output, INT* label, INT total, FILE* outputfile, FILE* rocfile)
{
	current_results(output, label, total, outputfile);

	REAL* fp= new REAL[total];	
	REAL* tp= new REAL[total];	
	INT possize=0;
	INT negsize=0;
	INT size=total;
	INT pointeven=math.calcroc(fp, tp, output, label, size, possize, negsize, threshold, rocfile);

	if (pointeven!=-1)
	{
		// rounding necessary due to (although very small) numerical deviations
		double correct=math.round(possize*tp[pointeven]+(1.0-fp[pointeven])*negsize);
		double fpo=math.round(fp[pointeven]*negsize);
		double fne=math.round((1-tp[pointeven])*possize);
		CIO::message("classified:\n");
		CIO::message("total: %i pos: %i, neg: %i\n", possize+negsize, possize, negsize);
		CIO::message("\tcorrect:%i\n", INT (correct));
		CIO::message("\twrong:%i (fp:%i,fn:%i)\n", int(fpo+fne), INT (fpo), INT (fne));
		CIO::message("of %i samples (c:%f,w:%f,fp:%f,tp:%f,tresh*:%+.18g)\n",total, correct/total, 1-correct/total, (double) fp[pointeven], (double) tp[pointeven], threshold);
		CIO::message("setting threshold to: %f\n", threshold);
	}

	delete[] fp;
	delete[] tp;
}

void CGUIMath::current_results(REAL* output, INT* label, INT total, FILE* outputfile)
{
	INT fp=0;
	INT fn=0;
	INT correct=0;
	INT pos=0;
	INT neg=0;
	INT unlabeled=0;

	for (INT dim=0; dim<total; dim++)
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
		CIO::message("\tcorrect:%i\n", INT (correct));
		CIO::message("\twrong:%i (fp:%i,fn:%i)\n", int(fp+fn), INT (fp), INT (fn));
		CIO::message("of %i samples (c:%f,w:%f,fp:%f,tp:%f,tresh:%+.18g)\n", total, ((double) correct)/((double)total), 1.0-((double)correct)/((double)total),
				((double) fp)/((double) neg), (double) (pos-fn)/((double)pos), ((double) threshold));
	}
}
