/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#ifndef HAVE_SWIG
#include "lib/Mathematics.h"
#include "guilib/GUIMath.h"
#include "lib/io.h"

CGUIMath::CGUIMath(CGUI* g) : gui(g), threshold(0.0)
{
}

void CGUIMath::set_threshold(CHAR* param)
{
	param=CIO::skip_spaces(param);
	CIO::message(M_INFO, "old threshold: %f", threshold);
	sscanf(param,"%le", &threshold);
	CIO::message(M_INFO, " new threshold: %f\n", threshold);
}

void CGUIMath::evaluate_results(DREAL* output, INT* label, INT total, FILE* outputfile, FILE* rocfile)
{
	current_results(output, label, total, outputfile);

	DREAL* fp= new DREAL[total];	
	DREAL* tp= new DREAL[total];	
	INT possize=0;
	INT negsize=0;
	INT size=total;
	INT pointeven=CMath::calcroc(fp, tp, output, label, size, possize, negsize, threshold, rocfile);

	if (pointeven!=-1)
	{
		// rounding necessary due to (although very small) numerical deviations
		double correct=CMath::round(possize*tp[pointeven]+(1.0-fp[pointeven])*negsize);
		double fpo=CMath::round(fp[pointeven]*negsize);
		double fne=CMath::round((1-tp[pointeven])*possize);
		CIO::message(M_INFO, "classified:\n");
		CIO::message(M_INFO, "total: %i pos: %i, neg: %i\n", possize+negsize, possize, negsize);
		CIO::message(M_INFO, "\tcorrect:%i\n", INT (correct));
		CIO::message(M_INFO, "\twrong:%i (fp:%i,fn:%i)\n", int(fpo+fne), INT (fpo), INT (fne));
		CIO::message(M_INFO, "of %i samples (c:%f,w:%f,fp:%f,tp:%f,tresh*:%+.18g)\n",total, correct/total, 1-correct/total, (double) fp[pointeven], (double) tp[pointeven], threshold);
		CIO::message(M_INFO, "setting threshold to: %f\n", threshold);
	}

	delete[] fp;
	delete[] tp;
}

void CGUIMath::current_results(DREAL* output, INT* label, INT total, FILE* outputfile)
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
		CIO::message(M_INFO, "classified %d examples\n", total);
	}
	else
	{
		CIO::message(M_INFO, "classified:\n");
		CIO::message(M_INFO, "\tcorrect:%i\n", INT (correct));
		CIO::message(M_INFO, "\twrong:%i (fp:%i,fn:%i)\n", int(fp+fn), INT (fp), INT (fn));
		CIO::message(M_INFO, "of %i samples (c:%f,w:%f,fp:%f,tp:%f,tresh:%+.18g)\n", total, ((double) correct)/((double)total), 1.0-((double)correct)/((double)total),
				((double) fp)/((double) neg), (double) (pos-fn)/((double)pos), ((double) threshold));
	}
}
#endif
