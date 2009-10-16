/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "GUIMath.h"
#include "SGInterface.h"

#include <shogun/lib/config.h>
#include <shogun/lib/io.h>
#include <shogun/lib/Mathematics.h>

using namespace shogun;

CGUIMath::CGUIMath(CSGInterface* ui_)
: CSGObject(), ui(ui_), threshold(0.0)
{
}

void CGUIMath::set_threshold(float64_t value)
{
	SG_INFO("Old threshold: %f.\n", threshold);
	threshold=value;
	SG_INFO("New threshold: %f.\n", threshold);
}

void CGUIMath::init_random(uint32_t initseed)
{
	CMath::init_random(initseed);
}

void CGUIMath::evaluate_results(
	float64_t* output, int32_t* label, int32_t total, FILE* outputfile,
	FILE* rocfile)
{
	current_results(output, label, total, outputfile);

	float64_t* fp= new float64_t[total];
	float64_t* tp= new float64_t[total];
	int32_t possize=0;
	int32_t negsize=0;
	int32_t size=total;
	int32_t pointeven=CMath::calcroc(fp, tp, output, label, size, possize, negsize, threshold, rocfile);

	if (pointeven!=-1)
	{
		// rounding necessary due to (although very small) numerical deviations
		float64_t correct=CMath::round(
			possize*tp[pointeven]+(1.0-fp[pointeven])*negsize);
		float64_t fpo=CMath::round(fp[pointeven]*negsize);
		float64_t fne=CMath::round((1-tp[pointeven])*possize);
		SG_INFO( "classified:\n");
		SG_INFO( "total: %i pos: %i, neg: %i\n", possize+negsize, possize, negsize);
		SG_INFO( "\tcorrect:%i\n", int32_t (correct));
		SG_INFO( "\twrong:%i (fp:%i,fn:%i)\n", int(fpo+fne), int32_t (fpo), int32_t (fne));
		SG_INFO( "of %i samples (c:%f,w:%f,fp:%f,tp:%f,tresh*:%+.18g)\n",total, correct/total, 1-correct/total, (double) fp[pointeven], (double) tp[pointeven], threshold);
		SG_INFO( "setting threshold to: %f\n", threshold);
	}

	delete[] fp;
	delete[] tp;
}

void CGUIMath::current_results(
	float64_t* output, int32_t* label, int32_t total, FILE* outputfile)
{
	int32_t fp=0;
	int32_t fn=0;
	int32_t correct=0;
	int32_t pos=0;
	int32_t neg=0;
	int32_t unlabeled=0;

	for (int32_t dim=0; dim<total; dim++)
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
		SG_INFO( "classified %d examples\n", total);
	}
	else
	{
		SG_INFO( "classified:\n");
		SG_INFO( "\tcorrect:%i\n", int32_t (correct));
		SG_INFO( "\twrong:%i (fp:%i,fn:%i)\n", int(fp+fn), int32_t (fp), int32_t (fn));
		SG_INFO( "of %i samples (c:%f,w:%f,fp:%f,tp:%f,tresh:%+.18g)\n", total, ((double) correct)/((double)total), 1.0-((double)correct)/((double)total),
				((double) fp)/((double) neg), (double) (pos-fn)/((double)pos), ((double) threshold));
	}
}
