// -*- C++ -*-
// Main functions of the LaRank algorithm for soving Multiclass SVM
// Copyright (C) 2008- Antoine Bordes
// Shogun specific adjustments (w) 2009 Soeren Sonnenburg

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA

#include <iostream>
#include <vector>
#include <algorithm>

#include <cstdlib>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cfloat>
#include <cassert>

#include "lib/io.h"
#include "lib/Signal.h"
#include "lib/Mathematics.h"
#include "classifier/svm/LaRank.h"
#include "kernel/Kernel.h"

using namespace shogun;
// Initializing an output class (basically creating a kernel cache for it)
void LaRankOutput::initialize (CKernel* kfunc, long cache)
{
	kernel = larank_kcache_create (kfunc);
	larank_kcache_set_maximum_size (kernel, cache * 1024 * 1024);
	beta = new float[1];
	g = new float[1];
	*beta=0;
	*g=0;
	l = 0;
}

// Destroying an output class (basically destroying the kernel cache)
void LaRankOutput::destroy ()
{
	larank_kcache_destroy (kernel);
	kernel=NULL;
	delete[] beta;
	delete[] g;
	beta=NULL;
	g=NULL;
}

// !Important! Computing the score of a given input vector for the actual output
double LaRankOutput::computeScore (int x_id)
{
	if (l == 0)
		return 0;
	else
	{
		float *row = larank_kcache_query_row (kernel, x_id, l);
		return CMath::dot (beta, row, l);
	}
}

// !Important! Computing the gradient of a given input vector for the actual output           
double LaRankOutput::computeGradient (int xi_id, int yi, int ythis)
{
	return (yi == ythis ? 1 : 0) - computeScore (xi_id);
}

// Updating the solution in the actual output
void LaRankOutput::update (int x_id, double lambda, double gp)
{
    int *r2i = larank_kcache_r2i (kernel, l);
    int xr = l + 1;
    for (int r = 0; r < l; r++)
      if (r2i[r] == x_id)
	{
	  xr = r;
	  break;
	}

	// updates the cache order and the beta coefficient
	if (xr < l)
	{
		beta[xr]+=lambda;
	}
	else
	{
		larank_kcache_swap_ri (kernel, l, x_id);
		CMath::resize(g, l, l+1);
		CMath::resize(beta, l, l+1);
		g[l]=gp;
		beta[l]=lambda;
		l++;
	}

	// update stored gradients
	float *row = larank_kcache_query_row (kernel, x_id, l);
	for (int r = 0; r < l; r++)
	{
		double oldg = g[r];
		g[r]=oldg - lambda * row[r];
	}
}

// Linking the cahe of this output to the cache of an other "buddy" output
// so that if a requested value is not found in this cache, you can ask your buddy if it has it.                              
void LaRankOutput::set_kernel_buddy (larank_kcache_t * bud)
{
	larank_kcache_set_buddy (bud, kernel);
}

// Removing useless support vectors (for which beta=0)                
int LaRankOutput::cleanup ()
{
	int count = 0;
	std::vector < int >idx;
	for (int x = 0; x < l; x++)
	{
		if ((beta[x] < FLT_EPSILON) && (beta[x] > -FLT_EPSILON))
		{
			idx.push_back (x);
			count++;
		}
	}
	int new_l = l - count;
	for (int xx = 0; xx < count; xx++)
	{
		int i = idx[xx] - xx;
		for (int r = i; r < (l - 1); r++)
		{
			larank_kcache_swap_rr (kernel, r, r + 1);
			beta[r]=beta[r + 1];
			g[r]=g[r + 1];
		}
	}
	CMath::resize(beta, l, new_l+1);
	CMath::resize(g, l, new_l+1);
	beta[new_l]=0;
	g[new_l]=0;
	l = new_l;
	return count;
}

// --- Below are information or "get" functions --- //
//
double LaRankOutput::getW2 ()
{
	double sum = 0;
	int *r2i = larank_kcache_r2i (kernel, l + 1);
	for (int r = 0; r < l; r++)
	{
		float *row_r = larank_kcache_query_row (kernel, r2i[r], l);
		sum += beta[r] * CMath::dot (beta, row_r, l);
	}
	return sum;
}

//
double LaRankOutput::getBeta (int x_id)
{
	int *r2i = larank_kcache_r2i (kernel, l);
	int xr = -1;
	for (int r = 0; r < l; r++)
		if (r2i[r] == x_id)
		{
			xr = r;
			break;
		}
	return (xr < 0 ? 0 : beta[xr]);
}

//
double LaRankOutput::getGradient (int x_id)
{
	int *r2i = larank_kcache_r2i (kernel, l);
	int xr = -1;
	for (int r = 0; r < l; r++)
		if (r2i[r] == x_id)
		{
			xr = r;
			break;
		}
	return (xr < 0 ? 0 : g[xr]);
}

//
int LaRankOutput::getSV (float* &sv) const
{
	sv=new float[l];
	int *r2i = larank_kcache_r2i (kernel, l);
	for (int r = 0; r < l; r++)
		sv[r]=r2i[r];
	return l;
}

CLaRank::CLaRank (): CMultiClassSVM(ONE_VS_REST), 
	nb_seen_examples (0), nb_removed (0),
	n_pro (0), n_rep (0), n_opt (0),
	w_pro (1), w_rep (1), w_opt (1), y0 (0), dual (0),
	batch_mode(true), step(0)
{
}

CLaRank::CLaRank (float64_t C, CKernel* k, CLabels* lab):
	CMultiClassSVM(ONE_VS_REST, C, k, lab), 
	nb_seen_examples (0), nb_removed (0),
	n_pro (0), n_rep (0), n_opt (0),
	w_pro (1), w_rep (1), w_opt (1), y0 (0), dual (0),
	batch_mode(true), step(0)
{
}

CLaRank::~CLaRank ()
{
	destroy();
}

bool CLaRank::train(CFeatures* data)
{
	tau = 0.0001;

	ASSERT(kernel);
	ASSERT(labels && labels->get_num_labels());

	CSignal::clear_cancel();

	if (data)
	{
		if (data->get_num_vectors() != labels->get_num_labels())
		{
			SG_ERROR("Numbert of vectors (%d) does not match number of labels (%d)\n",
					data->get_num_vectors(), labels->get_num_labels());
		}
		kernel->init(data, data);
	}

	ASSERT(kernel->get_num_vec_lhs() && kernel->get_num_vec_rhs());

	nb_train=labels->get_num_labels();
	cache = kernel->get_cache_size();

	int n_it = 1;
	//double initime = getTime ();
	double gap = DBL_MAX;

	SG_INFO("Training on %d examples\n", nb_train);
	while (gap > C1 && (!CSignal::cancel_computations()))      // stopping criteria
	{
		double tr_err = 0;
		int ind = step;
		for (int i = 0; i < nb_train; i++)
		{
			int y=labels->get_label(i);
			if (add (i, y) != y)   // call the add function
				tr_err++;

			if (ind && i / ind)
			{
				SG_DEBUG("Done: %d %% Train error (online): %f%%\n",
						(int) (((double) i) / nb_train * 100), (tr_err / ((double) i + 1)) * 100);
				//printStuff (initime, false);
				ind += step;
			}
		}

		SG_DEBUG("End of iteration %d\n", n_it++);
		SG_DEBUG("Train error (online): %f%%\n", (tr_err / nb_train) * 100);
		gap = computeGap ();
		//printStuff (initime, true);
		SG_ABS_PROGRESS(gap, -CMath::log10(gap), -CMath::log10(DBL_MAX), -CMath::log10(C1), 6);

		if (!batch_mode)        // skip stopping criteria if online mode
			gap = 0;
	}
	SG_DONE();

	int32_t num_classes = outputs.size();
	create_multiclass_svm(num_classes);
	SG_DEBUG("%d classes\n", num_classes);

	// Used for saving a model file
	int32_t i=0;
	for (outputhash_t::const_iterator it = outputs.begin (); it != outputs.end (); ++it)
	{
		const LaRankOutput* o=&(it->second);

		larank_kcache_t* k=o->getKernel();
		int l=o->get_l();
		float* beta=o->getBetas();
		int *r2i = larank_kcache_r2i (k, l);

		ASSERT(l>0);
		SG_DEBUG("svm[%d] has %d sv, b=%f\n", i, l, 0.0);

		CSVM* svm=new CSVM(l);

		for (int32_t j=0; j<l; j++)
		{
			svm->set_alpha(j, beta[j]);
			svm->set_support_vector(j, r2i[j]);
		}

		svm->set_bias(0);
		set_svm(i, svm);
		i++;
	}
	destroy();

	return true;
}





// LEARNING FUNCTION: add new patterns and run optimization steps selected with adaptative schedule
int CLaRank::add (int x_id, int yi)
{
	++nb_seen_examples;
	// create a new output object if this one has never been seen before 
	if (!getOutput (yi))
	{
		outputs.insert (std::make_pair (yi, LaRankOutput ()));
		LaRankOutput *cur = getOutput (yi);
		cur->initialize (kernel, cache);
		if (outputs.size () == 1)
			y0 = outputs.begin ()->first;
		// link the cache of this new output to a buddy 
		if (outputs.size () > 1)
		{
			LaRankOutput *out0 = getOutput (y0);
			cur->set_kernel_buddy (out0->getKernel ());
		}
	}

	LaRankPattern tpattern (x_id, yi);
	LaRankPattern & pattern = (patterns.isPattern (x_id)) ? patterns.getPattern (x_id) : tpattern;

	// ProcessNew with the "fresh" pattern
	double time1 = getTime ();
	process_return_t pro_ret = process (pattern, processNew);
	double dual_increase = pro_ret.dual_increase;
	double duration = (getTime () - time1);
	double coeff = dual_increase / (0.00001 + duration);
	dual += dual_increase;
	n_pro++;
	w_pro = 0.05 * coeff + (1 - 0.05) * w_pro;

	// ProcessOld & Optimize until ready for a new processnew
	// (Adaptative schedule here)
	for (;;)
	{
		double w_sum = w_pro + w_rep + w_opt;
		double prop_min = w_sum / 20;
		if (w_pro < prop_min)
			w_pro = prop_min;
		if (w_rep < prop_min)
			w_rep = prop_min;
		if (w_opt < prop_min)
			w_opt = prop_min;
		w_sum = w_pro + w_rep + w_opt;
		double r = rand () / (double) RAND_MAX * w_sum;
		if (r <= w_pro)
		{
			break;
		}
		else if ((r > w_pro) && (r <= w_pro + w_rep))	// ProcessOld here
		{
			double ltime1 = getTime ();
			double ldual_increase = reprocess ();
			double lduration = (getTime () - ltime1);
			double lcoeff = ldual_increase / (0.00001 + lduration);
			dual += ldual_increase;
			n_rep++;
			w_rep = 0.05 * lcoeff + (1 - 0.05) * w_rep;
		}
		else			// Optimize here 
		{
			double ltime1 = getTime ();
			double ldual_increase = optimize ();
			double lduration = (getTime () - ltime1);
			double lcoeff = ldual_increase / (0.00001 + lduration);
			dual += ldual_increase;
			n_opt++;
			w_opt = 0.05 * lcoeff + (1 - 0.05) * w_opt;
		}
	}
	if (nb_seen_examples % 100 == 0)	// Cleanup useless Support Vectors/Patterns sometimes
		nb_removed += cleanup ();
	return pro_ret.ypred;
}

// PREDICTION FUNCTION: main function in la_rank_classify
int CLaRank::predict (int x_id)
{
	int res = -1;
	double score_max = -DBL_MAX;
	for (outputhash_t::iterator it = outputs.begin (); it != outputs.end ();++it)
	{
		double score = it->second.computeScore (x_id);
		if (score > score_max)
		{
			score_max = score;
			res = it->first;
		}
	}
	return res;
}

void CLaRank::destroy ()
{
	for (outputhash_t::iterator it = outputs.begin (); it != outputs.end ();++it)
		it->second.destroy ();
}


// Compute Duality gap (costly but used in stopping criteria in batch mode)                     
double CLaRank::computeGap ()
{
	double sum_sl = 0;
	double sum_bi = 0;
	for (unsigned i = 0; i < patterns.maxcount (); ++i)
	{
		const LaRankPattern & p = patterns[i];
		if (!p.exists ())
			continue;
		LaRankOutput *out = getOutput (p.y);
		if (!out)
			continue;
		sum_bi += out->getBeta (p.x_id);
		double gi = out->computeGradient (p.x_id, p.y, p.y);
		double gmin = DBL_MAX;
		for (outputhash_t::iterator it = outputs.begin (); it != outputs.end (); ++it)
		{
			if (it->first != p.y && it->second.isSupportVector (p.x_id))
			{
				double g =
					it->second.computeGradient (p.x_id, p.y, it->first);
				if (g < gmin)
					gmin = g;
			}
		}
		sum_sl += CMath::max (0.0, gi - gmin);
	}
	return CMath::max (0.0, computeW2 () + C1 * sum_sl - sum_bi);
}

// Display stuffs along learning
void CLaRank::printStuff (double initime, bool print_dual)
{
	std::cout << "Current duration (CPUs): " << getTime () - initime << std::endl;
	if (print_dual)
		std::cout << "Dual: " << dual << std::endl;
	std::cout << "Number of Support Patterns: " << patterns.size () << " / " << nb_seen_examples << " (removed:" << nb_removed <<")" << std::endl;
	std::cout << "Number of Support Vectors: " << getNSV () << " (~ " << getNSV () / (double) patterns.size () << " SV/Pattern)" << std::endl;
	double w_sum = w_pro + w_rep + w_opt;
	std::cout << "ProcessNew:" << n_pro << " (" << w_pro / w_sum << ") ProcessOld:" 
		<< n_rep << " (" << w_rep / w_sum << ") Optimize:" << n_opt << " (" << w_opt / w_sum << ")" << std::endl;
	std::cout << "----" << std::endl;

}


// Nuber of classes so far
unsigned CLaRank::getNumOutputs () const
{
	return outputs.size ();
}

// Number of Support Vectors
int CLaRank::getNSV ()
{
	int res = 0;
	for (outputhash_t::const_iterator it = outputs.begin (); it != outputs.end (); ++it)
	{
		float* sv=NULL;
		res += it->second.getSV (sv);
		delete[] sv;
	}
	return res;
}

// Norm of the parameters vector
double CLaRank::computeW2 ()
{
	double res = 0;
	for (unsigned i = 0; i < patterns.maxcount (); ++i)
	{
		const LaRankPattern & p = patterns[i];
		if (!p.exists ())
			continue;
		for (outputhash_t::iterator it = outputs.begin (); it != outputs.end (); ++it)
			if (it->second.getBeta (p.x_id))
				res += it->second.getBeta (p.x_id) * it->second.computeScore (p.x_id);
	}
	return res;
}

// Compute Dual objective value
double CLaRank::getDual ()
{
	double res = 0;
	for (unsigned i = 0; i < patterns.maxcount (); ++i)
	{
		const LaRankPattern & p = patterns[i];
		if (!p.exists ())
			continue;
		LaRankOutput *out = getOutput (p.y);
		if (!out)
			continue;
		res += out->getBeta (p.x_id);
	}
	return res - computeW2 () / 2;
}

LaRankOutput *CLaRank::getOutput (int index)
{
	outputhash_t::iterator it = outputs.find (index);
	return it == outputs.end ()? NULL : &it->second;
}

// IMPORTANT Main SMO optimization step
CLaRank::process_return_t CLaRank::process (const LaRankPattern & pattern, process_type ptype)
{
	process_return_t pro_ret = process_return_t (0, 0);

	/*
	 ** compute gradient and sort   
	 */
	std::vector < outputgradient_t > outputgradients;
	
	outputgradients.reserve (getNumOutputs ());

	std::vector < outputgradient_t > outputscores;
	outputscores.reserve (getNumOutputs ());

	for (outputhash_t::iterator it = outputs.begin (); it != outputs.end (); ++it)
		if (ptype != processOptimize
				|| it->second.isSupportVector (pattern.x_id))
		{
			double g =
				it->second.computeGradient (pattern.x_id, pattern.y, it->first);
			outputgradients.push_back (outputgradient_t (it->first, g));
			if (it->first == pattern.y)
				outputscores.push_back (outputgradient_t (it->first, (1 - g)));
			else
				outputscores.push_back (outputgradient_t (it->first, -g));
		}

	std::sort (outputgradients.begin (), outputgradients.end ());

	/*
	 ** determine the prediction
	 */
	std::sort (outputscores.begin (), outputscores.end ());
	pro_ret.ypred = outputscores[0].output;

	/*
	 ** Find yp (1st part of the pair)
	 */
	outputgradient_t ygp;
	LaRankOutput *outp = NULL;
	unsigned p;
	for (p = 0; p < outputgradients.size (); ++p)
	{
		outputgradient_t & current = outputgradients[p];
		LaRankOutput *output = getOutput (current.output);
		bool support = ptype == processOptimize || output->isSupportVector (pattern.x_id);
		bool goodclass = current.output == pattern.y;
		if ((!support && goodclass) || (support && output->getBeta (pattern.x_id) < (goodclass ? C1 : 0)))
		{
			ygp = current;
			outp = output;
			break;
		}
	}
	if (p == outputgradients.size ())
		return pro_ret;

	/*
	 ** Find ym (2nd part of the pair)
	 */
	outputgradient_t ygm;
	LaRankOutput *outm = NULL;
	int m;
	for (m = outputgradients.size () - 1; m >= 0; --m)
	{
		outputgradient_t & current = outputgradients[m];
		LaRankOutput *output = getOutput (current.output);
		bool support = ptype == processOptimize || output->isSupportVector (pattern.x_id);
		bool goodclass = current.output == pattern.y;
		if (!goodclass || (support && output->getBeta (pattern.x_id) > 0))
		{
			ygm = current;
			outm = output;
			break;
		}
	}
	if (m < 0)
		return pro_ret;

	/*
	 ** Throw or Insert pattern
	 */
	if ((ygp.gradient - ygm.gradient) < tau)
		return pro_ret;
	if (ptype == processNew)
		patterns.insert (pattern);

	/*
	 ** compute lambda and clip it
	 */
	double kii = outp->getKii (pattern.x_id);
	double lambda = (ygp.gradient - ygm.gradient) / (2 * kii);
	if (ptype == processOptimize || outp->isSupportVector (pattern.x_id))
	{
		double beta = outp->getBeta (pattern.x_id);
		if (ygp.output == pattern.y)
			lambda = CMath::min (lambda, C1 - beta);
		else
			lambda = CMath::min (lambda, fabs (beta));
	}
	else
		lambda = CMath::min (lambda, C1);

	/*
	 ** update the solution
	 */
	outp->update (pattern.x_id, lambda, ygp.gradient);
	outm->update (pattern.x_id, -lambda, ygm.gradient);

	pro_ret.dual_increase = lambda * ((ygp.gradient - ygm.gradient) - lambda * kii);
	return pro_ret;
}

// ProcessOld
double CLaRank::reprocess ()
{
	if (patterns.size ())
		for (int n = 0; n < 10; ++n)
		{
			process_return_t pro_ret = process (patterns.sample (), processOld);
			if (pro_ret.dual_increase)
				return pro_ret.dual_increase;
		}
	return 0;
}

// Optimize
double CLaRank::optimize ()
{
	double dual_increase = 0;
	if (patterns.size ())
		for (int n = 0; n < 10; ++n)
		{
			process_return_t pro_ret =
				process (patterns.sample(), processOptimize);
			dual_increase += pro_ret.dual_increase;
		}
	return dual_increase;
}

// remove patterns and return the number of patterns that were removed
unsigned CLaRank::cleanup ()
{
	for (outputhash_t::iterator it = outputs.begin (); it != outputs.end (); ++it)
		it->second.cleanup ();
	unsigned res = 0;
	for (unsigned i = 0; i < patterns.size (); ++i)
	{
		LaRankPattern & p = patterns[i];
		if (p.exists () && !outputs[p.y].isSupportVector (p.x_id))
		{
			patterns.remove (i);
			++res;
		}
	}
	return res;
}

/***********************************************************************
 * 
 *  LUSH Lisp Universal Shell
 *    Copyright (C) 2002 Leon Bottou, Yann Le Cun, AT&T Corp, NECI.
 *  Includes parts of TL3:
 *    Copyright (C) 1987-1999 Leon Bottou and Neuristique.
 *  Includes selected parts of SN3.2:
 *    Copyright (C) 1991-2001 AT&T Corp.
 * 
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 * 
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 * 
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA
 * 
 ***********************************************************************/

/***********************************************************************
 * $Id: kcache.c,v 1.9 2007/01/25 22:42:09 leonb Exp $
 **********************************************************************/

#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "lib/io.h"
#include "lib/Mathematics.h"

namespace shogun
{
struct larank_kcache_s
{
  CKernel* func;
  larank_kcache_t *prevbuddy;
  larank_kcache_t *nextbuddy;
  long maxsize;
  long cursize;
  int l;
  int *i2r;
  int *r2i;
  int maxrowlen;
  /* Rows */
  int *rsize;
  float *rdiag;
  float **rdata;
  int *rnext;
  int *rprev;
  int *qnext;
  int *qprev;
};

static void *
xmalloc (int n)
{
  void *p = malloc (n);
  if (!p)
    SG_SERROR ("Function malloc() has returned zero\n");
  return p;
}

static void *
xrealloc (void *ptr, int n)
{
  if (!ptr)
    ptr = malloc (n);
  else
    ptr = realloc (ptr, n);
  if (!ptr)
    SG_SERROR ("Function realloc has returned zero\n");
  return ptr;
}

static void
xminsize (larank_kcache_t * self, int n)
{
  int ol = self->l;
  if (n > ol)
    {
      int i;
      int nl = CMath::max (256, ol);
      while (nl < n)
	nl = nl + nl;
      self->i2r = (int *) xrealloc (self->i2r, nl * sizeof (int));
      self->r2i = (int *) xrealloc (self->r2i, nl * sizeof (int));
      self->rsize = (int *) xrealloc (self->rsize, nl * sizeof (int));
      self->qnext = (int *) xrealloc (self->qnext, (1 + nl) * sizeof (int));
      self->qprev = (int *) xrealloc (self->qprev, (1 + nl) * sizeof (int));
      self->rdiag = (float *) xrealloc (self->rdiag, nl * sizeof (float));
      self->rdata = (float **) xrealloc (self->rdata, nl * sizeof (float *));
      self->rnext = self->qnext + 1;
      self->rprev = self->qprev + 1;
      for (i = ol; i < nl; i++)
	{
	  self->i2r[i] = i;
	  self->r2i[i] = i;
	  self->rsize[i] = -1;
	  self->rnext[i] = i;
	  self->rprev[i] = i;
	  self->rdata[i] = 0;
	}
      self->l = nl;
    }
}

larank_kcache_t *
larank_kcache_create (CKernel* kernelfunc)
{
  larank_kcache_t *self;
  self = (larank_kcache_t *) xmalloc (sizeof (larank_kcache_t));
  memset (self, 0, sizeof (larank_kcache_t));
  self->l = 0;
  self->maxrowlen = 0;
  self->func = kernelfunc;
  self->prevbuddy = self;
  self->nextbuddy = self;
  self->cursize = sizeof (larank_kcache_t);
  self->maxsize = 256 * 1024 * 1024;
  self->qprev = (int *) xmalloc (sizeof (int));
  self->qnext = (int *) xmalloc (sizeof (int));
  self->rnext = self->qnext + 1;
  self->rprev = self->qprev + 1;
  self->rprev[-1] = -1;
  self->rnext[-1] = -1;
  return self;
}

void
larank_kcache_destroy (larank_kcache_t * self)
{
  if (self)
    {
      int i;
      larank_kcache_t *nb = self->nextbuddy;
      larank_kcache_t *pb = self->prevbuddy;
      pb->nextbuddy = nb;
      nb->prevbuddy = pb;
      /* delete */
      if (self->i2r)
	free (self->i2r);
      if (self->r2i)
	free (self->r2i);
      if (self->rdata)
	for (i = 0; i < self->l; i++)
	  if (self->rdata[i])
	    free (self->rdata[i]);
      if (self->rdata)
	free (self->rdata);
      if (self->rsize)
	free (self->rsize);
      if (self->rdiag)
	free (self->rdiag);
      if (self->qnext)
	free (self->qnext);
      if (self->qprev)
	free (self->qprev);
      memset (self, 0, sizeof (larank_kcache_t));
      free (self);
    }
}

int *
larank_kcache_i2r (larank_kcache_t * self, int n)
{
  xminsize (self, n);
  return self->i2r;
}

int *
larank_kcache_r2i (larank_kcache_t * self, int n)
{
  xminsize (self, n);
  return self->r2i;
}

static void
xextend (larank_kcache_t * self, int k, int nlen)
{
  int olen = self->rsize[k];
  if (nlen > olen)
    {
      float *ndata = (float *) xmalloc (nlen * sizeof (float));
      if (olen > 0)
	{
	  float *odata = self->rdata[k];
	  memcpy (ndata, odata, olen * sizeof (float));
	  free (odata);
	}
      self->rdata[k] = ndata;
      self->rsize[k] = nlen;
      self->cursize += (long) (nlen - olen) * sizeof (float);
      self->maxrowlen = CMath::max (self->maxrowlen, nlen);
    }
}

static void
xtruncate (larank_kcache_t * self, int k, int nlen)
{
  int olen = self->rsize[k];
  if (nlen < olen)
    {
      float *ndata;
      float *odata = self->rdata[k];
      if (nlen > 0)
	{
	  ndata = (float *) xmalloc (nlen * sizeof (float));
	  memcpy (ndata, odata, nlen * sizeof (float));
	}
      else
	{
	  ndata = 0;
	  self->rnext[self->rprev[k]] = self->rnext[k];
	  self->rprev[self->rnext[k]] = self->rprev[k];
	  self->rnext[k] = self->rprev[k] = k;
	}
      free (odata);
      self->rdata[k] = ndata;
      self->rsize[k] = nlen;
      self->cursize += (long) (nlen - olen) * sizeof (float);
    }
}

static void
xswap (larank_kcache_t * self, int i1, int i2, int r1, int r2)
{
  /* swap row data */
  if (r1 < self->maxrowlen || r2 < self->maxrowlen)
    {
      int mrl = 0;
      int k = self->rnext[-1];
      while (k >= 0)
	{
	  int nk = self->rnext[k];
	  int n = self->rsize[k];
	  int rr = self->i2r[k];
	  float *d = self->rdata[k];
	  if (r1 < n)
	    {
	      if (r2 < n)
		{
		  float t1 = d[r1];
		  float t2 = d[r2];
		  d[r1] = t2;
		  d[r2] = t1;
		}
	      else if (rr == r2)
		{
		  d[r1] = self->rdiag[k];
		}
	      else
		{
		  int arsize = self->rsize[i2];
		  if (rr < arsize && rr != r1)
		    d[r1] = self->rdata[i2][rr];
		  else
		    xtruncate (self, k, r1);
		}
	    }
	  else if (r2 < n)
	    {
	      if (rr == r1)
		{
		  d[r2] = self->rdiag[k];
		}
	      else
		{
		  int arsize = self->rsize[i1];
		  if (rr < arsize && rr != r2)
		    d[r2] = self->rdata[i1][rr];
		  else
		    xtruncate (self, k, r2);
		}
	    }
	  mrl = CMath::max (mrl, self->rsize[k]);
	  k = nk;
	}
      self->maxrowlen = mrl;
    }
  /* swap r2i and i2r */
  self->r2i[r1] = i2;
  self->r2i[r2] = i1;
  self->i2r[i1] = r2;
  self->i2r[i2] = r1;
}

void
larank_kcache_swap_rr (larank_kcache_t * self, int r1, int r2)
{
  xminsize (self, 1 + CMath::max (r1, r2));
  xswap (self, self->r2i[r1], self->r2i[r2], r1, r2);
}

void
larank_kcache_swap_ii (larank_kcache_t * self, int i1, int i2)
{
  xminsize (self, 1 + CMath::max (i1, i2));
  xswap (self, i1, i2, self->i2r[i1], self->i2r[i2]);
}

void
larank_kcache_swap_ri (larank_kcache_t * self, int r1, int i2)
{
  xminsize (self, 1 + CMath::max (r1, i2));
  xswap (self, self->r2i[r1], i2, r1, self->i2r[i2]);
}

static double
xquery (larank_kcache_t * self, int i, int j)
{
  /* search buddies */
  larank_kcache_t *cache = self->nextbuddy;
  do
    {
      int l = cache->l;
      if (i < l && j < l)
	{
	  int s = cache->rsize[i];
	  int p = cache->i2r[j];
	  if (p < s)
	    return cache->rdata[i][p];
	  if (i == j && s >= 0)
	    return cache->rdiag[i];
	  p = cache->i2r[i];
	  s = cache->rsize[j];
	  if (p < s)
	    return cache->rdata[j][p];
	}
      cache = cache->nextbuddy;
    }
  while (cache != self);
  /* compute */
  return self->func->kernel(i, j);
}


double
larank_kcache_query (larank_kcache_t * self, int i, int j)
{
  ASSERT (self);
  ASSERT (i >= 0);
  ASSERT (j >= 0);
  return xquery (self, i, j);
}


static void
xpurge (larank_kcache_t * self)
{
  if (self->cursize > self->maxsize)
    {
      int k = self->rprev[-1];
      while (self->cursize > self->maxsize && k != self->rnext[-1])
	{
	  int pk = self->rprev[k];
	  xtruncate (self, k, 0);
	  k = pk;
	}
    }
}

float *
larank_kcache_query_row (larank_kcache_t * self, int i, int len)
{
  ASSERT (i >= 0);
  if (i < self->l && len <= self->rsize[i])
    {
      self->rnext[self->rprev[i]] = self->rnext[i];
      self->rprev[self->rnext[i]] = self->rprev[i];
    }
  else
    {
      int olen, p;
      float *d;
      if (i >= self->l || len >= self->l)
	xminsize (self, CMath::max (1 + i, len));
      olen = self->rsize[i];
      if (olen < len)
	{
	  if (olen < 0)
	    {
	      self->rdiag[i] = self->func->kernel(i, i);
	      olen = self->rsize[i] = 0;
	    }
	  xextend (self, i, len);
	  d = self->rdata[i];
	  self->rsize[i] = olen;
	  for (p = olen; p < len; p++)
	    d[p] = larank_kcache_query (self, self->r2i[p], i);
	  self->rsize[i] = len;
	}
      self->rnext[self->rprev[i]] = self->rnext[i];
      self->rprev[self->rnext[i]] = self->rprev[i];
      xpurge (self);
    }
  self->rprev[i] = -1;
  self->rnext[i] = self->rnext[-1];
  self->rnext[self->rprev[i]] = i;
  self->rprev[self->rnext[i]] = i;
  return self->rdata[i];
}

int
larank_kcache_status_row (larank_kcache_t * self, int i)
{
  ASSERT (self);
  ASSERT (i >= 0);
  if (i < self->l)
    return CMath::max (0, self->rsize[i]);
  return 0;
}

void
larank_kcache_discard_row (larank_kcache_t * self, int i)
{
  ASSERT (self);
  ASSERT (i >= 0);
  if (i < self->l && self->rsize[i] > 0)
    {
      self->rnext[self->rprev[i]] = self->rnext[i];
      self->rprev[self->rnext[i]] = self->rprev[i];
      self->rprev[i] = self->rprev[-1];
      self->rnext[i] = -1;
      self->rnext[self->rprev[i]] = i;
      self->rprev[self->rnext[i]] = i;
    }
}

void
larank_kcache_set_maximum_size (larank_kcache_t * self, long entries)
{
  ASSERT (self);
  ASSERT (entries > 0);
  self->maxsize = entries;
  xpurge (self);
}

long
larank_kcache_get_maximum_size (larank_kcache_t * self)
{
  ASSERT (self);
  return self->maxsize;
}

long
larank_kcache_get_current_size (larank_kcache_t * self)
{
  ASSERT (self);
  return self->cursize;
}

void
larank_kcache_set_buddy (larank_kcache_t * self, larank_kcache_t * buddy)
{
  larank_kcache_t *p = self;
  larank_kcache_t *selflast = self->prevbuddy;
  larank_kcache_t *buddylast = buddy->prevbuddy;
  /* check functions are identical */
  ASSERT (self->func == buddy->func);
  /* make sure we are not already buddies */
  do
    {
      if (p == buddy)
	return;
      p = p->nextbuddy;
    }
  while (p != self);
  /* link */
  selflast->nextbuddy = buddy;
  buddy->prevbuddy = selflast;
  buddylast->nextbuddy = self;
  self->prevbuddy = buddylast;
}
}
