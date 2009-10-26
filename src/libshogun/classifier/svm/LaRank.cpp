// -*- C++ -*-
// Main functions of the LaRank algorithm for soving Multiclass SVM
// Copyright (C) 2008- Antoine Bordes

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
#include "lib/Mathematics.h"
#include "classifier/svm/kcache.h"
#include "classifier/svm/LaRank.h"

using namespace shogun;
// Initializing an output class (basically creating a kernel cache for it)
void LaRankOutput::initialize (larank_kernel_t kfunc, long cache)
{
	kernel = larank_kcache_create (kfunc, NULL);
	larank_kcache_set_maximum_size (kernel, cache * 1024 * 1024);
	l = 0;
}

// Destroying an output class (basically destroying the kernel cache)
void LaRankOutput::destroy ()
{
	larank_kcache_destroy (kernel);
	delete[] beta;
	delete[] g;
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
		g[l]=gp;
		beta[l]=lambda;
		larank_kcache_swap_ri (kernel, l, x_id);
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
	l = new_l;
	float* beta_resized=CMath::clone_vector(beta, l);
	delete[] beta;
	beta=beta_resized;

	float* g_resized=CMath::clone_vector(g, l);
	g=g_resized;
	return count;
}

// --- Below are information or "get" functions --- //

//                            
larank_kcache_t *LaRankOutput::getKernel ()
{
	return kernel;
}

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

CLaRank::CLaRank (): nb_seen_examples (0), nb_removed (0),
	n_pro (0), n_rep (0), n_opt (0),
	w_pro (1), w_rep (1), w_opt (1), y0 (0), dual (0)
{
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
		cur->initialize (kfunc, cache);
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
	return CMath::max (0.0, computeW2 () + C * sum_sl - sum_bi);
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
		if ((!support && goodclass) || (support && output->getBeta (pattern.x_id) < (goodclass ? C : 0)))
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
			lambda = CMath::min (lambda, C - beta);
		else
			lambda = CMath::min (lambda, fabs (beta));
	}
	else
		lambda = CMath::min (lambda, C);

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
				process (patterns.sample (), processOptimize);
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
