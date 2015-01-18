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
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
//
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
 *  aint64_t with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA
 *
 ***********************************************************************/

/***********************************************************************
 * $Id: kcache.c,v 1.9 2007/01/25 22:42:09 leonb Exp $
 **********************************************************************/

#include <vector>
#include <algorithm>
#include <ctime>
#include <algorithm>
#include <sys/time.h>

#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/Time.h>
#include <shogun/mathematics/Math.h>
#include <shogun/multiclass/LaRank.h>
#include <shogun/multiclass/MulticlassOneVsRestStrategy.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/labels/MulticlassLabels.h>

using namespace shogun;

namespace shogun
{
	static larank_kcache_t* larank_kcache_create (CKernel* kernelfunc)
	{
		larank_kcache_t *self;
		self = SG_CALLOC (larank_kcache_t, 1);
		self->l = 0;
		self->maxrowlen = 0;
		self->func = kernelfunc;
		self->prevbuddy = self;
		self->nextbuddy = self;
		self->cursize = sizeof (larank_kcache_t);
		self->maxsize = 256 * 1024 * 1024;
		self->qprev = SG_MALLOC(int32_t, 1);
		self->qnext = SG_MALLOC(int32_t, 1);
		self->rnext = self->qnext + 1;
		self->rprev = self->qprev + 1;
		self->rprev[-1] = -1;
		self->rnext[-1] = -1;
		return self;
	}

	static void xtruncate (larank_kcache_t * self, int32_t k, int32_t nlen)
	{
		int32_t olen = self->rsize[k];
		if (nlen < olen)
		{
			float32_t *ndata;
			float32_t *odata = self->rdata[k];
			if (nlen > 0)
			{
				ndata = SG_MALLOC(float32_t, nlen);
				memcpy (ndata, odata, nlen * sizeof (float32_t));
			}
			else
			{
				ndata = 0;
				self->rnext[self->rprev[k]] = self->rnext[k];
				self->rprev[self->rnext[k]] = self->rprev[k];
				self->rnext[k] = self->rprev[k] = k;
			}
			SG_FREE (odata);
			self->rdata[k] = ndata;
			self->rsize[k] = nlen;
			self->cursize += (int64_t) (nlen - olen) * sizeof (float32_t);
		}
	}

	static void xpurge (larank_kcache_t * self)
	{
		if (self->cursize > self->maxsize)
		{
			int32_t k = self->rprev[-1];
			while (self->cursize > self->maxsize && k != self->rnext[-1])
			{
				int32_t pk = self->rprev[k];
				xtruncate (self, k, 0);
				k = pk;
			}
		}
	}

	static void larank_kcache_set_maximum_size (larank_kcache_t * self, int64_t entries)
	{
		ASSERT (self)
		ASSERT (entries > 0)
		self->maxsize = entries;
		xpurge (self);
	}

	static void larank_kcache_destroy (larank_kcache_t * self)
	{
		if (self)
		{
			int32_t i;
			larank_kcache_t *nb = self->nextbuddy;
			larank_kcache_t *pb = self->prevbuddy;
			pb->nextbuddy = nb;
			nb->prevbuddy = pb;
			/* delete */
			if (self->i2r)
				SG_FREE (self->i2r);
			if (self->r2i)
				SG_FREE (self->r2i);
			if (self->rdata)
				for (i = 0; i < self->l; i++)
					if (self->rdata[i])
						SG_FREE (self->rdata[i]);
			if (self->rdata)
				SG_FREE (self->rdata);
			if (self->rsize)
				SG_FREE (self->rsize);
			if (self->rdiag)
				SG_FREE (self->rdiag);
			if (self->qnext)
				SG_FREE (self->qnext);
			if (self->qprev)
				SG_FREE (self->qprev);
			memset (self, 0, sizeof (larank_kcache_t));
			SG_FREE (self);
		}
	}

	static void xminsize (larank_kcache_t * self, int32_t n)
	{
		int32_t ol = self->l;
		if (n > ol)
		{
			int32_t i;
			int32_t nl = CMath::max (256, ol);
			while (nl < n)
				nl = nl + nl;
			self->i2r = SG_REALLOC (int32_t, self->i2r, self->l, nl);
			self->r2i = SG_REALLOC (int32_t, self->r2i, self->l, nl);
			self->rsize = SG_REALLOC (int32_t, self->rsize, self->l, nl);
			self->qnext = SG_REALLOC (int32_t, self->qnext, 1+self->l, (1 + nl));
			self->qprev = SG_REALLOC (int32_t, self->qprev, 1+self->l, (1 + nl));
			self->rdiag = SG_REALLOC (float32_t, self->rdiag, self->l, nl);
			self->rdata = SG_REALLOC (float32_t*, self->rdata, self->l, nl);
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

	static int32_t* larank_kcache_r2i (larank_kcache_t * self, int32_t n)
	{
		xminsize (self, n);
		return self->r2i;
	}

	static void xextend (larank_kcache_t * self, int32_t k, int32_t nlen)
	{
		int32_t olen = self->rsize[k];
		if (nlen > olen)
		{
			float32_t *ndata = SG_MALLOC(float32_t, nlen);
			if (olen > 0)
			{
				float32_t *odata = self->rdata[k];
				memcpy (ndata, odata, olen * sizeof (float32_t));
				SG_FREE (odata);
			}
			self->rdata[k] = ndata;
			self->rsize[k] = nlen;
			self->cursize += (int64_t) (nlen - olen) * sizeof (float32_t);
			self->maxrowlen = CMath::max (self->maxrowlen, nlen);
		}
	}

	static void xswap (larank_kcache_t * self, int32_t i1, int32_t i2, int32_t r1, int32_t r2)
	{
		/* swap row data */
		if (r1 < self->maxrowlen || r2 < self->maxrowlen)
		{
			int32_t mrl = 0;
			int32_t k = self->rnext[-1];
			while (k >= 0)
			{
				int32_t nk = self->rnext[k];
				int32_t n = self->rsize[k];
				int32_t rr = self->i2r[k];
				float32_t *d = self->rdata[k];
				if (r1 < n)
				{
					if (r2 < n)
					{
						float32_t t1 = d[r1];
						float32_t t2 = d[r2];
						d[r1] = t2;
						d[r2] = t1;
					}
					else if (rr == r2)
					{
						d[r1] = self->rdiag[k];
					}
					else
					{
						int32_t arsize = self->rsize[i2];
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
						int32_t arsize = self->rsize[i1];
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

	static void larank_kcache_swap_rr (larank_kcache_t * self, int32_t r1, int32_t r2)
	{
		xminsize (self, 1 + CMath::max(r1, r2));
		xswap (self, self->r2i[r1], self->r2i[r2], r1, r2);
	}

	static void larank_kcache_swap_ri (larank_kcache_t * self, int32_t r1, int32_t i2)
	{
		xminsize (self, 1 + CMath::max (r1, i2));
		xswap (self, self->r2i[r1], i2, r1, self->i2r[i2]);
	}

	static float64_t xquery (larank_kcache_t * self, int32_t i, int32_t j)
	{
		/* search buddies */
		larank_kcache_t *cache = self->nextbuddy;
		do
		{
			int32_t l = cache->l;
			if (i < l && j < l)
			{
				int32_t s = cache->rsize[i];
				int32_t p = cache->i2r[j];
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


	static float64_t larank_kcache_query (larank_kcache_t * self, int32_t i, int32_t j)
	{
		ASSERT (self)
		ASSERT (i >= 0)
		ASSERT (j >= 0)
		return xquery (self, i, j);
	}


	static void larank_kcache_set_buddy (larank_kcache_t * self, larank_kcache_t * buddy)
	{
		larank_kcache_t *p = self;
		larank_kcache_t *selflast = self->prevbuddy;
		larank_kcache_t *buddylast = buddy->prevbuddy;
		/* check functions are identical */
		ASSERT (self->func == buddy->func)
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


	static float32_t* larank_kcache_query_row (larank_kcache_t * self, int32_t i, int32_t len)
	{
		ASSERT (i >= 0)
		if (i < self->l && len <= self->rsize[i])
		{
			self->rnext[self->rprev[i]] = self->rnext[i];
			self->rprev[self->rnext[i]] = self->rprev[i];
		}
		else
		{
			int32_t olen, p;
			float32_t *d;
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

}


// Initializing an output class (basically creating a kernel cache for it)
void LaRankOutput::initialize (CKernel* kfunc, int64_t cache)
{
	kernel = larank_kcache_create (kfunc);
	larank_kcache_set_maximum_size (kernel, cache * 1024 * 1024);
	beta = SG_MALLOC(float32_t, 1);
	g = SG_MALLOC(float32_t, 1);
	*beta=0;
	*g=0;
	l = 0;
}

// Destroying an output class (basically destroying the kernel cache)
void LaRankOutput::destroy ()
{
	larank_kcache_destroy (kernel);
	kernel=NULL;
	SG_FREE(beta);
	SG_FREE(g);
	beta=NULL;
	g=NULL;
}

// !Important! Computing the score of a given input vector for the actual output
float64_t LaRankOutput::computeScore (int32_t x_id)
{
	if (l == 0)
		return 0;
	else
	{
		float32_t *row = larank_kcache_query_row (kernel, x_id, l);
		return CMath::dot (beta, row, l);
	}
}

// !Important! Computing the gradient of a given input vector for the actual output
float64_t LaRankOutput::computeGradient (int32_t xi_id, int32_t yi, int32_t ythis)
{
	return (yi == ythis ? 1 : 0) - computeScore (xi_id);
}

// Updating the solution in the actual output
void LaRankOutput::update (int32_t x_id, float64_t lambda, float64_t gp)
{
	int32_t *r2i = larank_kcache_r2i (kernel, l);
	int64_t xr = l + 1;
	for (int32_t r = 0; r < l; r++)
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
		g = SG_REALLOC(float32_t, g, l, l+1);
		beta = SG_REALLOC(float32_t, beta, l, l+1);
		g[l]=gp;
		beta[l]=lambda;
		l++;
	}

	// update stored gradients
	float32_t *row = larank_kcache_query_row (kernel, x_id, l);
	for (int32_t r = 0; r < l; r++)
	{
		float64_t oldg = g[r];
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
int32_t LaRankOutput::cleanup ()
{
	int32_t count = 0;
	std::vector < int32_t >idx;
	for (int32_t x = 0; x < l; x++)
	{
		if ((beta[x] < FLT_EPSILON) && (beta[x] > -FLT_EPSILON))
		{
			idx.push_back (x);
			count++;
		}
	}
	int32_t new_l = l - count;
	for (int32_t xx = 0; xx < count; xx++)
	{
		int32_t i = idx[xx] - xx;
		for (int32_t r = i; r < (l - 1); r++)
		{
			larank_kcache_swap_rr (kernel, r, int64_t(r) + 1);
			beta[r]=beta[r + 1];
			g[r]=g[r + 1];
		}
	}
	beta = SG_REALLOC(float32_t, beta, l, new_l+1);
	g = SG_REALLOC(float32_t, g, l, new_l+1);
	beta[new_l]=0;
	g[new_l]=0;
	l = new_l;
	return count;
}

// --- Below are information or "get" functions --- //
//
float64_t LaRankOutput::getW2 ()
{
	float64_t sum = 0;
	int32_t *r2i = larank_kcache_r2i (kernel, l + 1);
	for (int32_t r = 0; r < l; r++)
	{
		float32_t *row_r = larank_kcache_query_row (kernel, r2i[r], l);
		sum += beta[r] * CMath::dot (beta, row_r, l);
	}
	return sum;
}

float64_t LaRankOutput::getKii (int32_t x_id)
{
	return larank_kcache_query (kernel, x_id, x_id);
}

//
float64_t LaRankOutput::getBeta (int32_t x_id)
{
	int32_t *r2i = larank_kcache_r2i (kernel, l);
	int32_t xr = -1;
	for (int32_t r = 0; r < l; r++)
		if (r2i[r] == x_id)
		{
			xr = r;
			break;
		}
	return (xr < 0 ? 0 : beta[xr]);
}

//
float64_t LaRankOutput::getGradient (int32_t x_id)
{
	int32_t *r2i = larank_kcache_r2i (kernel, l);
	int32_t xr = -1;
	for (int32_t r = 0; r < l; r++)
		if (r2i[r] == x_id)
		{
			xr = r;
			break;
		}
	return (xr < 0 ? 0 : g[xr]);
}
bool LaRankOutput::isSupportVector (int32_t x_id) const
{
	int32_t *r2i = larank_kcache_r2i (kernel, l);
	int32_t xr = -1;
	for (int32_t r = 0; r < l; r++)
		if (r2i[r] == x_id)
		{
			xr = r;
			break;
		}
	return (xr >= 0);
}

//
int32_t LaRankOutput::getSV (float32_t* &sv) const
{
	sv=SG_MALLOC(float32_t, l);
	int32_t *r2i = larank_kcache_r2i (kernel, l);
	for (int32_t r = 0; r < l; r++)
		sv[r]=r2i[r];
	return l;
}

CLaRank::CLaRank (): CMulticlassSVM(new CMulticlassOneVsRestStrategy()),
	nb_seen_examples (0), nb_removed (0),
	n_pro (0), n_rep (0), n_opt (0),
	w_pro (1), w_rep (1), w_opt (1), y0 (0), m_dual (0),
	batch_mode(true), step(0)
{
}

CLaRank::CLaRank (float64_t C, CKernel* k, CLabels* lab):
	CMulticlassSVM(new CMulticlassOneVsRestStrategy(), C, k, lab),
	nb_seen_examples (0), nb_removed (0),
	n_pro (0), n_rep (0), n_opt (0),
	w_pro (1), w_rep (1), w_opt (1), y0 (0), m_dual (0),
	batch_mode(true), step(0)
{
}

CLaRank::~CLaRank ()
{
	destroy();
}

bool CLaRank::train_machine(CFeatures* data)
{
	tau = 0.0001;

	ASSERT(m_kernel)
	ASSERT(m_labels && m_labels->get_num_labels())
	ASSERT(m_labels->get_label_type() == LT_MULTICLASS)

	CSignal::clear_cancel();

	if (data)
	{
		if (data->get_num_vectors() != m_labels->get_num_labels())
		{
			SG_ERROR("Numbert of vectors (%d) does not match number of labels (%d)\n",
					data->get_num_vectors(), m_labels->get_num_labels());
		}
		m_kernel->init(data, data);
	}

	ASSERT(m_kernel->get_num_vec_lhs() && m_kernel->get_num_vec_rhs())

	nb_train=m_labels->get_num_labels();
	cache = m_kernel->get_cache_size();

	int32_t n_it = 1;
	float64_t gap = DBL_MAX;

	SG_INFO("Training on %d examples\n", nb_train)
	while (gap > get_C() && (!CSignal::cancel_computations()))      // stopping criteria
	{
		float64_t tr_err = 0;
		int32_t ind = step;
		for (int32_t i = 0; i < nb_train; i++)
		{
			int32_t y=((CMulticlassLabels*) m_labels)->get_label(i);
			if (add (i, y) != y)   // call the add function
				tr_err++;

			if (ind && i / ind)
			{
				SG_DEBUG("Done: %d %% Train error (online): %f%%\n",
						(int32_t) (((float64_t) i) / nb_train * 100), (tr_err / ((float64_t) i + 1)) * 100);
				ind += step;
			}
		}

		SG_DEBUG("End of iteration %d\n", n_it++)
		SG_DEBUG("Train error (online): %f%%\n", (tr_err / nb_train) * 100)
		gap = computeGap ();
		SG_ABS_PROGRESS(gap, -CMath::log10(gap), -CMath::log10(DBL_MAX), -CMath::log10(get_C()), 6)

		if (!batch_mode)        // skip stopping criteria if online mode
			gap = 0;
	}
	SG_DONE()

	int32_t num_classes = outputs.size();
	create_multiclass_svm(num_classes);
	SG_DEBUG("%d classes\n", num_classes)

	// Used for saving a model file
	int32_t i=0;
	for (outputhash_t::const_iterator it = outputs.begin (); it != outputs.end (); ++it)
	{
		const LaRankOutput* o=&(it->second);

		larank_kcache_t* k=o->getKernel();
		int32_t l=o->get_l();
		float32_t* beta=o->getBetas();
		int32_t *r2i = larank_kcache_r2i (k, l);

		ASSERT(l>0)
		SG_DEBUG("svm[%d] has %d sv, b=%f\n", i, l, 0.0)

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
int32_t CLaRank::add (int32_t x_id, int32_t yi)
{
	++nb_seen_examples;
	// create a new output object if this one has never been seen before
	if (!getOutput (yi))
	{
		outputs.insert (std::make_pair (yi, LaRankOutput ()));
		LaRankOutput *cur = getOutput (yi);
		cur->initialize (m_kernel, cache);
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
	float64_t time1 = CTime::get_curtime();
	process_return_t pro_ret = process (pattern, processNew);
	float64_t dual_increase = pro_ret.dual_increase;
	float64_t duration = (CTime::get_curtime() - time1);
	float64_t coeff = dual_increase / (0.00001 + duration);
	m_dual += dual_increase;
	n_pro++;
	w_pro = 0.05 * coeff + (1 - 0.05) * w_pro;

	// ProcessOld & Optimize until ready for a new processnew
	// (Adaptative schedule here)
	for (;;)
	{
		float64_t w_sum = w_pro + w_rep + w_opt;
		float64_t prop_min = w_sum / 20;
		if (w_pro < prop_min)
			w_pro = prop_min;
		if (w_rep < prop_min)
			w_rep = prop_min;
		if (w_opt < prop_min)
			w_opt = prop_min;
		w_sum = w_pro + w_rep + w_opt;
		float64_t r = CMath::random(0.0, w_sum);
		if (r <= w_pro)
		{
			break;
		}
		else if ((r > w_pro) && (r <= w_pro + w_rep))	// ProcessOld here
		{
			float64_t ltime1 = CTime::get_curtime ();
			float64_t ldual_increase = reprocess ();
			float64_t lduration = (CTime::get_curtime () - ltime1);
			float64_t lcoeff = ldual_increase / (0.00001 + lduration);
			m_dual += ldual_increase;
			n_rep++;
			w_rep = 0.05 * lcoeff + (1 - 0.05) * w_rep;
		}
		else			// Optimize here
		{
			float64_t ltime1 = CTime::get_curtime ();
			float64_t ldual_increase = optimize ();
			float64_t lduration = (CTime::get_curtime () - ltime1);
			float64_t lcoeff = ldual_increase / (0.00001 + lduration);
			m_dual += ldual_increase;
			n_opt++;
			w_opt = 0.05 * lcoeff + (1 - 0.05) * w_opt;
		}
	}
	if (nb_seen_examples % 100 == 0)	// Cleanup useless Support Vectors/Patterns sometimes
		nb_removed += cleanup ();
	return pro_ret.ypred;
}

// PREDICTION FUNCTION: main function in la_rank_classify
int32_t CLaRank::predict (int32_t x_id)
{
	int32_t res = -1;
	float64_t score_max = -DBL_MAX;
	for (outputhash_t::iterator it = outputs.begin (); it != outputs.end ();++it)
	{
		float64_t score = it->second.computeScore (x_id);
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
	outputs.clear();
}


// Compute Duality gap (costly but used in stopping criteria in batch mode)
float64_t CLaRank::computeGap ()
{
	float64_t sum_sl = 0;
	float64_t sum_bi = 0;
	for (uint32_t i = 0; i < patterns.maxcount (); ++i)
	{
		const LaRankPattern & p = patterns[i];
		if (!p.exists ())
			continue;
		LaRankOutput *out = getOutput (p.y);
		if (!out)
			continue;
		sum_bi += out->getBeta (p.x_id);
		float64_t gi = out->computeGradient (p.x_id, p.y, p.y);
		float64_t gmin = DBL_MAX;
		for (outputhash_t::iterator it = outputs.begin (); it != outputs.end (); ++it)
		{
			if (it->first != p.y && it->second.isSupportVector (p.x_id))
			{
				float64_t g =
					it->second.computeGradient (p.x_id, p.y, it->first);
				if (g < gmin)
					gmin = g;
			}
		}
		sum_sl += CMath::max (0.0, gi - gmin);
	}
	return CMath::max (0.0, computeW2 () + get_C() * sum_sl - sum_bi);
}

// Nuber of classes so far
uint32_t CLaRank::getNumOutputs () const
{
	return outputs.size ();
}

// Number of Support Vectors
int32_t CLaRank::getNSV ()
{
	int32_t res = 0;
	for (outputhash_t::const_iterator it = outputs.begin (); it != outputs.end (); ++it)
	{
		float32_t* sv=NULL;
		res += it->second.getSV (sv);
		SG_FREE(sv);
	}
	return res;
}

// Norm of the parameters vector
float64_t CLaRank::computeW2 ()
{
	float64_t res = 0;
	for (uint32_t i = 0; i < patterns.maxcount (); ++i)
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
float64_t CLaRank::getDual ()
{
	float64_t res = 0;
	for (uint32_t i = 0; i < patterns.maxcount (); ++i)
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

LaRankOutput *CLaRank::getOutput (int32_t index)
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
	{
		if (ptype != processOptimize
				|| it->second.isSupportVector (pattern.x_id))
		{
			float64_t g =
				it->second.computeGradient (pattern.x_id, pattern.y, it->first);
			outputgradients.push_back (outputgradient_t (it->first, g));
			if (it->first == pattern.y)
				outputscores.push_back (outputgradient_t (it->first, (1 - g)));
			else
				outputscores.push_back (outputgradient_t (it->first, -g));
		}
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
	uint32_t p;
	for (p = 0; p < outputgradients.size (); ++p)
	{
		outputgradient_t & current = outputgradients[p];
		LaRankOutput *output = getOutput (current.output);
		bool support = ptype == processOptimize || output->isSupportVector (pattern.x_id);
		bool goodclass = current.output == pattern.y;
		if ((!support && goodclass) ||
				(support && output->getBeta (pattern.x_id) < (goodclass ? get_C() : 0)))
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
	int32_t m;
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
	float64_t kii = outp->getKii (pattern.x_id);
	float64_t lambda = (ygp.gradient - ygm.gradient) / (2 * kii);
	if (ptype == processOptimize || outp->isSupportVector (pattern.x_id))
	{
		float64_t beta = outp->getBeta (pattern.x_id);
		if (ygp.output == pattern.y)
			lambda = CMath::min (lambda, get_C() - beta);
		else
			lambda = CMath::min (lambda, fabs (beta));
	}
	else
		lambda = CMath::min (lambda, get_C());

	/*
	 ** update the solution
	 */
	outp->update (pattern.x_id, lambda, ygp.gradient);
	outm->update (pattern.x_id, -lambda, ygm.gradient);

	pro_ret.dual_increase = lambda * ((ygp.gradient - ygm.gradient) - lambda * kii);
	return pro_ret;
}

// ProcessOld
float64_t CLaRank::reprocess ()
{
	if (patterns.size ())
	{
		for (int32_t n = 0; n < 10; ++n)
		{
			process_return_t pro_ret = process (patterns.sample (), processOld);
			if (pro_ret.dual_increase)
				return pro_ret.dual_increase;
		}
	}
	return 0;
}

// Optimize
float64_t CLaRank::optimize ()
{
	float64_t dual_increase = 0;
	if (patterns.size ())
	{
		for (int32_t n = 0; n < 10; ++n)
		{
			process_return_t pro_ret =
				process (patterns.sample(), processOptimize);
			dual_increase += pro_ret.dual_increase;
		}
	}
	return dual_increase;
}

// remove patterns and return the number of patterns that were removed
uint32_t CLaRank::cleanup ()
{
	/*
	for (outputhash_t::iterator it = outputs.begin (); it != outputs.end (); ++it)
		it->second.cleanup ();

	uint32_t res = 0;
	for (uint32_t i = 0; i < patterns.size (); ++i)
	{
		LaRankPattern & p = patterns[i];
		if (p.exists () && !outputs[p.y].isSupportVector (p.x_id))
		{
			patterns.remove (i);
			++res;
		}
	}
	return res;
	*/
	return 0;
}
