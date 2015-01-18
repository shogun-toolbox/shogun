/*
 * Copyright (c) 2000-2009 Chih-Chung Chang and Chih-Jen Lin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. Neither name of copyright holders nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Shogun specific adjustments (w) 2006-2009 Soeren Sonnenburg
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <shogun/lib/external/shogun_libsvm.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Time.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

namespace shogun
{

typedef KERNELCACHE_ELEM Qfloat;
typedef float64_t schar;

template <class S, class T> inline void clone(T*& dst, S* src, int32_t n)
{
	dst = SG_MALLOC(T, n);
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define INF HUGE_VAL
#define TAU 1e-12

class QMatrix;
class SVC_QMC;

//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//
class Cache
{
public:
	Cache(int32_t l, int64_t size);
	~Cache();

	// request data [0,len)
	// return some position p where [p,len) need to be filled
	// (p >= len if nothing needs to be filled)
	int32_t get_data(const int32_t index, Qfloat **data, int32_t len);
	void swap_index(int32_t i, int32_t j);	// future_option

private:
	int32_t l;
	int64_t size;
	struct head_t
	{
		head_t *prev, *next;	// a circular list
		Qfloat *data;
		int32_t len;		// data[0,len) is cached in this entry
	};

	head_t *head;
	head_t lru_head;
	void lru_delete(head_t *h);
	void lru_insert(head_t *h);
};

Cache::Cache(int32_t l_, int64_t size_):l(l_),size(size_)
{
	head = (head_t *)SG_CALLOC(head_t, l);	// initialized to 0
	size /= sizeof(Qfloat);
	size -= l * sizeof(head_t) / sizeof(Qfloat);
	size = CMath::max(size, (int64_t) 2*l);	// cache must be large enough for two columns
	lru_head.next = lru_head.prev = &lru_head;
}

Cache::~Cache()
{
	for(head_t *h = lru_head.next; h != &lru_head; h=h->next)
		SG_FREE(h->data);
	SG_FREE(head);
}

void Cache::lru_delete(head_t *h)
{
	// delete from current location
	h->prev->next = h->next;
	h->next->prev = h->prev;
}

void Cache::lru_insert(head_t *h)
{
	// insert to last position
	h->next = &lru_head;
	h->prev = lru_head.prev;
	h->prev->next = h;
	h->next->prev = h;
}

int32_t Cache::get_data(const int32_t index, Qfloat **data, int32_t len)
{
	head_t *h = &head[index];
	if(h->len) lru_delete(h);
	int32_t more = len - h->len;

	if(more > 0)
	{
		// free old space
		while(size < more)
		{
			head_t *old = lru_head.next;
			lru_delete(old);
			SG_FREE(old->data);
			size += old->len;
			old->data = 0;
			old->len = 0;
		}

		// allocate new space
		h->data = SG_REALLOC(Qfloat, h->data, h->len, len);
		size -= more;
		CMath::swap(h->len,len);
	}

	lru_insert(h);
	*data = h->data;
	return len;
}

void Cache::swap_index(int32_t i, int32_t j)
{
	if(i==j) return;

	if(head[i].len) lru_delete(&head[i]);
	if(head[j].len) lru_delete(&head[j]);
	CMath::swap(head[i].data,head[j].data);
	CMath::swap(head[i].len,head[j].len);
	if(head[i].len) lru_insert(&head[i]);
	if(head[j].len) lru_insert(&head[j]);

	if(i>j) CMath::swap(i,j);
	for(head_t *h = lru_head.next; h!=&lru_head; h=h->next)
	{
		if(h->len > i)
		{
			if(h->len > j)
				CMath::swap(h->data[i],h->data[j]);
			else
			{
				// give up
				lru_delete(h);
				SG_FREE(h->data);
				size += h->len;
				h->data = 0;
				h->len = 0;
			}
		}
	}
}

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
class QMatrix {
public:
	virtual Qfloat *get_Q(int32_t column, int32_t len) const = 0;
	virtual Qfloat *get_QD() const = 0;
	virtual void swap_index(int32_t i, int32_t j) const = 0;
	virtual ~QMatrix() {}

	float64_t max_train_time;
};

class LibSVMKernel;

// helper struct for threaded processing
struct Q_THREAD_PARAM
{
	int32_t i;
	int32_t start;
	int32_t end;
	Qfloat* data;
	float64_t* y;
	const LibSVMKernel* q;
};

extern Parallel* sg_parallel;

class LibSVMKernel: public QMatrix {
public:
	LibSVMKernel(int32_t l, svm_node * const * x, const svm_parameter& param);
	virtual ~LibSVMKernel();

	virtual Qfloat *get_Q(int32_t column, int32_t len) const = 0;
	virtual Qfloat *get_QD() const = 0;
	virtual void swap_index(int32_t i, int32_t j) const	// no so const...
	{
		CMath::swap(x[i],x[j]);
		if(x_square) CMath::swap(x_square[i],x_square[j]);
	}

	static void* compute_Q_parallel_helper(void* p)
	{
		Q_THREAD_PARAM* params= (Q_THREAD_PARAM*) p;
		int32_t i=params->i;
		int32_t start=params->start;
		int32_t end=params->end;
		float64_t* y=params->y;
		Qfloat* data=params->data;
		const LibSVMKernel* q=params->q;

		if (y) // two class
		{
			for(int32_t j=start;j<end;j++)
				data[j] = (Qfloat) y[i]*y[j]*q->kernel_function(i,j);
		}
		else // one class, eps svr
		{
			for(int32_t j=start;j<end;j++)
				data[j] = (Qfloat) q->kernel_function(i,j);
		}

		return NULL;
	}

	void compute_Q_parallel(Qfloat* data, float64_t* lab, int32_t i, int32_t start, int32_t len) const
	{
		int32_t num_threads=sg_parallel->get_num_threads();
		if (num_threads < 2)
		{
			Q_THREAD_PARAM params;
			params.i=i;
			params.start=start;
			params.end=len;
			params.y=lab;
			params.data=data;
			params.q=this;
			compute_Q_parallel_helper((void*) &params);
		}
		else
		{
#ifdef HAVE_PTHREAD
			int32_t total_num=(len-start);
			pthread_t* threads = SG_MALLOC(pthread_t, num_threads-1);
			Q_THREAD_PARAM* params = SG_MALLOC(Q_THREAD_PARAM, num_threads);
			int32_t step= total_num/num_threads;

			int32_t t;

			num_threads--;
			for (t=0; t<num_threads; t++)
			{
				params[t].i=i;
				params[t].start=t*step;
				params[t].end=(t+1)*step;
				params[t].y=lab;
				params[t].data=data;
				params[t].q=this;

				int code=pthread_create(&threads[t], NULL,
						compute_Q_parallel_helper, (void*)&params[t]);

				if (code != 0)
				{
					SG_SWARNING("Thread creation failed (thread %d of %d) "
							"with error:'%s'\n",t, num_threads, strerror(code));
					num_threads=t;
					break;
				}
			}

			params[t].i=i;
			params[t].start=t*step;
			params[t].end=len;
			params[t].y=lab;
			params[t].data=data;
			params[t].q=this;
			compute_Q_parallel_helper(&params[t]);

			for (t=0; t<num_threads; t++)
			{
				if (pthread_join(threads[t], NULL) != 0)
					SG_SWARNING("pthread_join of thread %d/%d failed\n", t, num_threads)
			}

			SG_FREE(params);
			SG_FREE(threads);
#endif /* HAVE_PTHREAD */
		}
	}

	inline float64_t kernel_function(int32_t i, int32_t j) const
	{
		return kernel->kernel(x[i]->index,x[j]->index);
	}

private:
	CKernel* kernel;
	const svm_node **x;
	float64_t *x_square;
};

LibSVMKernel::LibSVMKernel(int32_t l, svm_node * const * x_, const svm_parameter& param)
{
	clone(x,x_,l);
	x_square = 0;
	kernel=param.kernel;
	max_train_time=param.max_train_time;
}

LibSVMKernel::~LibSVMKernel()
{
	SG_FREE(x);
	SG_FREE(x_square);
}

// Generalized SMO+SVMlight algorithm
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + b^T \alpha
//
//		y^T \alpha = \delta
//		y_i = +1 or -1
//		0 <= alpha_i <= Cp for y_i = 1
//		0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//	Q, p, y, Cp, Cn, and an initial feasible point \alpha
//	l is the size of vectors and matrices
//	eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//
class Solver {
public:
	Solver() {};
	virtual ~Solver() {};

	struct SolutionInfo {
		float64_t obj;
		float64_t rho;
		float64_t upper_bound_p;
		float64_t upper_bound_n;
		float64_t r;	// for Solver_NU
	};

	void Solve(
		int32_t l, const QMatrix& Q, const float64_t *p_, const schar *y_,
		float64_t *alpha_, float64_t Cp, float64_t Cn, float64_t eps,
		SolutionInfo* si, int32_t shrinking, bool use_bias);

protected:
	int32_t active_size;
	schar *y;
	float64_t *G;		// gradient of objective function
	enum { LOWER_BOUND, UPPER_BOUND, FREE };
	char *alpha_status;	// LOWER_BOUND, UPPER_BOUND, FREE
	float64_t *alpha;
	const QMatrix *Q;
	const Qfloat *QD;
	float64_t eps;
	float64_t Cp,Cn;
	float64_t *p;
	int32_t *active_set;
	float64_t *G_bar;		// gradient, if we treat free variables as 0
	int32_t l;
	bool unshrink;	// XXX

	float64_t get_C(int32_t i)
	{
		return (y[i] > 0)? Cp : Cn;
	}
	void update_alpha_status(int32_t i)
	{
		if(alpha[i] >= get_C(i))
			alpha_status[i] = UPPER_BOUND;
		else if(alpha[i] <= 0)
			alpha_status[i] = LOWER_BOUND;
		else alpha_status[i] = FREE;
	}
	bool is_upper_bound(int32_t i) { return alpha_status[i] == UPPER_BOUND; }
	bool is_lower_bound(int32_t i) { return alpha_status[i] == LOWER_BOUND; }
	bool is_free(int32_t i) { return alpha_status[i] == FREE; }
	void swap_index(int32_t i, int32_t j);
	void reconstruct_gradient();
	virtual int32_t select_working_set(int32_t &i, int32_t &j, float64_t &gap);
	virtual float64_t calculate_rho();
	virtual void do_shrinking();
private:
	bool be_shrunk(int32_t i, float64_t Gmax1, float64_t Gmax2);
};

void Solver::swap_index(int32_t i, int32_t j)
{
	Q->swap_index(i,j);
	CMath::swap(y[i],y[j]);
	CMath::swap(G[i],G[j]);
	CMath::swap(alpha_status[i],alpha_status[j]);
	CMath::swap(alpha[i],alpha[j]);
	CMath::swap(p[i],p[j]);
	CMath::swap(active_set[i],active_set[j]);
	CMath::swap(G_bar[i],G_bar[j]);
}

void Solver::reconstruct_gradient()
{
	// reconstruct inactive elements of G from G_bar and free variables

	if(active_size == l) return;

	int32_t i,j;
	int32_t nr_free = 0;

	for(j=active_size;j<l;j++)
		G[j] = G_bar[j] + p[j];

	for(j=0;j<active_size;j++)
		if(is_free(j))
			nr_free++;

	if (nr_free*l > 2*active_size*(l-active_size))
	{
		for(i=active_size;i<l;i++)
		{
			const Qfloat *Q_i = Q->get_Q(i,active_size);
			for(j=0;j<active_size;j++)
				if(is_free(j))
					G[i] += alpha[j] * Q_i[j];
		}
	}
	else
	{
		for(i=0;i<active_size;i++)
			if(is_free(i))
			{
				const Qfloat *Q_i = Q->get_Q(i,l);
				float64_t alpha_i = alpha[i];
				for(j=active_size;j<l;j++)
					G[j] += alpha_i * Q_i[j];
			}
	}
}

void Solver::Solve(
	int32_t p_l, const QMatrix& p_Q, const float64_t *p_p,
	const schar *p_y, float64_t *p_alpha, float64_t p_Cp, float64_t p_Cn,
	float64_t p_eps, SolutionInfo* p_si, int32_t shrinking, bool use_bias)
{
	this->l = p_l;
	this->Q = &p_Q;
	QD=Q->get_QD();
	clone(p, p_p,l);
	clone(y, p_y,l);
	clone(alpha,p_alpha,l);
	this->Cp = p_Cp;
	this->Cn = p_Cn;
	this->eps = p_eps;
	unshrink = false;

	// initialize alpha_status
	{
		alpha_status = SG_MALLOC(char, l);
		for(int32_t i=0;i<l;i++)
			update_alpha_status(i);
	}

	// initialize active set (for shrinking)
	{
		active_set = SG_MALLOC(int32_t, l);
		for(int32_t i=0;i<l;i++)
			active_set[i] = i;
		active_size = l;
	}

	// initialize gradient
	CSignal::clear_cancel();
	CTime start_time;
	{
		G = SG_MALLOC(float64_t, l);
		G_bar = SG_MALLOC(float64_t, l);
		int32_t i;
		for(i=0;i<l;i++)
		{
			G[i] = p_p[i];
			G_bar[i] = 0;
		}
		SG_SINFO("Computing gradient for initial set of non-zero alphas\n")
		//CMath::display_vector(alpha, l, "alphas");
		for(i=0;i<l && !CSignal::cancel_computations(); i++)
		{
			if(!is_lower_bound(i))
			{
				const Qfloat *Q_i = Q->get_Q(i,l);
				float64_t alpha_i = alpha[i];
				int32_t j;
				for(j=0;j<l;j++)
					G[j] += alpha_i*Q_i[j];
				if(is_upper_bound(i))
					for(j=0;j<l;j++)
						G_bar[j] += get_C(i) * Q_i[j];
			}
			SG_SPROGRESS(i, 0, l)
		}
		SG_SDONE()
	}

	// optimization step

	int32_t iter = 0;
	int32_t counter = CMath::min(l,1000)+1;

	while (!CSignal::cancel_computations())
	{
		if (Q->max_train_time > 0 && start_time.cur_time_diff() > Q->max_train_time)
		  break;

		// show progress and do shrinking

		if(--counter == 0)
		{
			counter = CMath::min(l,1000);
			if(shrinking) do_shrinking();
			//SG_SINFO(".")
		}

		int32_t i,j;
		float64_t gap;
		if(select_working_set(i,j, gap)!=0)
		{
			// reconstruct the whole gradient
			reconstruct_gradient();
			// reset active set size and check
			active_size = l;
			//SG_SINFO("*")
			if(select_working_set(i,j, gap)!=0)
				break;
			else
				counter = 1;	// do shrinking next iteration
		}

		SG_SABS_PROGRESS(gap, -CMath::log10(gap), -CMath::log10(1), -CMath::log10(eps), 6)

		++iter;

		// update alpha[i] and alpha[j], handle bounds carefully

		const Qfloat *Q_i = Q->get_Q(i,active_size);
		const Qfloat *Q_j = Q->get_Q(j,active_size);

		float64_t C_i = get_C(i);
		float64_t C_j = get_C(j);

		float64_t old_alpha_i = alpha[i];
		float64_t old_alpha_j = alpha[j];

		if (!use_bias)
		{
			double pi=G[i]-Q_i[i]*alpha[i]-Q_i[j]*alpha[j];
			double pj=G[j]-Q_i[j]*alpha[i]-Q_j[j]*alpha[j];
			double det=Q_i[i]*Q_j[j]-Q_i[j]*Q_i[j];
			double alpha_i=-(Q_j[j]*pi-Q_i[j]*pj)/det;
			alpha_i=CMath::min(C_i,CMath::max(0.0,alpha_i));
			double alpha_j=-(-Q_i[j]*pi+Q_i[i]*pj)/det;
			alpha_j=CMath::min(C_j,CMath::max(0.0,alpha_j));

			if (alpha_i==0 || alpha_i == C_i)
				alpha_j=CMath::min(C_j,CMath::max(0.0,-(pj+Q_i[j]*alpha_i)/Q_j[j]));
			if (alpha_j==0 || alpha_j == C_j)
				alpha_i=CMath::min(C_i,CMath::max(0.0,-(pi+Q_i[j]*alpha_j)/Q_i[i]));

			alpha[i]=alpha_i; alpha[j]=alpha_j;
		}
		else
		{
			if(y[i]!=y[j])
			{
				float64_t quad_coef = Q_i[i]+Q_j[j]+2*Q_i[j];
				if (quad_coef <= 0)
					quad_coef = TAU;
				float64_t delta = (-G[i]-G[j])/quad_coef;
				float64_t diff = alpha[i] - alpha[j];
				alpha[i] += delta;
				alpha[j] += delta;

				if(diff > 0)
				{
					if(alpha[j] < 0)
					{
						alpha[j] = 0;
						alpha[i] = diff;
					}
				}
				else
				{
					if(alpha[i] < 0)
					{
						alpha[i] = 0;
						alpha[j] = -diff;
					}
				}
				if(diff > C_i - C_j)
				{
					if(alpha[i] > C_i)
					{
						alpha[i] = C_i;
						alpha[j] = C_i - diff;
					}
				}
				else
				{
					if(alpha[j] > C_j)
					{
						alpha[j] = C_j;
						alpha[i] = C_j + diff;
					}
				}
			}
			else
			{
				float64_t quad_coef = Q_i[i]+Q_j[j]-2*Q_i[j];
				if (quad_coef <= 0)
					quad_coef = TAU;
				float64_t delta = (G[i]-G[j])/quad_coef;
				float64_t sum = alpha[i] + alpha[j];
				alpha[i] -= delta;
				alpha[j] += delta;

				if(sum > C_i)
				{
					if(alpha[i] > C_i)
					{
						alpha[i] = C_i;
						alpha[j] = sum - C_i;
					}
				}
				else
				{
					if(alpha[j] < 0)
					{
						alpha[j] = 0;
						alpha[i] = sum;
					}
				}
				if(sum > C_j)
				{
					if(alpha[j] > C_j)
					{
						alpha[j] = C_j;
						alpha[i] = sum - C_j;
					}
				}
				else
				{
					if(alpha[i] < 0)
					{
						alpha[i] = 0;
						alpha[j] = sum;
					}
				}
			}
		}

		// update G

		float64_t delta_alpha_i = alpha[i] - old_alpha_i;
		float64_t delta_alpha_j = alpha[j] - old_alpha_j;

		for(int32_t k=0;k<active_size;k++)
		{
			G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
		}

		// update alpha_status and G_bar

		{
			bool ui = is_upper_bound(i);
			bool uj = is_upper_bound(j);
			update_alpha_status(i);
			update_alpha_status(j);
			int32_t k;
			if(ui != is_upper_bound(i))
			{
				Q_i = Q->get_Q(i,l);
				if(ui)
					for(k=0;k<l;k++)
						G_bar[k] -= C_i * Q_i[k];
				else
					for(k=0;k<l;k++)
						G_bar[k] += C_i * Q_i[k];
			}

			if(uj != is_upper_bound(j))
			{
				Q_j = Q->get_Q(j,l);
				if(uj)
					for(k=0;k<l;k++)
						G_bar[k] -= C_j * Q_j[k];
				else
					for(k=0;k<l;k++)
						G_bar[k] += C_j * Q_j[k];
			}
		}

#ifdef MCSVM_DEBUG
		// calculate objective value
		{
			float64_t v = 0;
			for(i=0;i<l;i++)
				v += alpha[i] * (G[i] + p[i]);

			p_si->obj = v/2;

			float64_t primal=0;
			//float64_t gap=100000;
			SG_SPRINT("dual obj=%f primal obf=%f gap=%f\n", v/2, primal, gap)
		}
#endif
	}

	// calculate rho

	if (!use_bias)
		p_si->rho = 0;
	else
		p_si->rho = calculate_rho();

	// calculate objective value
	{
		float64_t v = 0;
		int32_t i;
		for(i=0;i<l;i++)
			v += alpha[i] * (G[i] + p[i]);

		p_si->obj = v/2;
	}

	// put back the solution
	{
		for(int32_t i=0;i<l;i++)
			p_alpha[active_set[i]] = alpha[i];
	}

	p_si->upper_bound_p = Cp;
	p_si->upper_bound_n = Cn;

	SG_SINFO("\noptimization finished, #iter = %d\n",iter)

	SG_FREE(p);
	SG_FREE(y);
	SG_FREE(alpha);
	SG_FREE(alpha_status);
	SG_FREE(active_set);
	SG_FREE(G);
	SG_FREE(G_bar);
}

// return 1 if already optimal, return 0 otherwise
int32_t Solver::select_working_set(
	int32_t &out_i, int32_t &out_j, float64_t &gap)
{
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficient <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

	float64_t Gmax = -INF;
	float64_t Gmax2 = -INF;
	int32_t Gmax_idx = -1;
	int32_t Gmin_idx = -1;
	float64_t obj_diff_min = INF;

	for(int32_t t=0;t<active_size;t++)
		if(y[t]==+1)
		{
			if(!is_upper_bound(t))
				if(-G[t] >= Gmax)
				{
					Gmax = -G[t];
					Gmax_idx = t;
				}
		}
		else
		{
			if(!is_lower_bound(t))
				if(G[t] >= Gmax)
				{
					Gmax = G[t];
					Gmax_idx = t;
				}
		}

	int32_t i = Gmax_idx;
	const Qfloat *Q_i = NULL;
	if(i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
		Q_i = Q->get_Q(i,active_size);

	for(int32_t j=0;j<active_size;j++)
	{
		if(y[j]==+1)
		{
			if (!is_lower_bound(j))
			{
				float64_t grad_diff=Gmax+G[j];
				if (G[j] >= Gmax2)
					Gmax2 = G[j];
				if (grad_diff > 0)
				{
					float64_t obj_diff;
					float64_t quad_coef=Q_i[i]+QD[j]-2.0*y[i]*Q_i[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
		else
		{
			if (!is_upper_bound(j))
			{
				float64_t grad_diff= Gmax-G[j];
				if (-G[j] >= Gmax2)
					Gmax2 = -G[j];
				if (grad_diff > 0)
				{
					float64_t obj_diff;
					float64_t quad_coef=Q_i[i]+QD[j]+2.0*y[i]*Q_i[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
	}

	gap=Gmax+Gmax2;
	if(gap < eps)
		return 1;

	out_i = Gmax_idx;
	out_j = Gmin_idx;
	return 0;
}

bool Solver::be_shrunk(int32_t i, float64_t Gmax1, float64_t Gmax2)
{
	if(is_upper_bound(i))
	{
		if(y[i]==+1)
			return(-G[i] > Gmax1);
		else
			return(-G[i] > Gmax2);
	}
	else if(is_lower_bound(i))
	{
		if(y[i]==+1)
			return(G[i] > Gmax2);
		else
			return(G[i] > Gmax1);
	}
	else
		return(false);
}


void Solver::do_shrinking()
{
	int32_t i;
	float64_t Gmax1 = -INF;		// max { -y_i * grad(f)_i | i in I_up(\alpha) }
	float64_t Gmax2 = -INF;		// max { y_i * grad(f)_i | i in I_low(\alpha) }

	// find maximal violating pair first
	for(i=0;i<active_size;i++)
	{
		if(y[i]==+1)
		{
			if(!is_upper_bound(i))
			{
				if(-G[i] >= Gmax1)
					Gmax1 = -G[i];
			}
			if(!is_lower_bound(i))
			{
				if(G[i] >= Gmax2)
					Gmax2 = G[i];
			}
		}
		else
		{
			if(!is_upper_bound(i))
			{
				if(-G[i] >= Gmax2)
					Gmax2 = -G[i];
			}
			if(!is_lower_bound(i))
			{
				if(G[i] >= Gmax1)
					Gmax1 = G[i];
			}
		}
	}

	if(unshrink == false && Gmax1 + Gmax2 <= eps*10)
	{
		unshrink = true;
		reconstruct_gradient();
		active_size = l;
	}

	for(i=0;i<active_size;i++)
		if (be_shrunk(i, Gmax1, Gmax2))
		{
			active_size--;
			while (active_size > i)
			{
				if (!be_shrunk(active_size, Gmax1, Gmax2))
				{
					swap_index(i,active_size);
					break;
				}
				active_size--;
			}
		}
}

float64_t Solver::calculate_rho()
{
	float64_t r;
	int32_t nr_free = 0;
	float64_t ub = INF, lb = -INF, sum_free = 0;
	for(int32_t i=0;i<active_size;i++)
	{
		float64_t yG = y[i]*G[i];

		if(is_upper_bound(i))
		{
			if(y[i]==-1)
				ub = CMath::min(ub,yG);
			else
				lb = CMath::max(lb,yG);
		}
		else if(is_lower_bound(i))
		{
			if(y[i]==+1)
				ub = CMath::min(ub,yG);
			else
				lb = CMath::max(lb,yG);
		}
		else
		{
			++nr_free;
			sum_free += yG;
		}
	}

	if(nr_free>0)
		r = sum_free/nr_free;
	else
		r = (ub+lb)/2;

	return r;
}


//
//Solve with individually weighted examples
//
class WeightedSolver : public Solver
{

public:

	WeightedSolver(float64_t* cost_vec)
	{

		this->Cs = cost_vec;

	}

	virtual float64_t get_C(int32_t i)
	{

		return Cs[i];
	}

protected:

  float64_t* Cs;

};


//
// Solver for nu-svm classification and regression
//
// additional constraint: e^T \alpha = constant
//
class Solver_NU : public Solver
{
public:
	Solver_NU() {}
	void Solve(
		int32_t p_l, const QMatrix& p_Q, const float64_t *p_p,
		const schar *p_y, float64_t* p_alpha, float64_t p_Cp, float64_t p_Cn,
		float64_t p_eps, SolutionInfo* p_si, int32_t shrinking, bool use_bias)
	{
		this->si = p_si;
		Solver::Solve(p_l,p_Q,p_p,p_y,p_alpha,p_Cp,p_Cn,p_eps,p_si,
				shrinking,use_bias);
	}
private:
	SolutionInfo *si;
	int32_t select_working_set(int32_t &i, int32_t &j, float64_t &gap);
	float64_t calculate_rho();
	bool be_shrunk(
		int32_t i, float64_t Gmax1, float64_t Gmax2, float64_t Gmax3,
		float64_t Gmax4);
	void do_shrinking();
};

// return 1 if already optimal, return 0 otherwise
int32_t Solver_NU::select_working_set(
	int32_t &out_i, int32_t &out_j, float64_t &gap)
{
	// return i,j such that y_i = y_j and
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficient <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

	float64_t Gmaxp = -INF;
	float64_t Gmaxp2 = -INF;
	int32_t Gmaxp_idx = -1;

	float64_t Gmaxn = -INF;
	float64_t Gmaxn2 = -INF;
	int32_t Gmaxn_idx = -1;

	int32_t Gmin_idx = -1;
	float64_t obj_diff_min = INF;

	for(int32_t t=0;t<active_size;t++)
		if(y[t]==+1)
		{
			if(!is_upper_bound(t))
				if(-G[t] >= Gmaxp)
				{
					Gmaxp = -G[t];
					Gmaxp_idx = t;
				}
		}
		else
		{
			if(!is_lower_bound(t))
				if(G[t] >= Gmaxn)
				{
					Gmaxn = G[t];
					Gmaxn_idx = t;
				}
		}

	int32_t ip = Gmaxp_idx;
	int32_t in = Gmaxn_idx;
	const Qfloat *Q_ip = NULL;
	const Qfloat *Q_in = NULL;
	if(ip != -1) // NULL Q_ip not accessed: Gmaxp=-INF if ip=-1
		Q_ip = Q->get_Q(ip,active_size);
	if(in != -1)
		Q_in = Q->get_Q(in,active_size);

	for(int32_t j=0;j<active_size;j++)
	{
		if(y[j]==+1)
		{
			if (!is_lower_bound(j))
			{
				float64_t grad_diff=Gmaxp+G[j];
				if (G[j] >= Gmaxp2)
					Gmaxp2 = G[j];
				if (grad_diff > 0)
				{
					float64_t obj_diff;
					float64_t quad_coef = Q_ip[ip]+QD[j]-2*Q_ip[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
		else
		{
			if (!is_upper_bound(j))
			{
				float64_t grad_diff=Gmaxn-G[j];
				if (-G[j] >= Gmaxn2)
					Gmaxn2 = -G[j];
				if (grad_diff > 0)
				{
					float64_t obj_diff;
					float64_t quad_coef = Q_in[in]+QD[j]-2*Q_in[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
	}

	gap=CMath::max(Gmaxp+Gmaxp2,Gmaxn+Gmaxn2);
	if(gap < eps)
		return 1;

	if (y[Gmin_idx] == +1)
		out_i = Gmaxp_idx;
	else
		out_i = Gmaxn_idx;
	out_j = Gmin_idx;

	return 0;
}

bool Solver_NU::be_shrunk(
	int32_t i, float64_t Gmax1, float64_t Gmax2, float64_t Gmax3,
	float64_t Gmax4)
{
	if(is_upper_bound(i))
	{
		if(y[i]==+1)
			return(-G[i] > Gmax1);
		else
			return(-G[i] > Gmax4);
	}
	else if(is_lower_bound(i))
	{
		if(y[i]==+1)
			return(G[i] > Gmax2);
		else
			return(G[i] > Gmax3);
	}
	else
		return(false);
}

void Solver_NU::do_shrinking()
{
	float64_t Gmax1 = -INF;	// max { -y_i * grad(f)_i | y_i = +1, i in I_up(\alpha) }
	float64_t Gmax2 = -INF;	// max { y_i * grad(f)_i | y_i = +1, i in I_low(\alpha) }
	float64_t Gmax3 = -INF;	// max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
	float64_t Gmax4 = -INF;	// max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

	// find maximal violating pair first
	int32_t i;
	for(i=0;i<active_size;i++)
	{
		if(!is_upper_bound(i))
		{
			if(y[i]==+1)
			{
				if(-G[i] > Gmax1) Gmax1 = -G[i];
			}
			else	if(-G[i] > Gmax4) Gmax4 = -G[i];
		}
		if(!is_lower_bound(i))
		{
			if(y[i]==+1)
			{
				if(G[i] > Gmax2) Gmax2 = G[i];
			}
			else	if(G[i] > Gmax3) Gmax3 = G[i];
		}
	}

	if(unshrink == false && CMath::max(Gmax1+Gmax2,Gmax3+Gmax4) <= eps*10)
	{
		unshrink = true;
		reconstruct_gradient();
		active_size = l;
	}

	for(i=0;i<active_size;i++)
		if (be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4))
		{
			active_size--;
			while (active_size > i)
			{
				if (!be_shrunk(active_size, Gmax1, Gmax2, Gmax3, Gmax4))
				{
					swap_index(i,active_size);
					break;
				}
				active_size--;
			}
		}
}

float64_t Solver_NU::calculate_rho()
{
	int32_t nr_free1 = 0,nr_free2 = 0;
	float64_t ub1 = INF, ub2 = INF;
	float64_t lb1 = -INF, lb2 = -INF;
	float64_t sum_free1 = 0, sum_free2 = 0;

	for(int32_t i=0; i<active_size; i++)
	{
		if(y[i]==+1)
		{
			if(is_upper_bound(i))
				lb1 = CMath::max(lb1,G[i]);
			else if(is_lower_bound(i))
				ub1 = CMath::min(ub1,G[i]);
			else
			{
				++nr_free1;
				sum_free1 += G[i];
			}
		}
		else
		{
			if(is_upper_bound(i))
				lb2 = CMath::max(lb2,G[i]);
			else if(is_lower_bound(i))
				ub2 = CMath::min(ub2,G[i]);
			else
			{
				++nr_free2;
				sum_free2 += G[i];
			}
		}
	}

	float64_t r1,r2;
	if(nr_free1 > 0)
		r1 = sum_free1/nr_free1;
	else
		r1 = (ub1+lb1)/2;

	if(nr_free2 > 0)
		r2 = sum_free2/nr_free2;
	else
		r2 = (ub2+lb2)/2;

	si->r = (r1+r2)/2;
	return (r1-r2)/2;
}

class SVC_QMC: public LibSVMKernel
{
public:
	SVC_QMC(const svm_problem& prob, const svm_parameter& param, const schar *y_, int32_t n_class, float64_t fac)
	:LibSVMKernel(prob.l, prob.x, param)
	{
		nr_class=n_class;
		factor=fac;
		clone(y,y_,prob.l);
		cache = new Cache(prob.l,(int64_t)(param.cache_size*(1l<<20)));
		QD = SG_MALLOC(Qfloat, prob.l);
		for(int32_t i=0;i<prob.l;i++)
		{
			QD[i]= factor*(nr_class-1)*kernel_function(i,i);
		}
	}

	Qfloat *get_Q(int32_t i, int32_t len) const
	{
		Qfloat *data;
		int32_t start;
		if((start = cache->get_data(i,&data,len)) < len)
		{
			compute_Q_parallel(data, NULL, i, start, len);

			for(int32_t j=start;j<len;j++)
			{
				if (y[i]==y[j])
					data[j] *= (factor*(nr_class-1));
				else
					data[j] *= (-factor);
			}
		}
		return data;
	}

	inline Qfloat get_orig_Qij(Qfloat Q, int32_t i, int32_t j)
	{
		if (y[i]==y[j])
			return Q/(nr_class-1);
		else
			return -Q;
	}

	Qfloat *get_QD() const
	{
		return QD;
	}

	void swap_index(int32_t i, int32_t j) const
	{
		cache->swap_index(i,j);
		LibSVMKernel::swap_index(i,j);
		CMath::swap(y[i],y[j]);
		CMath::swap(QD[i],QD[j]);
	}

	~SVC_QMC()
	{
		SG_FREE(y);
		delete cache;
		SG_FREE(QD);
	}
private:
	float64_t factor;
	float64_t nr_class;
	schar *y;
	Cache *cache;
	Qfloat *QD;
};

//
// Solver for nu-svm classification and regression
//
// additional constraint: e^T \alpha = constant
//
class Solver_NUMC : public Solver
{
public:
	Solver_NUMC(int32_t n_class, float64_t svm_nu)
	{
		nr_class=n_class;
		nu=svm_nu;
	}

	void Solve(
		int32_t p_l, const QMatrix& p_Q, const float64_t *p_p,
		const schar *p_y, float64_t* p_alpha, float64_t p_Cp, float64_t p_Cn,
		float64_t p_eps, SolutionInfo* p_si, int32_t shrinking, bool use_bias)
	{
		this->si = p_si;
		Solver::Solve(p_l,p_Q,p_p,p_y,p_alpha,p_Cp,p_Cn,p_eps,p_si,shrinking, use_bias);
	}
	float64_t compute_primal(const schar* p_y, float64_t* p_alpha, float64_t* biases,float64_t* normwcw);

private:
	SolutionInfo *si;
	int32_t select_working_set(int32_t &i, int32_t &j, float64_t &gap);
	float64_t calculate_rho();
	bool be_shrunk(
		int32_t i, float64_t Gmax1, float64_t Gmax2, float64_t Gmax3,
		float64_t Gmax4);
	void do_shrinking();

private:
	int32_t nr_class;
	float64_t  nu;
};

float64_t Solver_NUMC::compute_primal(const schar* p_y, float64_t* p_alpha, float64_t* biases, float64_t* normwcw)
{
	clone(y, p_y,l);
	clone(alpha,p_alpha,l);

	alpha_status = SG_MALLOC(char, l);
	for(int32_t i=0;i<l;i++)
		update_alpha_status(i);

	float64_t* class_count = SG_MALLOC(float64_t, nr_class);
	float64_t* outputs = SG_MALLOC(float64_t, l);

	for (int32_t i=0; i<nr_class; i++)
	{
		class_count[i]=0;
		biases[i+1]=0;
	}

	for (int32_t i=0; i<active_size; i++)
	{
		update_alpha_status(i);
		if(!is_upper_bound(i) && !is_lower_bound(i))
			class_count[(int32_t) y[i]]++;
	}

	//CMath::display_vector(class_count, nr_class, "class_count");

	float64_t mu=((float64_t) nr_class)/(nu*l);
	//SG_SPRINT("nr_class=%d, l=%d, active_size=%d, nu=%f, mu=%f\n", nr_class, l, active_size, nu, mu)

	float64_t rho=0;
	float64_t quad=0;
	float64_t* zero_counts  = SG_MALLOC(float64_t, nr_class);
	float64_t normwc_const = 0;

	for (int32_t i=0; i<nr_class; i++)
	{
		zero_counts[i]=-INF;
		normwcw[i]=0;
	}

	for (int32_t i=0; i<active_size; i++)
	{
		float64_t sum_free=0;
		float64_t sum_atbound=0;
		float64_t sum_zero_count=0;

		Qfloat* Q_i = Q->get_Q(i,active_size);
		outputs[i]=0;

		for (int j=0; j<active_size; j++)
		{
			quad+= alpha[i]*alpha[j]*Q_i[j];
			float64_t tmp= alpha[j]*Q_i[j]/mu;

			if(!is_upper_bound(i) && !is_lower_bound(i))
				sum_free+=tmp;
			else
				sum_atbound+=tmp;

			if (class_count[(int32_t) y[i]] == 0 && y[j]==y[i])
				sum_zero_count+= tmp;

			SVC_QMC* QMC=(SVC_QMC*) Q;
			float64_t norm_tmp=alpha[i]*alpha[j]*QMC->get_orig_Qij(Q_i[j], i, j);
			if (y[i]==y[j])
				normwcw[(int32_t) y[i]]+=norm_tmp;

			normwcw[(int32_t) y[i]]-=2.0/nr_class*norm_tmp;
			normwc_const+=norm_tmp;
		}

		if (class_count[(int32_t) y[i]] == 0)
		{
			if (zero_counts[(int32_t) y[i]]<sum_zero_count)
				zero_counts[(int32_t) y[i]]=sum_zero_count;
		}

		biases[(int32_t) y[i]+1]-=sum_free;
		if (class_count[(int32_t) y[i]] != 0.0)
			rho+=sum_free/class_count[(int32_t) y[i]];
		outputs[i]+=sum_free+sum_atbound;
	}

	for (int32_t i=0; i<nr_class; i++)
	{
		if (class_count[i] == 0.0)
			rho+=zero_counts[i];

		normwcw[i]+=normwc_const/CMath::sq(nr_class);
		normwcw[i]=CMath::sqrt(normwcw[i]);
	}

	SGVector<float64_t>::display_vector(normwcw, nr_class, "normwcw");

	rho/=nr_class;

	SG_SPRINT("rho=%f\n", rho)

	float64_t sumb=0;
	for (int32_t i=0; i<nr_class; i++)
	{
		if (class_count[i] != 0.0)
			biases[i+1]=biases[i+1]/class_count[i]+rho;
		else
			biases[i+1]+=rho-zero_counts[i];

		SG_SPRINT("biases=%f\n", biases[i+1])

		sumb+=biases[i+1];
	}
	SG_SPRINT("sumb=%f\n", sumb)

	SG_FREE(zero_counts);

	for (int32_t i=0; i<l; i++)
		outputs[i]+=biases[(int32_t) y[i]+1];

	biases[0]=rho;

	//CMath::display_vector(outputs, l, "outputs");


	float64_t xi=0;
	for (int32_t i=0; i<active_size; i++)
	{
		if (is_lower_bound(i))
			continue;
		xi+=rho-outputs[i];
	}

	//SG_SPRINT("xi=%f\n", xi)

	//SG_SPRINT("quad=%f Cp=%f xi*mu=%f\n", quad, nr_class*rho, xi*mu)

	float64_t primal=0.5*quad- nr_class*rho+xi*mu;

	//SG_SPRINT("primal=%10.10f\n", primal)

	SG_FREE(y);
	SG_FREE(alpha);
	SG_FREE(alpha_status);

	return primal;
}


// return 1 if already optimal, return 0 otherwise
int32_t Solver_NUMC::select_working_set(
	int32_t &out_i, int32_t &out_j, float64_t &gap)
{
	// return i,j such that y_i = y_j and
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficient <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

	int32_t retval=0;
	float64_t best_gap=0;
	int32_t best_out_i=-1;
	int32_t best_out_j=-1;

	float64_t* Gmaxp = SG_MALLOC(float64_t, nr_class);
	float64_t* Gmaxp2 = SG_MALLOC(float64_t, nr_class);
	int32_t* Gmaxp_idx = SG_MALLOC(int32_t, nr_class);

	int32_t* Gmin_idx = SG_MALLOC(int32_t, nr_class);
	float64_t* obj_diff_min = SG_MALLOC(float64_t, nr_class);

	for (int32_t i=0; i<nr_class; i++)
	{
		Gmaxp[i]=-INF;
		Gmaxp2[i]=-INF;
		Gmaxp_idx[i]=-1;
		Gmin_idx[i]=-1;
		obj_diff_min[i]=INF;
	}

	for(int32_t t=0;t<active_size;t++)
	{
		int32_t cidx=y[t];
		if(!is_upper_bound(t))
		{
			if(-G[t] >= Gmaxp[cidx])
			{
				Gmaxp[cidx] = -G[t];
				Gmaxp_idx[cidx] = t;
			}
		}
	}

	for(int32_t j=0;j<active_size;j++)
	{
		int32_t cidx=y[j];
		int32_t ip = Gmaxp_idx[cidx];
		const Qfloat *Q_ip = NULL;
		if(ip != -1) // NULL Q_ip not accessed: Gmaxp=-INF if ip=-1
			Q_ip = Q->get_Q(ip,active_size);

		if (!is_lower_bound(j))
		{
			float64_t grad_diff=Gmaxp[cidx]+G[j];
			if (G[j] >= Gmaxp2[cidx])
				Gmaxp2[cidx] = G[j];
			if (grad_diff > 0)
			{
				float64_t obj_diff;
				float64_t quad_coef = Q_ip[ip]+QD[j]-2*Q_ip[j];
				if (quad_coef > 0)
					obj_diff = -(grad_diff*grad_diff)/quad_coef;
				else
					obj_diff = -(grad_diff*grad_diff)/TAU;

				if (obj_diff <= obj_diff_min[cidx])
				{
					Gmin_idx[cidx]=j;
					obj_diff_min[cidx] = obj_diff;
				}
			}
		}

		gap=Gmaxp[cidx]+Gmaxp2[cidx];
		if (gap>=best_gap && Gmin_idx[cidx]>=0 &&
				Gmaxp_idx[cidx]>=0 && Gmin_idx[cidx]<active_size)
		{
			out_i = Gmaxp_idx[cidx];
			out_j = Gmin_idx[cidx];

			best_gap=gap;
			best_out_i=out_i;
			best_out_j=out_j;
		}
	}

	gap=best_gap;
	out_i=best_out_i;
	out_j=best_out_j;

	SG_SDEBUG("i=%d j=%d best_gap=%f y_i=%f y_j=%f\n", out_i, out_j, gap, y[out_i], y[out_j])


	if(gap < eps)
		retval=1;

	SG_FREE(Gmaxp);
	SG_FREE(Gmaxp2);
	SG_FREE(Gmaxp_idx);
	SG_FREE(Gmin_idx);
	SG_FREE(obj_diff_min);

	return retval;
}

bool Solver_NUMC::be_shrunk(
	int32_t i, float64_t Gmax1, float64_t Gmax2, float64_t Gmax3,
	float64_t Gmax4)
{
	return false;
}

void Solver_NUMC::do_shrinking()
{
}

float64_t Solver_NUMC::calculate_rho()
{
	return 0;
}


//
// Q matrices for various formulations
//
class SVC_Q: public LibSVMKernel
{
public:
	SVC_Q(const svm_problem& prob, const svm_parameter& param, const schar *y_)
	:LibSVMKernel(prob.l, prob.x, param)
	{
		clone(y,y_,prob.l);
		cache = new Cache(prob.l,(int64_t)(param.cache_size*(1l<<20)));
		QD = SG_MALLOC(Qfloat, prob.l);
		for(int32_t i=0;i<prob.l;i++)
			QD[i]= (Qfloat)kernel_function(i,i);
	}

	Qfloat *get_Q(int32_t i, int32_t len) const
	{
		Qfloat *data;
		int32_t start;
		if((start = cache->get_data(i,&data,len)) < len)
			compute_Q_parallel(data, y, i, start, len);

		return data;
	}

	Qfloat *get_QD() const
	{
		return QD;
	}

	void swap_index(int32_t i, int32_t j) const
	{
		cache->swap_index(i,j);
		LibSVMKernel::swap_index(i,j);
		CMath::swap(y[i],y[j]);
		CMath::swap(QD[i],QD[j]);
	}

	~SVC_Q()
	{
		SG_FREE(y);
		delete cache;
		SG_FREE(QD);
	}
private:
	schar *y;
	Cache *cache;
	Qfloat *QD;
};


class ONE_CLASS_Q: public LibSVMKernel
{
public:
	ONE_CLASS_Q(const svm_problem& prob, const svm_parameter& param)
	:LibSVMKernel(prob.l, prob.x, param)
	{
		cache = new Cache(prob.l,(int64_t)(param.cache_size*(1l<<20)));
		QD = SG_MALLOC(Qfloat, prob.l);
		for(int32_t i=0;i<prob.l;i++)
			QD[i]= (Qfloat)kernel_function(i,i);
	}

	Qfloat *get_Q(int32_t i, int32_t len) const
	{
		Qfloat *data;
		int32_t start;
		if((start = cache->get_data(i,&data,len)) < len)
			compute_Q_parallel(data, NULL, i, start, len);

		return data;
	}

	Qfloat *get_QD() const
	{
		return QD;
	}

	void swap_index(int32_t i, int32_t j) const
	{
		cache->swap_index(i,j);
		LibSVMKernel::swap_index(i,j);
		CMath::swap(QD[i],QD[j]);
	}

	~ONE_CLASS_Q()
	{
		delete cache;
		SG_FREE(QD);
	}
private:
	Cache *cache;
	Qfloat *QD;
};

class SVR_Q: public LibSVMKernel
{
public:
	SVR_Q(const svm_problem& prob, const svm_parameter& param)
	:LibSVMKernel(prob.l, prob.x, param)
	{
		l = prob.l;
		cache = new Cache(l,(int64_t)(param.cache_size*(1l<<20)));
		QD = SG_MALLOC(Qfloat, 2*l);
		sign = SG_MALLOC(schar, 2*l);
		index = SG_MALLOC(int32_t, 2*l);
		for(int32_t k=0;k<l;k++)
		{
			sign[k] = 1;
			sign[k+l] = -1;
			index[k] = k;
			index[k+l] = k;
			QD[k]= (Qfloat)kernel_function(k,k);
			QD[k+l]=QD[k];
		}
		buffer[0] = SG_MALLOC(Qfloat, 2*l);
		buffer[1] = SG_MALLOC(Qfloat, 2*l);
		next_buffer = 0;
	}

	void swap_index(int32_t i, int32_t j) const
	{
		CMath::swap(sign[i],sign[j]);
		CMath::swap(index[i],index[j]);
		CMath::swap(QD[i],QD[j]);
	}

	Qfloat *get_Q(int32_t i, int32_t len) const
	{
		Qfloat *data;
		int32_t real_i = index[i];
		if(cache->get_data(real_i,&data,l) < l)
			compute_Q_parallel(data, NULL, real_i, 0, l);

		// reorder and copy
		Qfloat *buf = buffer[next_buffer];
		next_buffer = 1 - next_buffer;
		schar si = sign[i];
		for(int32_t j=0;j<len;j++)
			buf[j] = si * sign[j] * data[index[j]];
		return buf;
	}

	Qfloat *get_QD() const
	{
		return QD;
	}

	~SVR_Q()
	{
		delete cache;
		SG_FREE(sign);
		SG_FREE(index);
		SG_FREE(buffer[0]);
		SG_FREE(buffer[1]);
		SG_FREE(QD);
	}

private:
	int32_t l;
	Cache *cache;
	schar *sign;
	int32_t *index;
	mutable int32_t next_buffer;
	Qfloat *buffer[2];
	Qfloat *QD;
};

//
// construct and solve various formulations
//
static void solve_c_svc(
	const svm_problem *prob, const svm_parameter* param,
	float64_t *alpha, Solver::SolutionInfo* si, float64_t Cp, float64_t Cn)
{
	int32_t l = prob->l;
	schar *y = SG_MALLOC(schar, l);

	int32_t i;

	for(i=0;i<l;i++)
	{
		alpha[i] = 0;
		if(prob->y[i] > 0) y[i] = +1; else y[i]=-1;
	}

	Solver s;
	s.Solve(l, SVC_Q(*prob,*param,y), prob->pv, y,
		alpha, Cp, Cn, param->eps, si, param->shrinking, param->use_bias);

	float64_t sum_alpha=0;
	for(i=0;i<l;i++)
		sum_alpha += alpha[i];

	if (Cp==Cn)
		SG_SINFO("nu = %f\n", sum_alpha/(param->C*prob->l))

	for(i=0;i<l;i++)
		alpha[i] *= y[i];

	SG_FREE(y);
}


//two weighted datasets
void solve_c_svc_weighted(
	const svm_problem *prob, const svm_parameter* param,
	float64_t *alpha, Solver::SolutionInfo* si, float64_t Cp, float64_t Cn)
{
	int l = prob->l;
	float64_t *minus_ones = SG_MALLOC(float64_t, l);
	schar *y = SG_MALLOC(schar, l);

	int i;

	for(i=0;i<l;i++)
	{
		alpha[i] = 0;
		minus_ones[i] = -1;
		if(prob->y[i] > 0) y[i] = +1; else y[i]=-1;
	}

	WeightedSolver s = WeightedSolver(prob->C);
	s.Solve(l, SVC_Q(*prob,*param,y), minus_ones, y,
		alpha, Cp, Cn, param->eps, si, param->shrinking, param->use_bias);

	float64_t sum_alpha=0;
	for(i=0;i<l;i++)
		sum_alpha += alpha[i];

	//if (Cp==Cn)
	//	SG_SINFO("nu = %f\n", sum_alpha/(prob->C*prob->l))

	for(i=0;i<l;i++)
		alpha[i] *= y[i];

	SG_FREE(minus_ones);
	SG_FREE(y);
}

static void solve_nu_svc(
	const svm_problem *prob, const svm_parameter *param,
	float64_t *alpha, Solver::SolutionInfo* si)
{
	int32_t i;
	int32_t l = prob->l;
	float64_t nu = param->nu;

	schar *y = SG_MALLOC(schar, l);

	for(i=0;i<l;i++)
		if(prob->y[i]>0)
			y[i] = +1;
		else
			y[i] = -1;

	float64_t sum_pos = nu*l/2;
	float64_t sum_neg = nu*l/2;

	for(i=0;i<l;i++)
		if(y[i] == +1)
		{
			alpha[i] = CMath::min(1.0,sum_pos);
			sum_pos -= alpha[i];
		}
		else
		{
			alpha[i] = CMath::min(1.0,sum_neg);
			sum_neg -= alpha[i];
		}

	float64_t *zeros = SG_MALLOC(float64_t, l);

	for(i=0;i<l;i++)
		zeros[i] = 0;

	Solver_NU s;
	s.Solve(l, SVC_Q(*prob,*param,y), zeros, y,
		alpha, 1.0, 1.0, param->eps, si,  param->shrinking, param->use_bias);
	float64_t r = si->r;

	SG_SINFO("C = %f\n",1/r)

	for(i=0;i<l;i++)
		alpha[i] *= y[i]/r;

	si->rho /= r;
	si->obj /= (r*r);
	si->upper_bound_p = 1/r;
	si->upper_bound_n = 1/r;

	SG_FREE(y);
	SG_FREE(zeros);
}

static void solve_nu_multiclass_svc(const svm_problem *prob,
		const svm_parameter *param, Solver::SolutionInfo* si, svm_model* model)
{
	int32_t l = prob->l;
	float64_t nu = param->nu;

	float64_t *alpha = SG_MALLOC(float64_t, prob->l);
	schar *y = SG_MALLOC(schar, l);

	for(int32_t i=0;i<l;i++)
	{
		alpha[i] = 0;
		y[i]=prob->y[i];
	}

	int32_t nr_class=param->nr_class;
	float64_t* sum_class = SG_MALLOC(float64_t, nr_class);

	for (int32_t j=0; j<nr_class; j++)
		sum_class[j] = nu*l/nr_class;

	for(int32_t i=0;i<l;i++)
	{
		alpha[i] = CMath::min(1.0,sum_class[int32_t(y[i])]);
		sum_class[int32_t(y[i])] -= alpha[i];
	}
	SG_FREE(sum_class);


	float64_t *zeros = SG_MALLOC(float64_t, l);

	for (int32_t i=0;i<l;i++)
		zeros[i] = 0;

	Solver_NUMC s(nr_class, nu);
	SVC_QMC Q(*prob,*param,y, nr_class, ((float64_t) nr_class)/CMath::sq(nu*l));

	s.Solve(l, Q, zeros, y,
		alpha, 1.0, 1.0, param->eps, si,  param->shrinking, param->use_bias);


	int32_t* class_sv_count=SG_MALLOC(int32_t, nr_class);

	for (int32_t i=0; i<nr_class; i++)
		class_sv_count[i]=0;

	for (int32_t i=0; i<l; i++)
	{
		if (CMath::abs(alpha[i]) > 0)
			class_sv_count[(int32_t) y[i]]++;
	}

	model->l=l;
	// rho[0]= rho in mcsvm paper, rho[1]...rho[nr_class] is bias in mcsvm paper
	model->rho = SG_MALLOC(float64_t, nr_class+1);
	model->nr_class = nr_class;
	model->label = NULL;
	model->SV = SG_MALLOC(svm_node*,nr_class);
	model->nSV = SG_MALLOC(int32_t, nr_class);
	model->sv_coef = SG_MALLOC(float64_t *,nr_class);
	model->normwcw = SG_MALLOC(float64_t,nr_class);

	for (int32_t i=0; i<nr_class+1; i++)
		model->rho[i]=0;

	model->objective = si->obj;

	if (param->use_bias)
	{
		SG_SDEBUG("Computing biases and primal objective\n")
		float64_t primal = s.compute_primal(y, alpha, model->rho, model->normwcw);
		SG_SINFO("Primal = %10.10f\n", primal)
	}

	for (int32_t i=0; i<nr_class; i++)
	{
		model->nSV[i]=class_sv_count[i];
		model->SV[i] = SG_MALLOC(svm_node,class_sv_count[i]);
		model->sv_coef[i] = SG_MALLOC(float64_t,class_sv_count[i]);
		class_sv_count[i]=0;
	}

	for (int32_t i=0;i<prob->l;i++)
	{
		if(fabs(alpha[i]) > 0)
		{
			model->SV[(int32_t) y[i]][class_sv_count[(int32_t) y[i]]].index = prob->x[i]->index;
			model->sv_coef[(int32_t) y[i]][class_sv_count[(int32_t) y[i]]] = alpha[i];
			class_sv_count[(int32_t) y[i]]++;
		}
	}

	SG_FREE(y);
	SG_FREE(zeros);
	SG_FREE(alpha);
	SG_FREE(class_sv_count);
}

static void solve_one_class(
	const svm_problem *prob, const svm_parameter *param,
	float64_t *alpha, Solver::SolutionInfo* si)
{
	int32_t l = prob->l;
	float64_t *zeros = SG_MALLOC(float64_t, l);
	schar *ones = SG_MALLOC(schar, l);
	int32_t i;

	int32_t n = (int32_t)(param->nu*prob->l);	// # of alpha's at upper bound

	for(i=0;i<n;i++)
		alpha[i] = 1;
	if(n<prob->l)
		alpha[n] = param->nu * prob->l - n;
	for(i=n+1;i<l;i++)
		alpha[i] = 0;

	for(i=0;i<l;i++)
	{
		zeros[i] = 0;
		ones[i] = 1;
	}

	Solver s;
	s.Solve(l, ONE_CLASS_Q(*prob,*param), zeros, ones,
		alpha, 1.0, 1.0, param->eps, si, param->shrinking, param->use_bias);

	SG_FREE(zeros);
	SG_FREE(ones);
}

static void solve_epsilon_svr(
	const svm_problem *prob, const svm_parameter *param,
	float64_t *alpha, Solver::SolutionInfo* si)
{
	int32_t l = prob->l;
	float64_t *alpha2 = SG_MALLOC(float64_t, 2*l);
	float64_t *linear_term = SG_MALLOC(float64_t, 2*l);
	schar *y = SG_MALLOC(schar, 2*l);
	int32_t i;

	for(i=0;i<l;i++)
	{
		alpha2[i] = 0;
		linear_term[i] = param->p - prob->y[i];
		y[i] = 1;

		alpha2[i+l] = 0;
		linear_term[i+l] = param->p + prob->y[i];
		y[i+l] = -1;
	}

	Solver s;
	s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
		alpha2, param->C, param->C, param->eps, si, param->shrinking, param->use_bias);

	float64_t sum_alpha = 0;
	for(i=0;i<l;i++)
	{
		alpha[i] = alpha2[i] - alpha2[i+l];
		sum_alpha += fabs(alpha[i]);
	}
	SG_SINFO("nu = %f\n",sum_alpha/(param->C*l))

	SG_FREE(alpha2);
	SG_FREE(linear_term);
	SG_FREE(y);
}

static void solve_nu_svr(
	const svm_problem *prob, const svm_parameter *param,
	float64_t *alpha, Solver::SolutionInfo* si)
{
	int32_t l = prob->l;
	float64_t C = param->C;
	float64_t *alpha2 = SG_MALLOC(float64_t, 2*l);
	float64_t *linear_term = SG_MALLOC(float64_t, 2*l);
	schar *y = SG_MALLOC(schar, 2*l);
	int32_t i;

	float64_t sum = C * param->nu * l / 2;
	for(i=0;i<l;i++)
	{
		alpha2[i] = alpha2[i+l] = CMath::min(sum,C);
		sum -= alpha2[i];

		linear_term[i] = - prob->y[i];
		y[i] = 1;

		linear_term[i+l] = prob->y[i];
		y[i+l] = -1;
	}

	Solver_NU s;
	s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
		alpha2, C, C, param->eps, si, param->shrinking, param->use_bias);

	SG_SINFO("epsilon = %f\n",-si->r)

	for(i=0;i<l;i++)
		alpha[i] = alpha2[i] - alpha2[i+l];

	SG_FREE(alpha2);
	SG_FREE(linear_term);
	SG_FREE(y);
}

//
// decision_function
//
struct decision_function
{
	float64_t *alpha;
	float64_t rho;
	float64_t objective;
};

decision_function svm_train_one(
	const svm_problem *prob, const svm_parameter *param,
	float64_t Cp, float64_t Cn)
{
	float64_t *alpha = SG_MALLOC(float64_t, prob->l);
	Solver::SolutionInfo si;
	switch(param->svm_type)
	{
		case C_SVC:
			solve_c_svc(prob,param,alpha,&si,Cp,Cn);
			break;
		case NU_SVC:
			solve_nu_svc(prob,param,alpha,&si);
			break;
		case ONE_CLASS:
			solve_one_class(prob,param,alpha,&si);
			break;
		case EPSILON_SVR:
			solve_epsilon_svr(prob,param,alpha,&si);
			break;
		case NU_SVR:
			solve_nu_svr(prob,param,alpha,&si);
			break;
	}

	SG_SINFO("obj = %.16f, rho = %.16f\n",si.obj,si.rho)

	// output SVs
	if (param->svm_type != ONE_CLASS)
	{
		int32_t nSV = 0;
		int32_t nBSV = 0;
		for(int32_t i=0;i<prob->l;i++)
		{
			if(fabs(alpha[i]) > 0)
			{
				++nSV;
				if(prob->y[i] > 0)
				{
					if(fabs(alpha[i]) >= si.upper_bound_p)
						++nBSV;
				}
				else
				{
					if(fabs(alpha[i]) >= si.upper_bound_n)
						++nBSV;
				}
			}
		}
		SG_SINFO("nSV = %d, nBSV = %d\n",nSV,nBSV)
	}

	decision_function f;
	f.alpha = alpha;
	f.rho = si.rho;
	f.objective=si.obj;
	return f;
}

//
// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
void svm_group_classes(
	const svm_problem *prob, int32_t *nr_class_ret, int32_t **label_ret,
	int32_t **start_ret, int32_t **count_ret, int32_t *perm)
{
	int32_t l = prob->l;
	int32_t max_nr_class = 16;
	int32_t nr_class = 0;
	int32_t *label = SG_MALLOC(int32_t, max_nr_class);
	int32_t *count = SG_MALLOC(int32_t, max_nr_class);
	int32_t *data_label = SG_MALLOC(int32_t, l);
	int32_t i;

	for(i=0;i<l;i++)
	{
		int32_t this_label=(int32_t) prob->y[i];
		int32_t j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				int32_t old_max_nr_class=max_nr_class;
				max_nr_class *= 2;
				label=SG_REALLOC(int32_t, label,old_max_nr_class, max_nr_class);
				count=SG_REALLOC(int32_t, count,old_max_nr_class, max_nr_class);
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	int32_t *start = SG_MALLOC(int32_t, nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	SG_FREE(data_label);
}

//
// Interface functions
//
svm_model *svm_train(const svm_problem *prob, const svm_parameter *param)
{
	svm_model *model = SG_MALLOC(svm_model,1);
	model->param = *param;
	model->free_sv = 0;	// XXX

	if(param->svm_type == ONE_CLASS ||
	   param->svm_type == EPSILON_SVR ||
	   param->svm_type == NU_SVR)
	{
		SG_SINFO("training one class svm or doing epsilon sv regression\n")

		// regression or one-class-svm
		model->nr_class = 2;
		model->label = NULL;
		model->nSV = NULL;
		model->sv_coef = SG_MALLOC(float64_t *,1);
		decision_function f = svm_train_one(prob,param,0,0);
		model->rho = SG_MALLOC(float64_t, 1);
		model->rho[0] = f.rho;
		model->objective = f.objective;

		int32_t nSV = 0;
		int32_t i;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i]) > 0) ++nSV;
		model->l = nSV;
		model->SV = SG_MALLOC(svm_node *,nSV);
		model->sv_coef[0] = SG_MALLOC(float64_t, nSV);
		int32_t j = 0;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i]) > 0)
			{
				model->SV[j] = prob->x[i];
				model->sv_coef[0][j] = f.alpha[i];
				++j;
			}

		SG_FREE(f.alpha);
	}
	else if(param->svm_type == NU_MULTICLASS_SVC)
	{
		Solver::SolutionInfo si;
		solve_nu_multiclass_svc(prob,param,&si,model);
		SG_SINFO("obj = %.16f, rho = %.16f\n",si.obj,si.rho)
	}
	else
	{
		// classification
		int32_t l = prob->l;
		int32_t nr_class;
		int32_t *label = NULL;
		int32_t *start = NULL;
		int32_t *count = NULL;
		int32_t *perm = SG_MALLOC(int32_t, l);

		// group training data of the same class
		svm_group_classes(prob,&nr_class,&label,&start,&count,perm);
		svm_node **x = SG_MALLOC(svm_node *,l);
		float64_t *C = SG_MALLOC(float64_t,l);
		float64_t *pv = SG_MALLOC(float64_t,l);


		int32_t i;
		for(i=0;i<l;i++) {
			x[i] = prob->x[perm[i]];
            C[i] = prob->C[perm[i]];

            if (prob->pv)
            {
	pv[i] = prob->pv[perm[i]];
            }
            else
            {
				//no custom linear term is set
	pv[i] = -1.0;
            }

		}


		// calculate weighted C
		float64_t *weighted_C = SG_MALLOC(float64_t, nr_class);
		for(i=0;i<nr_class;i++)
			weighted_C[i] = param->C;
		for(i=0;i<param->nr_weight;i++)
		{
			int32_t j;
			for(j=0;j<nr_class;j++)
				if(param->weight_label[i] == label[j])
					break;
			if(j == nr_class)
				SG_SWARNING("warning: class label %d specified in weight is not found\n", param->weight_label[i])
			else
				weighted_C[j] *= param->weight[i];
		}

		// train k*(k-1)/2 models

		bool *nonzero = SG_MALLOC(bool,l);
		for(i=0;i<l;i++)
			nonzero[i] = false;
		decision_function *f = SG_MALLOC(decision_function,nr_class*(nr_class-1)/2);

		int32_t p = 0;
		for(i=0;i<nr_class;i++)
			for(int32_t j=i+1;j<nr_class;j++)
			{
				svm_problem sub_prob;
				int32_t si = start[i], sj = start[j];
				int32_t ci = count[i], cj = count[j];
				sub_prob.l = ci+cj;
				sub_prob.x = SG_MALLOC(svm_node *,sub_prob.l);
				sub_prob.y = SG_MALLOC(float64_t,sub_prob.l+1); //dirty hack to surpress valgrind err
				sub_prob.C = SG_MALLOC(float64_t,sub_prob.l+1);
				sub_prob.pv = SG_MALLOC(float64_t,sub_prob.l+1);

				int32_t k;
				for(k=0;k<ci;k++)
				{
					sub_prob.x[k] = x[si+k];
					sub_prob.y[k] = +1;
                    sub_prob.C[k] = C[si+k];
                    sub_prob.pv[k] = pv[si+k];

				}
				for(k=0;k<cj;k++)
				{
					sub_prob.x[ci+k] = x[sj+k];
					sub_prob.y[ci+k] = -1;
                    sub_prob.C[ci+k] = C[sj+k];
                    sub_prob.pv[ci+k] = pv[sj+k];
				}
				sub_prob.y[sub_prob.l]=-1; //dirty hack to surpress valgrind err
				sub_prob.C[sub_prob.l]=-1;
				sub_prob.pv[sub_prob.l]=-1;

				f[p] = svm_train_one(&sub_prob,param,weighted_C[i],weighted_C[j]);
				for(k=0;k<ci;k++)
					if(!nonzero[si+k] && fabs(f[p].alpha[k]) > 0)
						nonzero[si+k] = true;
				for(k=0;k<cj;k++)
					if(!nonzero[sj+k] && fabs(f[p].alpha[ci+k]) > 0)
						nonzero[sj+k] = true;
				SG_FREE(sub_prob.x);
				SG_FREE(sub_prob.y);
				SG_FREE(sub_prob.C);
				SG_FREE(sub_prob.pv);
				++p;
			}

		// build output

		model->objective = f[0].objective;
		model->nr_class = nr_class;

		model->label = SG_MALLOC(int32_t, nr_class);
		for(i=0;i<nr_class;i++)
			model->label[i] = label[i];

		model->rho = SG_MALLOC(float64_t, nr_class*(nr_class-1)/2);
		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			model->rho[i] = f[i].rho;

		int32_t total_sv = 0;
		int32_t *nz_count = SG_MALLOC(int32_t, nr_class);
		model->nSV = SG_MALLOC(int32_t, nr_class);
		for(i=0;i<nr_class;i++)
		{
			int32_t nSV = 0;
			for(int32_t j=0;j<count[i];j++)
				if(nonzero[start[i]+j])
				{
					++nSV;
					++total_sv;
				}
			model->nSV[i] = nSV;
			nz_count[i] = nSV;
		}

		SG_SINFO("Total nSV = %d\n",total_sv)

		model->l = total_sv;
		model->SV = SG_MALLOC(svm_node *,total_sv);
		p = 0;
		for(i=0;i<l;i++)
			if(nonzero[i]) model->SV[p++] = x[i];

		int32_t *nz_start = SG_MALLOC(int32_t, nr_class);
		nz_start[0] = 0;
		for(i=1;i<nr_class;i++)
			nz_start[i] = nz_start[i-1]+nz_count[i-1];

		model->sv_coef = SG_MALLOC(float64_t *,nr_class-1);
		for(i=0;i<nr_class-1;i++)
			model->sv_coef[i] = SG_MALLOC(float64_t, total_sv);

		p = 0;
		for(i=0;i<nr_class;i++)
			for(int32_t j=i+1;j<nr_class;j++)
			{
				// classifier (i,j): coefficients with
				// i are in sv_coef[j-1][nz_start[i]...],
				// j are in sv_coef[i][nz_start[j]...]

				int32_t si = start[i];
				int32_t sj = start[j];
				int32_t ci = count[i];
				int32_t cj = count[j];

				int32_t q = nz_start[i];
				int32_t k;
				for(k=0;k<ci;k++)
					if(nonzero[si+k])
						model->sv_coef[j-1][q++] = f[p].alpha[k];
				q = nz_start[j];
				for(k=0;k<cj;k++)
					if(nonzero[sj+k])
						model->sv_coef[i][q++] = f[p].alpha[ci+k];
				++p;
			}

		SG_FREE(label);
		SG_FREE(count);
		SG_FREE(perm);
		SG_FREE(start);
		SG_FREE(x);
		SG_FREE(C);
		SG_FREE(pv);
		SG_FREE(weighted_C);
		SG_FREE(nonzero);
		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			SG_FREE(f[i].alpha);
		SG_FREE(f);
		SG_FREE(nz_count);
		SG_FREE(nz_start);
	}
	return model;
}

void svm_destroy_model(svm_model* model)
{
	if(model->free_sv && model->l > 0)
		SG_FREE((void *)(model->SV[0]));
	for(int32_t i=0;i<model->nr_class-1;i++)
		SG_FREE(model->sv_coef[i]);
	SG_FREE(model->SV);
	SG_FREE(model->sv_coef);
	SG_FREE(model->rho);
	SG_FREE(model->label);
	SG_FREE(model->nSV);
	SG_FREE(model);
}

void svm_destroy_param(svm_parameter* param)
{
	SG_FREE(param->weight_label);
	SG_FREE(param->weight);
}

const char *svm_check_parameter(
	const svm_problem *prob, const svm_parameter *param)
{
	// svm_type

	int32_t svm_type = param->svm_type;
	if(svm_type != C_SVC &&
	   svm_type != NU_SVC &&
	   svm_type != ONE_CLASS &&
	   svm_type != EPSILON_SVR &&
	   svm_type != NU_SVR &&
	   svm_type != NU_MULTICLASS_SVC)
		return "unknown svm type";

	// kernel_type, degree

	int32_t kernel_type = param->kernel_type;
	if(kernel_type != LINEAR &&
	   kernel_type != POLY &&
	   kernel_type != RBF &&
	   kernel_type != SIGMOID &&
	   kernel_type != PRECOMPUTED)
		return "unknown kernel type";

	if(param->degree < 0)
		return "degree of polynomial kernel < 0";

	// cache_size,eps,C,nu,p,shrinking

	if(param->cache_size <= 0)
		return "cache_size <= 0";

	if(param->eps <= 0)
		return "eps <= 0";

	if(svm_type == C_SVC ||
	   svm_type == EPSILON_SVR ||
	   svm_type == NU_SVR)
		if(param->C <= 0)
			return "C <= 0";

	if(svm_type == NU_SVC ||
	   svm_type == ONE_CLASS ||
	   svm_type == NU_SVR)
		if(param->nu <= 0 || param->nu > 1)
			return "nu <= 0 or nu > 1";

	if(svm_type == EPSILON_SVR)
		if(param->p < 0)
			return "p < 0";

	if(param->shrinking != 0 &&
	   param->shrinking != 1)
		return "shrinking != 0 and shrinking != 1";


	// check whether nu-svc is feasible

	if(svm_type == NU_SVC)
	{
		int32_t l = prob->l;
		int32_t max_nr_class = 16;
		int32_t nr_class = 0;
		int32_t *label = SG_MALLOC(int32_t, max_nr_class);
		int32_t *count = SG_MALLOC(int32_t, max_nr_class);

		int32_t i;
		for(i=0;i<l;i++)
		{
			int32_t this_label = (int32_t) prob->y[i];
			int32_t j;
			for(j=0;j<nr_class;j++)
				if(this_label == label[j])
				{
					++count[j];
					break;
				}
			if(j == nr_class)
			{
				if(nr_class == max_nr_class)
				{
					int32_t old_max_nr_class = max_nr_class;
					max_nr_class *= 2;
					label=SG_REALLOC(int32_t, label, old_max_nr_class, max_nr_class);
					count=SG_REALLOC(int32_t, count, old_max_nr_class, max_nr_class);
				}
				label[nr_class] = this_label;
				count[nr_class] = 1;
				++nr_class;
			}
		}

		for(i=0;i<nr_class;i++)
		{
			int32_t n1 = count[i];
			for(int32_t j=i+1;j<nr_class;j++)
			{
				int32_t n2 = count[j];
				if(param->nu*(n1+n2)/2 > CMath::min(n1,n2))
				{
					SG_FREE(label);
					SG_FREE(count);
					return "specified nu is infeasible";
				}
			}
		}
		SG_FREE(label);
		SG_FREE(count);
	}

	return NULL;
}
}
#endif // DOXYGEN_SHOULD_SKIP_THIS
