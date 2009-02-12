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

#include "lib/memory.h"
#include "classifier/svm/SVM_libsvm.h"
#include "kernel/Kernel.h"
#include "lib/io.h"
#include "lib/common.h"
#include "lib/Mathematics.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
typedef KERNELCACHE_ELEM Qfloat;
typedef float64_t schar;
#ifndef min
template <class T> inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class T> inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
template <class S, class T> inline void clone(T*& dst, S* src, int32_t n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

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
	head = (head_t *)calloc(l,sizeof(head_t));	// initialized to 0
	size /= sizeof(Qfloat);
	size -= l * sizeof(head_t) / sizeof(Qfloat);
	size = max(size, (int64_t) 2*l);	// cache must be large enough for two columns
	lru_head.next = lru_head.prev = &lru_head;
}

Cache::~Cache()
{
	for(head_t *h = lru_head.next; h != &lru_head; h=h->next)
		free(h->data);
	free(head);
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
			free(old->data);
			size += old->len;
			old->data = 0;
			old->len = 0;
		}

		// allocate new space
		h->data = (Qfloat *)realloc(h->data,sizeof(Qfloat)*len);
		size -= more;
		swap(h->len,len);
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
	swap(head[i].data,head[j].data);
	swap(head[i].len,head[j].len);
	if(head[i].len) lru_insert(&head[i]);
	if(head[j].len) lru_insert(&head[j]);

	if(i>j) swap(i,j);
	for(head_t *h = lru_head.next; h!=&lru_head; h=h->next)
	{
		if(h->len > i)
		{
			if(h->len > j)
				swap(h->data[i],h->data[j]);
			else
			{
				// give up
				lru_delete(h);
				free(h->data);
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
};

class Kernel: public QMatrix {
public:
	Kernel(int32_t l, svm_node * const * x, const svm_parameter& param);
	virtual ~Kernel();

	virtual Qfloat *get_Q(int32_t column, int32_t len) const = 0;
	virtual Qfloat *get_QD() const = 0;
	virtual void swap_index(int32_t i, int32_t j) const	// no so const...
	{
		swap(x[i],x[j]);
		if(x_square) swap(x_square[i],x_square[j]);
	}

	inline float64_t kernel_function(int32_t i, int32_t j) const
	{
		return kernel->kernel(x[i]->index,x[j]->index);
	}

private:
	CKernel* kernel;
	const svm_node **x;
	float64_t *x_square;

	// svm_parameter
	const int32_t kernel_type;
	const int32_t degree;
	const float64_t gamma;
	const float64_t coef0;
};

Kernel::Kernel(int32_t l, svm_node * const * x_, const svm_parameter& param)
: kernel_type(param.kernel_type), degree(param.degree),
 gamma(param.gamma), coef0(param.coef0)
{
	clone(x,x_,l);
	x_square = 0;
	kernel=param.kernel;
}

Kernel::~Kernel()
{
	delete[] x;
	delete[] x_square;
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
		SolutionInfo* si, int32_t shrinking);

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
	swap(y[i],y[j]);
	swap(G[i],G[j]);
	swap(alpha_status[i],alpha_status[j]);
	swap(alpha[i],alpha[j]);
	swap(p[i],p[j]);
	swap(active_set[i],active_set[j]);
	swap(G_bar[i],G_bar[j]);
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
	float64_t p_eps, SolutionInfo* p_si, int32_t shrinking)
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
		alpha_status = new char[l];
		for(int32_t i=0;i<l;i++)
			update_alpha_status(i);
	}

	// initialize active set (for shrinking)
	{
		active_set = new int32_t[l];
		for(int32_t i=0;i<l;i++)
			active_set[i] = i;
		active_size = l;
	}

	// initialize gradient
	{
		G = new float64_t[l];
		G_bar = new float64_t[l];
		int32_t i;
		for(i=0;i<l;i++)
		{
			G[i] = p_p[i];
			G_bar[i] = 0;
		}
		for(i=0;i<l;i++)
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
	}

	// optimization step

	int32_t iter = 0;
	int32_t counter = min(l,1000)+1;

	while(1)
	{
		// show progress and do shrinking

		if(--counter == 0)
		{
			counter = min(l,1000);
			if(shrinking) do_shrinking();
			//SG_SINFO(".");
		}

		int32_t i,j;
		float64_t gap;
		if(select_working_set(i,j, gap)!=0)
		{
			// reconstruct the whole gradient
			reconstruct_gradient();
			// reset active set size and check
			active_size = l;
			//SG_SINFO("*");
			if(select_working_set(i,j, gap)!=0)
				break;
			else
				counter = 1;	// do shrinking next iteration
		}

		SG_SABS_PROGRESS(gap, -CMath::log10(gap), -CMath::log10(1), -CMath::log10(eps), 6);
		
		++iter;

		// update alpha[i] and alpha[j], handle bounds carefully
		
		const Qfloat *Q_i = Q->get_Q(i,active_size);
		const Qfloat *Q_j = Q->get_Q(j,active_size);

		float64_t C_i = get_C(i);
		float64_t C_j = get_C(j);

		float64_t old_alpha_i = alpha[i];
		float64_t old_alpha_j = alpha[j];

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
	}

	// calculate rho

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

	SG_SINFO("\noptimization finished, #iter = %d\n",iter);

	delete[] p;
	delete[] y;
	delete[] alpha;
	delete[] alpha_status;
	delete[] active_set;
	delete[] G;
	delete[] G_bar;
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

bool Solver::be_shrunk(int i, float64_t Gmax1, float64_t Gmax2)
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
				ub = min(ub,yG);
			else
				lb = max(lb,yG);
		}
		else if(is_lower_bound(i))
		{
			if(y[i]==+1)
				ub = min(ub,yG);
			else
				lb = max(lb,yG);
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
		float64_t p_eps, SolutionInfo* p_si, int32_t shrinking)
	{
		this->si = p_si;
		Solver::Solve(p_l,p_Q,p,p_y,p_alpha,p_Cp,p_Cn,p_eps,p_si,shrinking);
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

	gap=max(Gmaxp+Gmaxp2,Gmaxn+Gmaxn2);
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

	if(unshrink == false && max(Gmax1+Gmax2,Gmax3+Gmax4) <= eps*10) 
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
				lb1 = max(lb1,G[i]);
			else if(is_lower_bound(i))
				ub1 = min(ub1,G[i]);
			else
			{
				++nr_free1;
				sum_free1 += G[i];
			}
		}
		else
		{
			if(is_upper_bound(i))
				lb2 = max(lb2,G[i]);
			else if(is_lower_bound(i))
				ub2 = min(ub2,G[i]);
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

//
// Q matrices for various formulations
//
class SVC_Q: public Kernel
{ 
public:
	SVC_Q(const svm_problem& prob, const svm_parameter& param, const schar *y_)
	:Kernel(prob.l, prob.x, param)
	{
		clone(y,y_,prob.l);
		cache = new Cache(prob.l,(int64_t)(param.cache_size*(1l<<20)));
		QD = new Qfloat[prob.l];
		for(int32_t i=0;i<prob.l;i++)
			QD[i]= (Qfloat)kernel_function(i,i);
	}
	
	Qfloat *get_Q(int32_t i, int32_t len) const
	{
		Qfloat *data;
		int32_t start;
		if((start = cache->get_data(i,&data,len)) < len)
		{
			for(int32_t j=start;j<len;j++)
				data[j] = (Qfloat) y[i]*y[j]*kernel_function(i,j);
		}
		return data;
	}

	Qfloat *get_QD() const
	{
		return QD;
	}

	void swap_index(int32_t i, int32_t j) const
	{
		cache->swap_index(i,j);
		Kernel::swap_index(i,j);
		swap(y[i],y[j]);
		swap(QD[i],QD[j]);
	}

	~SVC_Q()
	{
		delete[] y;
		delete cache;
		delete[] QD;
	}
private:
	schar *y;
	Cache *cache;
	Qfloat *QD;
};

class ONE_CLASS_Q: public Kernel
{
public:
	ONE_CLASS_Q(const svm_problem& prob, const svm_parameter& param)
	:Kernel(prob.l, prob.x, param)
	{
		cache = new Cache(prob.l,(int64_t)(param.cache_size*(1l<<20)));
		QD = new Qfloat[prob.l];
		for(int32_t i=0;i<prob.l;i++)
			QD[i]= (Qfloat)kernel_function(i,i);
	}
	
	Qfloat *get_Q(int32_t i, int32_t len) const
	{
		Qfloat *data;
		int32_t start;
		if((start = cache->get_data(i,&data,len)) < len)
		{
			for(int32_t j=start;j<len;j++)
				data[j] = (Qfloat) kernel_function(i,j);
		}
		return data;
	}

	Qfloat *get_QD() const
	{
		return QD;
	}

	void swap_index(int32_t i, int32_t j) const
	{
		cache->swap_index(i,j);
		Kernel::swap_index(i,j);
		swap(QD[i],QD[j]);
	}

	~ONE_CLASS_Q()
	{
		delete cache;
		delete[] QD;
	}
private:
	Cache *cache;
	Qfloat *QD;
};

class SVR_Q: public Kernel
{ 
public:
	SVR_Q(const svm_problem& prob, const svm_parameter& param)
	:Kernel(prob.l, prob.x, param)
	{
		l = prob.l;
		cache = new Cache(l,(int64_t)(param.cache_size*(1l<<20)));
		QD = new Qfloat[2*l];
		sign = new schar[2*l];
		index = new int32_t[2*l];
		for(int32_t k=0;k<l;k++)
		{
			sign[k] = 1;
			sign[k+l] = -1;
			index[k] = k;
			index[k+l] = k;
			QD[k]= (Qfloat)kernel_function(k,k);
			QD[k+l]=QD[k];
		}
		buffer[0] = new Qfloat[2*l];
		buffer[1] = new Qfloat[2*l];
		next_buffer = 0;
	}

	void swap_index(int32_t i, int32_t j) const
	{
		swap(sign[i],sign[j]);
		swap(index[i],index[j]);
		swap(QD[i],QD[j]);
	}
	
	Qfloat *get_Q(int32_t i, int32_t len) const
	{
		Qfloat *data;
		int32_t real_i = index[i];
		if(cache->get_data(real_i,&data,l) < l)
		{
			for(int32_t j=0;j<l;j++)
				data[j] = (Qfloat)kernel_function(real_i,j);
		}

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
		delete[] sign;
		delete[] index;
		delete[] buffer[0];
		delete[] buffer[1];
		delete[] QD;
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
	float64_t *minus_ones = new float64_t[l];
	schar *y = new schar[l];

	int32_t i;

	for(i=0;i<l;i++)
	{
		alpha[i] = 0;
		minus_ones[i] = -1;
		if(prob->y[i] > 0) y[i] = +1; else y[i]=-1;
	}

	Solver s;
	s.Solve(l, SVC_Q(*prob,*param,y), minus_ones, y,
		alpha, Cp, Cn, param->eps, si, param->shrinking);

	float64_t sum_alpha=0;
	for(i=0;i<l;i++)
		sum_alpha += alpha[i];

	if (Cp==Cn)
		SG_SINFO("nu = %f\n", sum_alpha/(param->C*prob->l));

	for(i=0;i<l;i++)
		alpha[i] *= y[i];

	delete[] minus_ones;
	delete[] y;
}

static void solve_nu_svc(
	const svm_problem *prob, const svm_parameter *param,
	float64_t *alpha, Solver::SolutionInfo* si)
{
	int32_t i;
	int32_t l = prob->l;
	float64_t nu = param->nu;

	schar *y = new schar[l];

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
			alpha[i] = min(1.0,sum_pos);
			sum_pos -= alpha[i];
		}
		else
		{
			alpha[i] = min(1.0,sum_neg);
			sum_neg -= alpha[i];
		}

	float64_t *zeros = new float64_t[l];

	for(i=0;i<l;i++)
		zeros[i] = 0;

	Solver_NU s;
	s.Solve(l, SVC_Q(*prob,*param,y), zeros, y,
		alpha, 1.0, 1.0, param->eps, si,  param->shrinking);
	float64_t r = si->r;

	SG_SINFO("C = %f\n",1/r);

	for(i=0;i<l;i++)
		alpha[i] *= y[i]/r;

	si->rho /= r;
	si->obj /= (r*r);
	si->upper_bound_p = 1/r;
	si->upper_bound_n = 1/r;

	delete[] y;
	delete[] zeros;
}

static void solve_one_class(
	const svm_problem *prob, const svm_parameter *param,
	float64_t *alpha, Solver::SolutionInfo* si)
{
	int32_t l = prob->l;
	float64_t *zeros = new float64_t[l];
	schar *ones = new schar[l];
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
		alpha, 1.0, 1.0, param->eps, si, param->shrinking);

	delete[] zeros;
	delete[] ones;
}

static void solve_epsilon_svr(
	const svm_problem *prob, const svm_parameter *param,
	float64_t *alpha, Solver::SolutionInfo* si)
{
	int32_t l = prob->l;
	float64_t *alpha2 = new float64_t[2*l];
	float64_t *linear_term = new float64_t[2*l];
	schar *y = new schar[2*l];
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
		alpha2, param->C, param->C, param->eps, si, param->shrinking);

	float64_t sum_alpha = 0;
	for(i=0;i<l;i++)
	{
		alpha[i] = alpha2[i] - alpha2[i+l];
		sum_alpha += fabs(alpha[i]);
	}
	SG_SINFO("nu = %f\n",sum_alpha/(param->C*l));

	delete[] alpha2;
	delete[] linear_term;
	delete[] y;
}

static void solve_nu_svr(
	const svm_problem *prob, const svm_parameter *param,
	float64_t *alpha, Solver::SolutionInfo* si)
{
	int32_t l = prob->l;
	float64_t C = param->C;
	float64_t *alpha2 = new float64_t[2*l];
	float64_t *linear_term = new float64_t[2*l];
	schar *y = new schar[2*l];
	int32_t i;

	float64_t sum = C * param->nu * l / 2;
	for(i=0;i<l;i++)
	{
		alpha2[i] = alpha2[i+l] = min(sum,C);
		sum -= alpha2[i];

		linear_term[i] = - prob->y[i];
		y[i] = 1;

		linear_term[i+l] = prob->y[i];
		y[i+l] = -1;
	}

	Solver_NU s;
	s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
		alpha2, C, C, param->eps, si, param->shrinking);

	SG_SINFO("epsilon = %f\n",-si->r);

	for(i=0;i<l;i++)
		alpha[i] = alpha2[i] - alpha2[i+l];

	delete[] alpha2;
	delete[] linear_term;
	delete[] y;
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
	float64_t *alpha = Malloc(float64_t, prob->l);
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

	SG_SINFO("obj = %.16f, rho = %.16f\n",si.obj,si.rho);

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
		SG_SINFO("nSV = %d, nBSV = %d\n",nSV,nBSV);
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
	int32_t *label = Malloc(int32_t, max_nr_class);
	int32_t *count = Malloc(int32_t, max_nr_class);
	int32_t *data_label = Malloc(int32_t, l);
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
				max_nr_class *= 2;
				label=(int32_t *) realloc(label,max_nr_class*sizeof(int32_t));
				count=(int32_t *) realloc(count,max_nr_class*sizeof(int32_t));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	int32_t *start = Malloc(int32_t, nr_class);
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
	free(data_label);
}

//
// Interface functions
//
svm_model *svm_train(const svm_problem *prob, const svm_parameter *param)
{
	svm_model *model = Malloc(svm_model,1);
	model->param = *param;
	model->free_sv = 0;	// XXX

	if(param->svm_type == ONE_CLASS ||
	   param->svm_type == EPSILON_SVR ||
	   param->svm_type == NU_SVR)
	{
		SG_SINFO("training one class svm or doing epsilon sv regression\n");

		// regression or one-class-svm
		model->nr_class = 2;
		model->label = NULL;
		model->nSV = NULL;
		model->sv_coef = Malloc(float64_t *,1);
		decision_function f = svm_train_one(prob,param,0,0);
		model->rho = Malloc(float64_t, 1);
		model->rho[0] = f.rho;
		model->objective = f.objective;

		int32_t nSV = 0;
		int32_t i;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i]) > 0) ++nSV;
		model->l = nSV;
		model->SV = Malloc(svm_node *,nSV);
		model->sv_coef[0] = Malloc(float64_t, nSV);
		int32_t j = 0;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i]) > 0)
			{
				model->SV[j] = prob->x[i];
				model->sv_coef[0][j] = f.alpha[i];
				++j;
			}		

		free(f.alpha);
	}
	else
	{
		// classification
		int32_t l = prob->l;
		int32_t nr_class;
		int32_t *label = NULL;
		int32_t *start = NULL;
		int32_t *count = NULL;
		int32_t *perm = Malloc(int32_t, l);

		// group training data of the same class
		svm_group_classes(prob,&nr_class,&label,&start,&count,perm);
		svm_node **x = Malloc(svm_node *,l);
		int32_t i;
		for(i=0;i<l;i++)
			x[i] = prob->x[perm[i]];

		// calculate weighted C

		float64_t *weighted_C = Malloc(float64_t,  nr_class);
		for(i=0;i<nr_class;i++)
			weighted_C[i] = param->C;
		for(i=0;i<param->nr_weight;i++)
		{
			int32_t j;
			for(j=0;j<nr_class;j++)
				if(param->weight_label[i] == label[j])
					break;
			if(j == nr_class)
				SG_SWARNING("warning: class label %d specified in weight is not found\n", param->weight_label[i]);
			else
				weighted_C[j] *= param->weight[i];
		}

		// train k*(k-1)/2 models
		
		bool *nonzero = Malloc(bool,l);
		for(i=0;i<l;i++)
			nonzero[i] = false;
		decision_function *f = Malloc(decision_function,nr_class*(nr_class-1)/2);

		int32_t p = 0;
		for(i=0;i<nr_class;i++)
			for(int32_t j=i+1;j<nr_class;j++)
			{
				svm_problem sub_prob;
				int32_t si = start[i], sj = start[j];
				int32_t ci = count[i], cj = count[j];
				sub_prob.l = ci+cj;
				sub_prob.x = Malloc(svm_node *,sub_prob.l);
				sub_prob.y = Malloc(float64_t, sub_prob.l+1); //dirty hack to surpress valgrind err

				int32_t k;
				for(k=0;k<ci;k++)
				{
					sub_prob.x[k] = x[si+k];
					sub_prob.y[k] = +1;
				}
				for(k=0;k<cj;k++)
				{
					sub_prob.x[ci+k] = x[sj+k];
					sub_prob.y[ci+k] = -1;
				}
				sub_prob.y[sub_prob.l]=-1; //dirty hack to surpress valgrind err
				
				f[p] = svm_train_one(&sub_prob,param,weighted_C[i],weighted_C[j]);
				for(k=0;k<ci;k++)
					if(!nonzero[si+k] && fabs(f[p].alpha[k]) > 0)
						nonzero[si+k] = true;
				for(k=0;k<cj;k++)
					if(!nonzero[sj+k] && fabs(f[p].alpha[ci+k]) > 0)
						nonzero[sj+k] = true;
				free(sub_prob.x);
				free(sub_prob.y);
				++p;
			}

		// build output

		model->objective = f[0].objective;
		model->nr_class = nr_class;
		
		model->label = Malloc(int32_t, nr_class);
		for(i=0;i<nr_class;i++)
			model->label[i] = label[i];
		
		model->rho = Malloc(float64_t, nr_class*(nr_class-1)/2);
		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			model->rho[i] = f[i].rho;

		int32_t total_sv = 0;
		int32_t *nz_count = Malloc(int32_t, nr_class);
		model->nSV = Malloc(int32_t, nr_class);
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
		
		SG_SINFO("Total nSV = %d\n",total_sv);

		model->l = total_sv;
		model->SV = Malloc(svm_node *,total_sv);
		p = 0;
		for(i=0;i<l;i++)
			if(nonzero[i]) model->SV[p++] = x[i];

		int32_t *nz_start = Malloc(int32_t, nr_class);
		nz_start[0] = 0;
		for(i=1;i<nr_class;i++)
			nz_start[i] = nz_start[i-1]+nz_count[i-1];

		model->sv_coef = Malloc(float64_t *,nr_class-1);
		for(i=0;i<nr_class-1;i++)
			model->sv_coef[i] = Malloc(float64_t, total_sv);

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
		
		free(label);
		free(count);
		free(perm);
		free(start);
		free(x);
		free(weighted_C);
		free(nonzero);
		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			free(f[i].alpha);
		free(f);
		free(nz_count);
		free(nz_start);
	}
	return model;
}

const char *svm_type_table[] =
{
	"c_svc","nu_svc","one_class","epsilon_svr","nu_svr",NULL
};

const char *kernel_type_table[]=
{
	"linear","polynomial","rbf","sigmoid","precomputed",NULL
};

void svm_destroy_model(svm_model* model)
{
	if(model->free_sv && model->l > 0)
		free((void *)(model->SV[0]));
	for(int32_t i=0;i<model->nr_class-1;i++)
		free(model->sv_coef[i]);
	free(model->SV);
	free(model->sv_coef);
	free(model->rho);
	free(model->label);
	free(model->nSV);
	free(model);
}

void svm_destroy_param(svm_parameter* param)
{
	free(param->weight_label);
	free(param->weight);
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
	   svm_type != NU_SVR)
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
		int32_t *label = Malloc(int32_t, max_nr_class);
		int32_t *count = Malloc(int32_t, max_nr_class);

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
					max_nr_class *= 2;
					label=(int32_t *) realloc(label,
						max_nr_class*sizeof(int32_t));
					count=(int32_t *) realloc(count,
						max_nr_class*sizeof(int32_t));
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
				if(param->nu*(n1+n2)/2 > min(n1,n2))
				{
					free(label);
					free(count);
					return "specified nu is infeasible";
				}
			}
		}
		free(label);
		free(count);
	}

	return NULL;
}
