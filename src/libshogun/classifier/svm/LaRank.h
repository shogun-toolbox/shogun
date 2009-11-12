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

#ifndef LARANK_H
#define LARANK_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
# include <ctime>
# include <sys/time.h>
# include <ext/hash_map>
# include <ext/hash_set>
# include <cmath>
# define STDEXT_NAMESPACE __gnu_cxx
# define std_hash_map STDEXT_NAMESPACE::hash_map
# define std_hash_set STDEXT_NAMESPACE::hash_set

#include "lib/io.h"
#include "classifier/svm/MultiClassSVM.h"

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
 * $Id: kcache.h,v 1.8 2007/01/25 22:42:09 leonb Exp $
 **********************************************************************/

#ifndef KCACHE_H
#define KCACHE_H

#include "kernel/Kernel.h"

namespace shogun
{
/* ------------------------------------- */
/* CACHE FOR KERNEL VALUES */


/* --- larank_kcache_t
   This is the opaque data structure for a kernel cache.
*/
  typedef struct larank_kcache_s larank_kcache_t;

/* --- larank_kcache_create
   Returns a cache object for kernel <kernelfun>.
   The cache handles a Gram matrix of size <n>x<n> at most.
 */
  larank_kcache_t *larank_kcache_create (CKernel* kernelfunc);

/* --- larank_kcache_destroy
   Deallocates a kernel cache object.
*/
  void larank_kcache_destroy (larank_kcache_t * self);

/* --- larank_kcache_set_maximum_size
   Sets the maximum memory size used by the cache.
   Argument <entries> indicates the maximum cache memory in bytes
   The default size is 256Mb.
*/
  void larank_kcache_set_maximum_size (larank_kcache_t * self, long entries);

/* --- larank_kcache_get_maximum_size
   Returns the maximum cache memory.
 */
  long larank_kcache_get_maximum_size (larank_kcache_t * self);

/* --- larank_kcache_get_current_size
   Returns the currently used cache memory.
   This can slighly exceed the value specified by 
   <larank_kcache_set_maximum_size>.
 */
  long larank_kcache_get_current_size (larank_kcache_t * self);

/* --- larank_kcache_query
   Returns the possibly cached value of the Gram matrix element (<i>,<j>).
   This function will not modify the cache geometry.
 */
  double larank_kcache_query (larank_kcache_t * self, int i, int j);

/* --- larank_kcache_query_row
   Returns the <len> first elements of row <i> of the Gram matrix.
   The cache user can modify the order of the row elements
   using the larank_kcache_swap() functions.  Functions larank_kcache_i2r() 
   and larank_kcache_r2i() convert from example index to row position 
   and vice-versa.
*/

  float *larank_kcache_query_row (larank_kcache_t * self, int i, int len);

/* --- larank_kcache_status_row
   Returns the number of cached entries for row i.
*/

  int larank_kcache_status_row (larank_kcache_t * self, int i);

/* --- larank_kcache_discard_row
   Indicates that we wont need row i in the near future.
*/

  void larank_kcache_discard_row (larank_kcache_t * self, int i);


/* --- larank_kcache_i2r
   --- larank_kcache_r2i
   Return an array of integer of length at least <n> containing
   the conversion table from example index to row position and vice-versa. 
*/

  int *larank_kcache_i2r (larank_kcache_t * self, int n);
  int *larank_kcache_r2i (larank_kcache_t * self, int n);


/* --- larank_kcache_swap_rr
   --- larank_kcache_swap_ii
   --- larank_kcache_swap_ri
   Swaps examples in the row ordering table.
   Examples can be specified by indicating their row position (<r1>, <r2>)
   or by indicating the example number (<i1>, <i2>).
*/

  void larank_kcache_swap_rr (larank_kcache_t * self, int r1, int r2);
  void larank_kcache_swap_ii (larank_kcache_t * self, int i1, int i2);
  void larank_kcache_swap_ri (larank_kcache_t * self, int r1, int i2);


/* --- larank_kcache_set_buddy
   This function is called to indicate that the caches <self> and <buddy>
   implement the same kernel function. When a buddy experiences a cache
   miss, it can try querying its buddies instead of calling the 
   kernel function.  Buddy relationship is transitive. */

  void larank_kcache_set_buddy (larank_kcache_t * self,
				larank_kcache_t * buddy);
}
#endif

namespace shogun
{
	/*
	 **	LARANKPATTERN: to keep track of the support patterns
	 */
	class LaRankPattern
	{
		public:
			LaRankPattern (int x_index, int label) 
				: x_id (x_index), y (label) {}
			LaRankPattern () 
				: x_id (0) {}

			bool exists () const
			{
				return x_id >= 0;
			}

			void clear ()
			{
				x_id = -1;
			}

			int x_id;
			int y;
	};


	/*
	 **  LARANKPATTERNS: the collection of support patterns
	 */
	class LaRankPatterns
	{
		public:
			LaRankPatterns () {}
			~LaRankPatterns () {}

			void insert (const LaRankPattern & pattern)
			{
				if (!isPattern (pattern.x_id))
				{
					if (freeidx.size ())
					{
						std_hash_set < unsigned >::iterator it = freeidx.begin ();
						patterns[*it] = pattern;
						x_id2rank[pattern.x_id] = *it;
						freeidx.erase (it);
					}
					else
					{
						patterns.push_back (pattern);
						x_id2rank[pattern.x_id] = patterns.size () - 1;
					}
				}
				else
				{
					int rank = getPatternRank (pattern.x_id);
					patterns[rank] = pattern;
				}
			}

			void remove (unsigned i)
			{
				x_id2rank[patterns[i].x_id] = 0;
				patterns[i].clear ();
				freeidx.insert (i);
			}

			bool empty () const
			{
				return patterns.size () == freeidx.size ();
			}

			unsigned size () const
			{
				return patterns.size () - freeidx.size ();
			}

			LaRankPattern & sample ()
			{
				ASSERT (!empty ());
				while (true)
				{
					unsigned r = rand () % patterns.size ();
					if (patterns[r].exists ())
						return patterns[r];
				}
				return patterns[0];
			}

			unsigned getPatternRank (int x_id)
			{
				return x_id2rank[x_id];
			}

			bool isPattern (int x_id)
			{
				return x_id2rank[x_id] != 0;
			}

			LaRankPattern & getPattern (int x_id)
			{
				unsigned rank = x_id2rank[x_id];
				return patterns[rank];
			}

			unsigned maxcount () const
			{
				return patterns.size ();
			}

			LaRankPattern & operator [] (unsigned i)
			{
				return patterns[i];
			}

			const LaRankPattern & operator [] (unsigned i) const
			{
				return patterns[i];
			}

		private:
			std_hash_set < unsigned >freeidx;
			std::vector < LaRankPattern > patterns;
			std_hash_map < int, unsigned >x_id2rank;
	};


	/*
	 ** OUTPUT: one per class of the raining set, keep tracks of support
	 * vectors and their beta coefficients
	 */
	class LaRankOutput
	{
		public:
			LaRankOutput () {}
			~LaRankOutput () {}

			// Initializing an output class (basically creating a kernel cache for it)
			void initialize (CKernel* kfunc, long cache);

			// Destroying an output class (basically destroying the kernel cache)
			void destroy ();

			// !Important! Computing the score of a given input vector for the actual output
			double computeScore (int x_id);

			// !Important! Computing the gradient of a given input vector for the actual output           
			double computeGradient (int xi_id, int yi, int ythis);

			// Updating the solution in the actual output
			void update (int x_id, double lambda, double gp);

			// Linking the cahe of this output to the cache of an other "buddy" output
			// so that if a requested value is not found in this cache, you can ask your buddy if it has it.                              
			void set_kernel_buddy (larank_kcache_t * bud);

			// Removing useless support vectors (for which beta=0)                
			int cleanup ();

			// --- Below are information or "get" functions --- //

			//                            
			inline larank_kcache_t *getKernel () const
			{
				return kernel;
			}
			//                            
			inline int32_t get_l () const
			{
				return l;
			}

			//
			double getW2 ();

			//
			inline double getKii (int x_id)
			{
				return larank_kcache_query (kernel, x_id, x_id);
			}

			//
			double getBeta (int x_id);

			//
			inline float* getBetas () const
			{
				return beta;
			}

			//
			double getGradient (int x_id);

			//
			inline bool isSupportVector (int x_id) const
			{
				int *r2i = larank_kcache_r2i (kernel, l);
				int xr = -1;
				for (int r = 0; r < l; r++)
					if (r2i[r] == x_id)
					{
						xr = r;
						break;
					}
				return (xr >= 0);
			}

			//
			int getSV (float* &sv) const;

		private:
			// the solution of LaRank relative to the actual class is stored in this parameters

			float* beta;		// Beta coefficiens
			float* g;		// Strored gradient derivatives
			larank_kcache_t *kernel;	// Cache for kernel values
			int l;			// Number of support vectors 
	};

	/*
	 ** MACHINE: the main thing, which is trained.
	 */
	class CLaRank:  public CMultiClassSVM
	{
		public:
			CLaRank ();

			/** constructor
			 *
			 * @param C constant C
			 * @param k kernel
			 * @param lab labels
			 */
			CLaRank(float64_t C, CKernel* k, CLabels* lab);

			~CLaRank () {}

			bool train(CFeatures* data);


			// LEARNING FUNCTION: add new patterns and run optimization steps selected with adaptative schedule
			virtual int add (int x_id, int yi);

			// PREDICTION FUNCTION: main function in la_rank_classify
			virtual int predict (int x_id);

			virtual void destroy ();

			// Compute Duality gap (costly but used in stopping criteria in batch mode)                     
			virtual double computeGap ();

			// Display stuffs along learning
			virtual void printStuff (double initime, bool print_dual);


			// Nuber of classes so far
			virtual unsigned getNumOutputs () const;

			// Number of Support Vectors
			int getNSV ();

			// Norm of the parameters vector
			double computeW2 ();

			// Compute Dual objective value
			double getDual ();

			/** get classifier type
			 *
			 * @return classifier type LIBSVM
			 */
			virtual inline EClassifierType get_classifier_type() { return CT_LARANK; }

			/** @return object name */
			inline virtual const char* get_name() const { return "LaRank"; }

			void set_batch_mode(bool enable) { batch_mode=enable; };
			bool get_batch_mode() { return batch_mode; };
			void set_tau(double t) { tau=t; };
			double get_tau() { return tau; };


		private:
			/*
			 ** MAIN DARK OPTIMIZATION PROCESSES
			 */

			// Hash Table used to store the different outputs
			typedef std_hash_map < int, LaRankOutput > outputhash_t;	// class index -> LaRankOutput
			outputhash_t outputs;
			LaRankOutput *getOutput (int index);

			// 
			LaRankPatterns patterns;

			// Parameters
			int nb_seen_examples;
			int nb_removed;

			// Numbers of each operation performed so far
			int n_pro;
			int n_rep;
			int n_opt;

			// Running estimates for each operations 
			double w_pro;
			double w_rep;
			double w_opt;

			int y0;
			double dual;

			struct outputgradient_t
			{
				outputgradient_t (int result_output, double result_gradient)
					: output (result_output), gradient (result_gradient) {}
				outputgradient_t ()
					: output (0), gradient (0) {}

				int output;
				double gradient;

				bool operator < (const outputgradient_t & og) const
				{
					return gradient > og.gradient;
				}
			};

			//3 types of operations in LaRank               
			enum process_type
			{
				processNew,
				processOld,
				processOptimize
			};

			struct process_return_t
			{
				process_return_t (double dual, int yprediction) 
					: dual_increase (dual), ypred (yprediction) {}
				process_return_t () {}
				double dual_increase;
				int ypred;
			};

			// IMPORTANT Main SMO optimization step
			process_return_t process (const LaRankPattern & pattern, process_type ptype);

			// ProcessOld
			double reprocess ();

			// Optimize
			double optimize ();

			// remove patterns and return the number of patterns that were removed
			unsigned cleanup ();

		protected:

			std_hash_set < int >classes;

			unsigned class_count () const
			{
				return classes.size ();
			}

			double tau;
			int nb_train;
			long cache;
			// whether to use online learning or batch training
			bool batch_mode;
	};

	inline double getTime ()
	{
		struct timeval tv;
		struct timezone tz;
		long int sec;
		long int usec;
		double mytime;
		gettimeofday (&tv, &tz);
		sec = (long int) tv.tv_sec;
		usec = (long int) tv.tv_usec;
		mytime = (double) sec + usec * 0.000001;
		return mytime;
	}
}
#endif
