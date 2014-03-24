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
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA
 *
 ***********************************************************************/

/***********************************************************************
 * $Id: kcache.h,v 1.8 2007/01/25 22:42:09 leonb Exp $
 **********************************************************************/

#ifndef LARANK_H
#define LARANK_H

#include <ctime>
#include <vector>
#include <algorithm>
#include <sys/time.h>
#include <set>
#include <map>
#define STDEXT_NAMESPACE __gnu_cxx
#define std_hash_map std::map
#define std_hash_set std::set

#include <shogun/lib/config.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/multiclass/MulticlassSVM.h>

namespace shogun
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
	struct larank_kcache_s;
	typedef struct larank_kcache_s larank_kcache_t;
	struct larank_kcache_s
	{
		CKernel* func;
		larank_kcache_t *prevbuddy;
		larank_kcache_t *nextbuddy;
		int64_t maxsize;
		int64_t cursize;
		int32_t l;
		int32_t *i2r;
		int32_t *r2i;
		int32_t maxrowlen;
		/* Rows */
		int32_t *rsize;
		float32_t *rdiag;
		float32_t **rdata;
		int32_t *rnext;
		int32_t *rprev;
		int32_t *qnext;
		int32_t *qprev;
	};

	/*
	 ** OUTPUT: one per class of the raining set, keep tracks of support
	 * vectors and their beta coefficients
	 */
	class LaRankOutput
	{
		public:
			LaRankOutput () : beta(NULL), g(NULL), kernel(NULL), l(0)
		{
		}
			virtual ~LaRankOutput ()
			{
				destroy();
			}

			// Initializing an output class (basically creating a kernel cache for it)
			void initialize (CKernel* kfunc, int64_t cache);

			// Destroying an output class (basically destroying the kernel cache)
			void destroy ();

			// !Important! Computing the score of a given input vector for the actual output
			float64_t computeScore (int32_t x_id);

			// !Important! Computing the gradient of a given input vector for the actual output
			float64_t computeGradient (int32_t xi_id, int32_t yi, int32_t ythis);

			// Updating the solution in the actual output
			void update (int32_t x_id, float64_t lambda, float64_t gp);

			// Linking the cache of this output to the cache of an other "buddy" output
			// so that if a requested value is not found in this cache, you can
			// ask your buddy if it has it.
			void set_kernel_buddy (larank_kcache_t * bud);

			// Removing useless support vectors (for which beta=0)
			int32_t cleanup ();

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
			float64_t getW2 ();

			//
			float64_t getKii (int32_t x_id);

			//
			float64_t getBeta (int32_t x_id);

			//
			inline float32_t* getBetas () const
			{
				return beta;
			}

			//
			float64_t getGradient (int32_t x_id);

			//
			bool isSupportVector (int32_t x_id) const;

			//
			int32_t getSV (float32_t* &sv) const;

		private:
			// the solution of LaRank relative to the actual class is stored in
			// this parameters
			float32_t* beta;		// Beta coefficiens
			float32_t* g;		// Strored gradient derivatives
			larank_kcache_t *kernel;	// Cache for kernel values
			int32_t l;			// Number of support vectors
	};

	/*
	 **	LARANKPATTERN: to keep track of the support patterns
	 */
	class LaRankPattern
	{
		public:
			LaRankPattern (int32_t x_index, int32_t label)
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

			int32_t x_id;
			int32_t y;
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
						std_hash_set < uint32_t >::iterator it = freeidx.begin ();
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
					int32_t rank = getPatternRank (pattern.x_id);
					patterns[rank] = pattern;
				}
			}

			void remove (uint32_t i)
			{
				x_id2rank[patterns[i].x_id] = 0;
				patterns[i].clear ();
				freeidx.insert (i);
			}

			bool empty () const
			{
				return patterns.size () == freeidx.size ();
			}

			uint32_t size () const
			{
				return patterns.size () - freeidx.size ();
			}

			LaRankPattern & sample ()
			{
				ASSERT (!empty ())
				while (true)
				{
					uint32_t r = CMath::random(uint32_t(0), uint32_t(patterns.size ()-1));
					if (patterns[r].exists ())
						return patterns[r];
				}
				return patterns[0];
			}

			uint32_t getPatternRank (int32_t x_id)
			{
				return x_id2rank[x_id];
			}

			bool isPattern (int32_t x_id)
			{
				return x_id2rank[x_id] != 0;
			}

			LaRankPattern & getPattern (int32_t x_id)
			{
				uint32_t rank = x_id2rank[x_id];
				return patterns[rank];
			}

			uint32_t maxcount () const
			{
				return patterns.size ();
			}

			LaRankPattern & operator [] (uint32_t i)
			{
				return patterns[i];
			}

			const LaRankPattern & operator [] (uint32_t i) const
			{
				return patterns[i];
			}

		private:
			std_hash_set < uint32_t >freeidx;
			std::vector < LaRankPattern > patterns;
			std_hash_map < int32_t, uint32_t >x_id2rank;
	};


#endif // DOXYGEN_SHOULD_SKIP_THIS


	/** @brief the LaRank multiclass SVM machine
	 *
	 */
	class CLaRank:  public CMulticlassSVM
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

			virtual ~CLaRank ();

			// LEARNING FUNCTION: add new patterns and run optimization steps
			// selected with adaptative schedule
			/** add
			 * @param x_id
			 * @param yi
			 */
			virtual int32_t add (int32_t x_id, int32_t yi);

			// PREDICTION FUNCTION: main function in la_rank_classify
			/** predict
			 * @param x_id
			 */
			virtual int32_t predict (int32_t x_id);

			/** destroy */
			virtual void destroy ();

			// Compute Duality gap (costly but used in stopping criteria in batch mode)
			/** computeGap */
			virtual float64_t computeGap ();

			// Nuber of classes so far
			/** get num outputs */
			virtual uint32_t getNumOutputs () const;

			// Number of Support Vectors
			/** get NSV */
			int32_t getNSV ();

			// Norm of the parameters vector
			/** compute W2 */
			float64_t computeW2 ();

			// Compute Dual objective value
			/** get Dual */
			float64_t getDual ();

			/** get classifier type
			 *
			 * @return classifier type LIBSVM
			 */
			virtual EMachineType get_classifier_type() { return CT_LARANK; }

			/** @return object name */
			virtual const char* get_name() const { return "LaRank"; }

			/** set batch mode
			 * @param enable
			 */
			void set_batch_mode(bool enable) { batch_mode=enable; };
			/** get batch mode */
			bool get_batch_mode() { return batch_mode; };
			/** set tau
			 * @param t
			 */
			void set_tau(float64_t t) { tau=t; };
			/** get tau
			 * @return tau
			 */
			float64_t get_tau() { return tau; };

		protected:
			/** train machine */
			bool train_machine(CFeatures* data);

		private:
			/*
			 ** MAIN DARK OPTIMIZATION PROCESSES
			 */

			// Hash Table used to store the different outputs
			/** output hash */
			typedef std_hash_map < int32_t, LaRankOutput > outputhash_t;	// class index -> LaRankOutput

			/** outputs */
			outputhash_t outputs;

			LaRankOutput *getOutput (int32_t index);

			//
			LaRankPatterns patterns;

			// Parameters
			int32_t nb_seen_examples;
			int32_t nb_removed;

			// Numbers of each operation performed so far
			int32_t n_pro;
			int32_t n_rep;
			int32_t n_opt;

			// Running estimates for each operations
			float64_t w_pro;
			float64_t w_rep;
			float64_t w_opt;

			int32_t y0;
			float64_t m_dual;

			struct outputgradient_t
			{
				outputgradient_t (int32_t result_output, float64_t result_gradient)
					: output (result_output), gradient (result_gradient) {}
				outputgradient_t ()
					: output (0), gradient (0) {}

				int32_t output;
				float64_t gradient;

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
				process_return_t (float64_t dual, int32_t yprediction)
					: dual_increase (dual), ypred (yprediction) {}
				process_return_t () {}
				float64_t dual_increase;
				int32_t ypred;
			};

			// IMPORTANT Main SMO optimization step
			process_return_t process (const LaRankPattern & pattern, process_type ptype);

			// ProcessOld
			float64_t reprocess ();

			// Optimize
			float64_t optimize ();

			// remove patterns and return the number of patterns that were removed
			uint32_t cleanup ();

		protected:

			/// classes
			std_hash_set < int32_t >classes;

			/// class count
			inline uint32_t class_count () const
			{
				return classes.size ();
			}

			/// tau
			float64_t tau;

			/// nb train
			int32_t nb_train;
			/// cache
			int64_t cache;
			/// whether to use online learning or batch training
			bool batch_mode;

			/// progess output
			int32_t step;
	};
}
#endif // LARANK_H
