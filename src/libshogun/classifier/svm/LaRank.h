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
#include "classifier/svm/kcache.h"
#include "classifier/svm/SVM.h"

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
	 ** MACHINE: the main thing, which is trained.
	 */
	class Machine : public CSVM
	{
		public:
			virtual ~ Machine () {};
			virtual void destroy () = 0;

			// MAIN functions for straining and testing      
			virtual int add (int x_id, int classnumber) = 0;
			virtual int predict (int x_id) = 0;

			// Information functions
			virtual void printStuff (double initime, bool print_dual) = 0;
			virtual double computeGap () = 0;

			std_hash_set < int >classes;

			unsigned class_count () const
			{
				return classes.size ();
			}

			double C;
			double tau;
			int nb_train;
			long cache;
			larank_kernel_t kfunc;

	};

	/*
	 ** OUTPUT: one per class of the raining set, keep tracks of support vectors and their beta coefficients
	 */
	class LaRankOutput
	{
		public:
			LaRankOutput () {}
			~LaRankOutput () {}

			// Initializing an output class (basically creating a kernel cache for it)
			void initialize (larank_kernel_t kfunc, long cache);

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
			inline larank_kcache_t *getKernel ();

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

	class CLaRank:public Machine
	{
		public:
			CLaRank ();
			~CLaRank () {}

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
