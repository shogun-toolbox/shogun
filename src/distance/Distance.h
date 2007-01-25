/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Written (W) 2006 Soeren Sonnenburg
 * Copyright (C) 2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DISTANCE_H___
#define _DISTANCE_H___

#include "lib/common.h"
#include "lib/Mathematics.h"
#include "base/SGObject.h"
#include "features/Features.h"

#include <stdio.h>

class CDistance : public CSGObject
{
	public:
		CDistance();
		CDistance(CFeatures* lhs, CFeatures* rhs);
		virtual ~CDistance();

		/** get distance function for lhs feature vector a 
		  and rhs feature vector b
		 */
		inline DREAL distance(INT idx_a, INT idx_b)
		{
			if (idx_a < 0 || idx_b <0)
				return 0;

			if (lhs==rhs)
			{
				int num_vectors = lhs->get_num_vectors();

				if (idx_a>=num_vectors)
					idx_a=2*num_vectors-1-idx_a;

				if (idx_b>=num_vectors)
					idx_b=2*num_vectors-1-idx_b;
			}

			if (precompute_matrix && (precomputed_matrix==NULL) && (lhs==rhs))
				do_precompute_matrix() ;

			if (precompute_matrix && (precomputed_matrix!=NULL))
			{
				if (idx_a>=idx_b)
					return precomputed_matrix[idx_a*(idx_a+1)/2+idx_b] ;
				else
					return precomputed_matrix[idx_b*(idx_b+1)/2+idx_a] ;
			}

			return compute(idx_a, idx_b);
		}
		
		void get_distance_matrix(DREAL** dst,INT* m, INT* n);

		virtual DREAL* get_distance_matrix_real(int &m,int &n, DREAL* target);

		virtual SHORTREAL* get_distance_matrix_shortreal(int &m,int &n,SHORTREAL* target);

		/** initialize distance cache
		 *  make sure to check that your distance can deal with the
		 *  supplied features (!)
		*/
		virtual bool init(CFeatures* lhs, CFeatures* rhs);

		/// clean up your kernel
		virtual void cleanup()=0;

		/// load and save the distance matrix
		bool load(CHAR* fname);
		bool save(CHAR* fname);

		/// load and save distance init_data
		virtual bool load_init(FILE* src)=0;
		virtual bool save_init(FILE* dest)=0;
		
		/// get left/right hand side of features used in distance matrix
		inline CFeatures* get_lhs() { return lhs; } ;
		inline CFeatures* get_rhs() { return rhs;  } ;

		/// takes all necessary steps if the lhs is removed from distance matrix
		virtual void remove_lhs();

		/// takes all necessary steps if the rhs is removed from distance matrix
		virtual void remove_rhs();
		
		// return what type of distance we are using
		virtual EDistanceType get_distance_type()=0 ;

		/** return feature type the distance can deal with
		  */
		virtual EFeatureType get_feature_type()=0;

		/** return feature class the distance can deal with
		  */
		virtual EFeatureClass get_feature_class()=0;

		// return the name of a distance
		virtual const CHAR* get_name()=0 ;


		//fixme: precompute matrix should be dropped, handling should be via customdistance
		inline bool get_precompute_matrix() { return precompute_matrix ;  }
		
		inline virtual void set_precompute_matrix(bool flag)
		{ 
			precompute_matrix = flag ; 
		
			if (!precompute_matrix)
			{
				delete[] precomputed_matrix ;
				precomputed_matrix = NULL ;
			}
		}

		inline DREAL get_max(DREAL a,DREAL b)
		{
			if(a>b)
			 return a;
			return b;
		}

	protected:
		/// compute distance function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual DREAL compute(INT x, INT y)=0;

		/// matrix precomputation
		void do_precompute_matrix() ;

	protected:
		
		SHORTREAL * precomputed_matrix ;

		bool precompute_matrix ;

		/// feature vectors to occur on left hand side
		CFeatures* lhs;
		/// feature vectors to occur on right hand side
		CFeatures* rhs;

};
#endif
