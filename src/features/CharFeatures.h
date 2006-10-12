/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CHARFEATURES__H__
#define _CHARFEATURES__H__

#include "features/SimpleFeatures.h"
#include "features/Alphabet.h"
#include "lib/common.h"

class CCharFeatures: public CSimpleFeatures<CHAR>
{
	public:
		CCharFeatures(E_ALPHABET, INT size);
		CCharFeatures(CAlphabet* alpha, INT size);
		CCharFeatures(const CCharFeatures & orig);
        CCharFeatures(E_ALPHABET alphabet, CHAR* feature_matrix, INT num_feat, INT num_vec);
		CCharFeatures(E_ALPHABET alphabet, CHAR* fname);

		~CCharFeatures();
      
        void testAddParam(int alphabet, char* feature_matrix, int num_feat, int num_vec) {
            CIO::message(M_MESSAGEONLY,"d1:%d\n",alphabet);
        }
        
        void testAddParam2(int alphabet, double* feature_matrix, int num_feat, int num_vec) {
            CIO::message(M_MESSAGEONLY,"d1:%d\n",alphabet);
        }

        void testAddParam3(int alphabet, double* feature_matrix, int num_feat, int num_vec) {
            CIO::message(M_MESSAGEONLY,"d1:%d\n",alphabet);
        }
      void testNumpy(char *feature_matrix, int d1, int d2) {
         CIO::message(M_MESSAGEONLY,"d1:%d\n",d1);
         CIO::message(M_MESSAGEONLY,"d2:%d\n",d2);

		 for (int i=0; i<d1*d2; i++)
			 CIO::message(M_MESSAGEONLY,"%c",feature_matrix[i]);
      }

      void testNumpy2(int *feature_matrix, int d1, int d2) {
         CIO::message(M_MESSAGEONLY,"d1:%d\n",d1);
         CIO::message(M_MESSAGEONLY,"d2:%d\n",d2);
      }

      void testNumpy3(double*feature_matrix, int d1, int d2) {
         CIO::message(M_MESSAGEONLY,"d1:%d\n",d1);
         CIO::message(M_MESSAGEONLY,"d2:%d\n",d2);
      }

		double doubleSum(double* series, int size) {
			double result = 0.0;
			for (int i=0; i<size; ++i) result += series[i];
			return result;
		}

		inline CAlphabet* get_alphabet()
		{
			return alphabet;
		}

		virtual CFeatures* duplicate() const;
		virtual bool load(CHAR* fname);
		virtual bool save(CHAR* fname);
	protected:
		CAlphabet* alphabet;
};
#endif
