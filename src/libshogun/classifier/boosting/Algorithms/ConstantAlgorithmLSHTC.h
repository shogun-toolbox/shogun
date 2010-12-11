/*
*
*    MultiBoost - Multi-purpose boosting package
*
*    Copyright (C) 2010   AppStat group
*                         Laboratoire de l'Accelerateur Lineaire
*                         Universite Paris-Sud, 11, CNRS
*
*    This file is part of the MultiBoost library
*
*    This library is free software; you can redistribute it 
*    and/or modify it under the terms of the GNU General Public
*    License as published by the Free Software Foundation; either
*    version 2.1 of the License, or (at your option) any later version.
*
*    This library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*    General Public License for more details.
*
*    You should have received a copy of the GNU General Public
*    License along with this library; if not, write to the Free Software
*    Foundation, Inc., 51 Franklin St, 5th Floor, Boston, MA 02110-1301 USA
*
*    Contact: Balazs Kegl (balazs.kegl@gmail.com)
*             Norman Casagrande (nova77@gmail.com)
*             Robert Busa-Fekete (busarobi@gmail.com)
*
*    For more information and up-to-date version, please visit
*        
*                       http://www.multiboost.org/
*
*/


#ifndef __CONSTANT_ALGORITHM_LSHTC_H
#define __CONSTANT_ALGORITHM_LSHTC_H

#include <vector>
#include <cassert>

#include "classifier/boosting/IO/InputData.h"
#include "classifier/boosting/Others/Rates.h"

using namespace std;

namespace shogun {

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

class ConstantAlgorithmLSHTC
{
public:

   /**
   * Computes the half of the classwise weights and half of the
   * classwise edges.
   * @param pData The pointer to the data.
   * @param pHalfWeightsPerClass The pointer to the half of the classwise weights.
   * @param pHalfEdges The pointer to the half of the classwise edges.
   * @date 30/06/2006
   */
   void findConstantWeightsEdges( InputData* pData, 
                                  vector<float>& pHalfWeightsPerClass,
                                  vector<float>& pHalfEdges);
   /**
   * Computes the classwise mus and votes (alignment) of the constant classifier
   * @param pData The pointer to the data.
   * @param pMu The The class-wise rates to update.
   * @param pV The alignment vector to update.
   * @return The edge.
   * @date 20/07/2006
   */
   float findConstant( InputData* pData, 
                        vector<sRates>* pMu, vector<float>* pV);
};

} // end of namespace shogun

#endif // __CONSTANT_ALGORITHM_H
