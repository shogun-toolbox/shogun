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


#include "ConstantAlgorithm.h"

#include <vector>
#include <cassert>

#include "classifier/boosting/IO/InputData.h"
#include "classifier/boosting/Others/Rates.h"
#include "classifier/boosting/IO/NameMap.h"

//To be checked with sparse with BreastCancer

namespace MultiBoost {

// ------------------------------------------------------------------------------

void ConstantAlgorithm::findConstantWeightsEdges( InputData* pData,
						  vector<float>& halfWeightsPerClass,
						  vector<float>& halfEdges)
{ 
   const int numClasses = pData->getNumClasses();
   const int numExamples = pData->getNumExamples();

   memset(&(halfWeightsPerClass[0]), 0, sizeof(float) * halfWeightsPerClass.size() );
   memset(&(halfEdges[0]), 0, sizeof(float) * halfEdges.size() );
   //fill(halfWeightsPerClass.begin(), halfWeightsPerClass.end(), 0);
   //fill(halfEdges.begin(), halfEdges.end(), 0);

   for (int i = 0; i < numExamples; ++i)
   {
      vector<Label>& labels = pData->getLabels(i);
      vector<Label>::iterator lIt;

      int l = 0;
      for (lIt = labels.begin(); lIt != labels.end(); ++lIt, l )
      {
         halfWeightsPerClass[ lIt->idx ] += lIt->weight;
         halfEdges[ lIt->idx ] += lIt->weight * lIt->y;
      }
   }

  for (int l = 0; l < numClasses; ++l)
  {
      halfWeightsPerClass[l] /= 2.0;
      halfEdges[l] /= 2.0;
   }
} // end of findConstantWeightsEdges

// ------------------------------------------------------------------------------

float ConstantAlgorithm::findConstant(InputData* pData,
				       vector<sRates>* pMu, vector<float>* pV)
{ 
   const int numClasses = pData->getNumClasses();

   vector<float> halfWeightsPerClass(numClasses);
   vector<float> halfEdges(numClasses);
   
   findConstantWeightsEdges(pData, halfWeightsPerClass, halfEdges);
   
   float halfEdge = 0;

   for (int l = 0; l < numClasses; ++l)
   {
      if (halfEdges[l] > 0)
	 (*pV)[l] = +1;
      else
	 (*pV)[l] = -1;

      halfEdge += (*pV)[l] * halfEdges[l];

      (*pMu)[l].classIdx = l;
      
      (*pMu)[l].rPls  = halfWeightsPerClass[l] + (*pV)[l] * halfEdges[l];
      (*pMu)[l].rMin  = halfWeightsPerClass[l] - (*pV)[l] * halfEdges[l];
      (*pMu)[l].rZero = (*pMu)[l].rPls + (*pMu)[l].rMin; // == weightsPerClass[l]
   }

   return 2 * halfEdge;

} // end of findConstant

// ------------------------------------------------------------------------------

} // end of namespace MultiBoost
