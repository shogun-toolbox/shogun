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


#include "SingleSparseStumpLearner.h"

#include "classifier/boosting/IO/Serialization.h"
#include "classifier/boosting/IO/SortedData.h"
#include "classifier/boosting/Algorithms/StumpAlgorithmLSHTC.h"
#include "classifier/boosting/Algorithms/ConstantAlgorithmLSHTC.h"

#include <limits> // for numeric_limits<>
#include <sstream> // for _id

namespace shogun {

//REGISTER_LEARNER_NAME(SingleStump, SingleSparseStumpLearner)
REGISTER_LEARNER(SingleSparseStumpLearner)

// ------------------------------------------------------------------------------

float SingleSparseStumpLearner::run()
{
   const int numClasses = _pTrainingData->getNumClasses();
   const int numColumns = _pTrainingData->getNumAttributes();

   // set the smoothing value to avoid numerical problem
   // when theta=0.
   setSmoothingVal( 1.0 / (float)_pTrainingData->getNumExamples() * 0.01 );

   vector<sRates> mu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.
   vector<float> tmpV(numClasses); // The class-wise votes/abstentions

   float tmpThreshold;
   float tmpAlpha;

   float bestEnergy = numeric_limits<float>::max();
   float tmpEnergy;

   StumpAlgorithmLSHTC<float> sAlgo(numClasses);
   sAlgo.initSearchLoop(_pTrainingData);
   
   float halfTheta;
   if ( _abstention == ABST_REAL || _abstention == ABST_CLASSWISE )
      halfTheta = _theta/2.0;
   else
      halfTheta = 0;

   int numOfDimensions = _maxNumOfDimensions;
   for (int j = 0; j < numColumns; ++j)
   {
      // Tricky way to select numOfDimensions columns randomly out of numColumns
      int rest = numColumns - j;
      float r = rand()/static_cast<float>(RAND_MAX);

      if ( static_cast<float>(numOfDimensions) / rest > r ) 
      {
         --numOfDimensions;
		 
		 //if ( static_cast<SortedData*>(_pTrainingData)->isAttributeEmpty( j ) ) continue;

         const pair<vpReverseIterator,vpReverseIterator> dataBeginEnd = 
			 static_cast<SortedData*>(_pTrainingData)->getFileteredReverseBeginEnd(j);
		 
		 //if ( static_cast<SortedData*>(_pTrainingData)->isFilteredAttributeEmpty() ) continue;
		 //if ( static_cast<SortedData*>(_pTrainingData)->isFilteredAttributeHasOneValue() ) continue;

         const vpReverseIterator dataBegin = dataBeginEnd.first;
         const vpReverseIterator dataEnd = dataBeginEnd.second;


		 
         // also sets mu, tmpV, and bestHalfEdge
         tmpThreshold = sAlgo.findSingleThresholdWithInit(dataBegin, dataEnd, _pTrainingData, 
                                                          halfTheta, &mu, &tmpV);

         if (tmpThreshold == tmpThreshold) // tricky way to test Nan
         { 
            // small inconsistency compared to the standard algo (but a good
            // trade-off): in findThreshold we maximize the edge (suboptimal but
            // fast) but here (among dimensions) we minimize the energy.
            tmpEnergy = getEnergy(mu, tmpAlpha, tmpV);

            if (tmpEnergy < bestEnergy && tmpAlpha > 0)
            {
               // Store it in the current weak hypothesis.
               // note: I don't really like having so many temp variables
               // but the alternative would be a structure, which would need
               // to be inheritable to make things more consistent. But this would
               // make it less flexible. Therefore, I am still undecided. This
               // might change!

               _alpha = tmpAlpha;
               _v = tmpV;
               _selectedColumn = j;
               _threshold = tmpThreshold;

               bestEnergy = tmpEnergy;
            }
         } // tmpThreshold == tmpThreshold
      }
   }

   stringstream thresholdString;
   thresholdString << _threshold;
   _id = _pTrainingData->getAttributeNameMap().getNameFromIdx(_selectedColumn) + thresholdString.str();

   
   return bestEnergy;
   
}

// ------------------------------------------------------------------------------

float SingleSparseStumpLearner::run( int colIdx )
{
   const int numClasses = _pTrainingData->getNumClasses();
   const int numColumns = _pTrainingData->getNumAttributes();

   // set the smoothing value to avoid numerical problem
   // when theta=0.
   setSmoothingVal( 1.0 / (float)_pTrainingData->getNumExamples() * 0.01 );

   vector<sRates> mu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.
   vector<float> tmpV(numClasses); // The class-wise votes/abstentions

   float tmpAlpha;

   float bestEnergy = numeric_limits<float>::max();

   StumpAlgorithmLSHTC<float> sAlgo(numClasses);
   sAlgo.initSearchLoop(_pTrainingData);
   
   float halfTheta;
   if ( _abstention == ABST_REAL || _abstention == ABST_CLASSWISE )
      halfTheta = _theta/2.0;
   else
      halfTheta = 0;

   int numOfDimensions = _maxNumOfDimensions;

   
   const pair<vpReverseIterator,vpReverseIterator> dataBeginEnd = 
		 static_cast<SortedData*>(_pTrainingData)->getFileteredReverseBeginEnd( colIdx );
	 

     const vpReverseIterator dataBegin = dataBeginEnd.first;
     const vpReverseIterator dataEnd = dataBeginEnd.second;

     // also sets mu, tmpV, and bestHalfEdge
     _threshold = sAlgo.findSingleThresholdWithInit(dataBegin, dataEnd, _pTrainingData, 
                                                      halfTheta, &mu, &tmpV);

    bestEnergy = getEnergy(mu, tmpAlpha, tmpV);

   _alpha = tmpAlpha;
   _v = tmpV;
   _selectedColumn = colIdx;
    
   stringstream thresholdString;
   thresholdString << _threshold;
   _id = _pTrainingData->getAttributeNameMap().getNameFromIdx(_selectedColumn) + thresholdString.str();

   
   return bestEnergy;
   
}



// ------------------------------------------------------------------------------

float SingleSparseStumpLearner::phi(float val, int /*classIdx*/) const
{
   if (val > _threshold)
      return +1;
   else
      return -1;
}

// ------------------------------------------------------------------------------

float SingleSparseStumpLearner::phi(InputData* pData,int pointIdx) const
{
   return phi(pData->getValue(pointIdx,_selectedColumn),0);
}

// -----------------------------------------------------------------------

void SingleSparseStumpLearner::save(ofstream& outputStream, int numTabs)
{
   // Calling the super-class method
   FeaturewiseLearner::save(outputStream, numTabs);

   // save selectedCoulumn
   outputStream << Serialization::standardTag("threshold", _threshold, numTabs) << endl;
   
}

// -----------------------------------------------------------------------

void SingleSparseStumpLearner::load(nor_utils::StreamTokenizer& st)
{
   // Calling the super-class method
   FeaturewiseLearner::load(st);

   _threshold = UnSerialization::seekAndParseEnclosedValue<float>(st, "threshold");

   stringstream thresholdString;
   thresholdString << _threshold;
   _id = _id + thresholdString.str();
}

// -----------------------------------------------------------------------

void SingleSparseStumpLearner::subCopyState(BaseLearner *pBaseLearner)
{
   FeaturewiseLearner::subCopyState(pBaseLearner);

   SingleSparseStumpLearner* pSingleSparseStumpLearner =
      dynamic_cast<SingleSparseStumpLearner*>(pBaseLearner);

   pSingleSparseStumpLearner->_threshold = _threshold;
}

// -----------------------------------------------------------------------

//void SingleSparseStumpLearner::getStateData( vector<float>& data, const string& /*reason*/, InputData* pData )
//{
//   const int numClasses = pData->getNumClasses();
//   const int numExamples = pData->getNumExamples();
//
//   // reason ignored for the moment as it is used for a single task
//   data.resize( numClasses + numExamples );
//
//   int pos = 0;
//
//   for (int l = 0; l < numClasses; ++l)
//      data[pos++] = _v[l];
//
//   for (int i = 0; i < numExamples; ++i)
//      data[pos++] = SingleSparseStumpLearner::phi( pData->getValue( i, _selectedColumn), 0 );
//}

// -----------------------------------------------------------------------

} // end of namespace shogun
