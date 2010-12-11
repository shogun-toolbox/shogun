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


#include "UCB1SingleStumpLearner.h"

#include "classifier/boosting/IO/Serialization.h"
#include "classifier/boosting/IO/SortedData.h"
#include "classifier/boosting/Algorithms/StumpAlgorithm.h"
#include "classifier/boosting/Algorithms/ConstantAlgorithm.h"

#include <limits> // for numeric_limits<>
#include <sstream> // for _id
#include <math.h> //for log

namespace MultiBoost {

//REGISTER_LEARNER_NAME(SingleStump, UCB1SingleStumpLearner)
REGISTER_LEARNER(UCB1SingleStumpLearner)

vector< int > UCB1SingleStumpLearner::_T; // the number of a feature has been selected 
int UCB1SingleStumpLearner::_numOfCalling = 0; //number of the single stump learner had been called
vector< float > UCB1SingleStumpLearner::_X; // the everage reward of a feature

//-------------------------------------------------------------------------------

void UCB1SingleStumpLearner::init() {
	UCB1SingleStumpLearner::_numOfCalling = 1;
	UCB1SingleStumpLearner::_T.resize( _pTrainingData->getNumAttributes() );
	UCB1SingleStumpLearner::_X.resize( _pTrainingData->getNumAttributes() );

   const int numClasses = _pTrainingData->getNumClasses();
   const int numColumns = _pTrainingData->getNumAttributes();

   // set the smoothing value to avoid numerical problem
   // when theta=0.
   setSmoothingVal( 1.0 / (float)_pTrainingData->getNumExamples() * 0.01 );

   vector<sRates> mu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.
   vector<float> tmpV(numClasses); // The class-wise votes/abstentions

   float tmpThreshold;

   float bestEnergy = numeric_limits<float>::max();

   StumpAlgorithm<float> sAlgo(numClasses);
   sAlgo.initSearchLoop(_pTrainingData);
   
   float halfTheta;
   if ( _abstention == ABST_REAL || _abstention == ABST_CLASSWISE )
      halfTheta = _theta/2.0;
   else
      halfTheta = 0;

   for (int j = 0; j < numColumns; ++j)
   {
     const pair<vpIterator,vpIterator> dataBeginEnd = 
		 static_cast<SortedData*>(_pTrainingData)->getFileteredBeginEnd(j);
	 

     const vpIterator dataBegin = dataBeginEnd.first;
     const vpIterator dataEnd = dataBeginEnd.second;

     // also sets mu, tmpV, and bestHalfEdge
     tmpThreshold = sAlgo.findSingleThresholdWithInit(dataBegin, dataEnd, _pTrainingData, 
                                                      halfTheta, &mu, &tmpV);

     if (tmpThreshold == tmpThreshold) // tricky way to test Nan
     { 
		float edge = 0.0;
		for( vector<sRates>::iterator itR = mu.begin(); itR != mu.end(); itR++ ) edge += ( itR->rPls - itR->rMin ); 
		
		UCB1SingleStumpLearner::_T[j] = 1;
		UCB1SingleStumpLearner::_X[j] = edge * edge;
     } // tmpThreshold == tmpThreshold
   }
}


// ------------------------------------------------------------------------------

float UCB1SingleStumpLearner::run()
{
   if ( UCB1SingleStumpLearner::_numOfCalling == 0 ) {
		init();
   }

	~UCB1SingleStumpLearner::_numOfCalling++; 

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

   StumpAlgorithm<float> sAlgo(numClasses);
   sAlgo.initSearchLoop(_pTrainingData);
   
   float halfTheta;
   if ( _abstention == ABST_REAL || _abstention == ABST_CLASSWISE )
      halfTheta = _theta/2.0;
   else
      halfTheta = 0;


   //chose an index accroding to the UCB1 policy
   int UCB1columnIndex = 0;
   float maxReward = numeric_limits<float>::min();
   float tmpReward = 0.0;
   for (int j = 0; j < numColumns; ++j) {
	   tmpReward = UCB1SingleStumpLearner::_X[j] / (float) UCB1SingleStumpLearner::_T[j];
	   tmpReward += sqrt( ( 2 * log( (float)UCB1SingleStumpLearner::_numOfCalling ) ) / UCB1SingleStumpLearner::_T[j] );

	   if ( tmpReward > maxReward ) {
		   maxReward = tmpReward;
		   UCB1columnIndex = j;
	   }
   }



     const pair<vpIterator,vpIterator> dataBeginEnd = 
		 static_cast<SortedData*>(_pTrainingData)->getFileteredBeginEnd(UCB1columnIndex);
	 

     const vpIterator dataBegin = dataBeginEnd.first;
     const vpIterator dataEnd = dataBeginEnd.second;

     // also sets mu, tmpV, and bestHalfEdge
     tmpThreshold = sAlgo.findSingleThresholdWithInit(dataBegin, dataEnd, _pTrainingData, 
                                                      halfTheta, &mu, &tmpV);

    // small inconsistency compared to the standard algo (but a good
    // trade-off): in findThreshold we maximize the edge (suboptimal but
    // fast) but here (among dimensions) we minimize the energy.
    bestEnergy = getEnergy(mu, tmpAlpha, tmpV);


   _alpha = tmpAlpha;
   _v = tmpV;
   _selectedColumn = UCB1columnIndex;
   _threshold = tmpThreshold;


	float edge = 0.0;
	for( vector<sRates>::iterator itR = mu.begin(); itR != mu.end(); itR++ ) edge += ( itR->rPls - itR->rMin ); 

	UCB1SingleStumpLearner::_X[UCB1columnIndex] += ( edge * edge );
	UCB1SingleStumpLearner::_T[UCB1columnIndex]++;

	cout << "Column to be selected: " << UCB1columnIndex << endl;

   stringstream thresholdString;
   thresholdString << _threshold;
   _id = _pTrainingData->getAttributeNameMap().getNameFromIdx(_selectedColumn) + thresholdString.str();

   
   return bestEnergy;
   
}

// ------------------------------------------------------------------------------

float UCB1SingleStumpLearner::phi(float val, int /*classIdx*/) const
{
   if (val > _threshold)
      return +1;
   else
      return -1;
}

// ------------------------------------------------------------------------------

float UCB1SingleStumpLearner::phi(InputData* pData,int pointIdx) const
{
   return phi(pData->getValue(pointIdx,_selectedColumn),0);
}

// -----------------------------------------------------------------------

void UCB1SingleStumpLearner::save(ofstream& outputStream, int numTabs)
{
   // Calling the super-class method
   FeaturewiseLearner::save(outputStream, numTabs);

   // save selectedCoulumn
   outputStream << Serialization::standardTag("threshold", _threshold, numTabs) << endl;
   
}

// -----------------------------------------------------------------------

void UCB1SingleStumpLearner::load(nor_utils::StreamTokenizer& st)
{
   // Calling the super-class method
   FeaturewiseLearner::load(st);

   _threshold = UnSerialization::seekAndParseEnclosedValue<float>(st, "threshold");

   stringstream thresholdString;
   thresholdString << _threshold;
   _id = _id + thresholdString.str();
}

// -----------------------------------------------------------------------

void UCB1SingleStumpLearner::subCopyState(BaseLearner *pBaseLearner)
{
   FeaturewiseLearner::subCopyState(pBaseLearner);

   UCB1SingleStumpLearner* pUCB1SingleStumpLearner =
      dynamic_cast<UCB1SingleStumpLearner*>(pBaseLearner);

   pUCB1SingleStumpLearner->_threshold = _threshold;
}

// -----------------------------------------------------------------------

//void UCB1SingleStumpLearner::getStateData( vector<float>& data, const string& /*reason*/, InputData* pData )
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
//      data[pos++] = UCB1SingleStumpLearner::phi( pData->getValue( i, _selectedColumn), 0 );
//}

// -----------------------------------------------------------------------

} // end of namespace MultiBoost
