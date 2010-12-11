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


#include "ParasiteLearner.h"
#include "ConstantLearner.h"

#include "classifier/boosting/IO/Serialization.h"
#include "classifier/boosting/Others/Example.h"

#include <math.h>
#include <limits>

namespace shogun {

//REGISTER_LEARNER_NAME(Parasite, ParasiteLearner)
REGISTER_LEARNER(ParasiteLearner)

// -----------------------------------------------------------------------

int ParasiteLearner::_numBaseLearners = -1;
string ParasiteLearner::_nameBaseLearnerFile = "";
vector<BaseLearner*> ParasiteLearner::_baseLearners;

// -----------------------------------------------------------------------

void ParasiteLearner::declareArguments(nor_utils::Args& args)
{
   BaseLearner::declareArguments(args);

   args.declareArgument("pool", 
                        "The name of the shyp file containing the pool of\n"
                        "  weak learners, followed by the number of desired\n"
                        "  weak learners. If -1 or more than the number of \n"
                        "  weak learners, we use all of them",
                        2, "<fileName> <nBaseLearners>");
         
   args.declareArgument("closed", "Include negatives of weak learners (default = false).");

}

// ------------------------------------------------------------------------------

void ParasiteLearner::initLearningOptions(const nor_utils::Args& args)
{
   BaseLearner::initLearningOptions(args);

   args.getValue("pool", 0, _nameBaseLearnerFile);   
   args.getValue("pool", 1, _numBaseLearners);   

   if ( args.hasArgument("closed") )
      _closed = 1;
}

// ------------------------------------------------------------------------------

float ParasiteLearner::classify(InputData* pData, int idx, int classIdx)
{
    return _signOfAlpha * 
	_baseLearners[_selectedIdx]->classify( pData, idx, classIdx );
}

// ------------------------------------------------------------------------------

float ParasiteLearner::run()
{
   if (_baseLearners.size() == 0) {
      // load the base learners
      if (_verbose >= 2)
	 cout << "loading " << _nameBaseLearnerFile << ".." << flush;
      UnSerialization us;
      us.loadHypotheses( _nameBaseLearnerFile, _baseLearners, _pTrainingData, _verbose);
      if (_verbose >= 2)
	 cout << "finished " << endl << flush;
   }
   
   if ( _numBaseLearners == -1 || _numBaseLearners > _baseLearners.size())
      _numBaseLearners = _baseLearners.size();
   
   
   const int numClasses = _pTrainingData->getNumClasses();
   const int numExamples = _pTrainingData->getNumExamples();
   float tmpAlpha;
   float bestE = numeric_limits<float>::max();
   float sumGamma, bestSumGamma = -numeric_limits<float>::max();
   float tmpE, gamma;
   float eps_min,eps_pls;
   int tmpSignOfAlpha;

   // This is the bottleneck, squeeze out every microsecond
   if (_closed) {
      bestSumGamma = 0;
      if ( nor_utils::is_zero(_theta) ) {
	 for (int j = 0; j < _numBaseLearners; ++j) {
	    sumGamma = 0;
	    for (int i = 0; i < numExamples; ++i) {
	       vector<Label> labels = _pTrainingData->getLabels(i);
	       for (int l = 0; l < numClasses; ++l)
		  sumGamma += labels[l].weight * 
		     _baseLearners[j]->classify(_pTrainingData,i,l) * labels[l].y;
	    }
	    if (fabs(sumGamma) > fabs(bestSumGamma)) {
	       _selectedIdx = j;
	       bestSumGamma = sumGamma;
	    }
	 }
	 eps_pls = eps_min = 0;
	 for (int i = 0; i < numExamples; ++i) {
	    vector<Label> labels = _pTrainingData->getLabels(i);
	    for (int l = 0; l < numClasses; ++l) {
	       gamma = _baseLearners[_selectedIdx]->classify(_pTrainingData,i,l) *
		  labels[l].y;
	       if ( gamma > 0 )
		  eps_pls += labels[l].weight;
	       else if ( gamma < 0 )
		  eps_min += labels[l].weight;
	    }
	 }
	 if (eps_min > eps_pls) {
	    float tmpSwap = eps_min;
	    eps_min = eps_pls;
	    eps_pls = tmpSwap;
	    _signOfAlpha = -1;
	 }
	 _alpha = getAlpha(eps_min, eps_pls);
	 bestE = BaseLearner::getEnergy( eps_min, eps_pls );
      }
      else {
	 for (int j = 0; j < _numBaseLearners; ++j) {
	    eps_pls = eps_min = 0;
	    for (int i = 0; i < numExamples; ++i) {
	       vector<Label> labels = _pTrainingData->getLabels(i);
	       for (int l = 0; l < numClasses; ++l) {
		  gamma = _baseLearners[j]->classify(_pTrainingData,i,l) * labels[l].y;
		  if ( gamma > 0 )
		     eps_pls += labels[l].weight;
		  else if ( gamma < 0 )
		     eps_min += labels[l].weight;
	       }
	    }
	    if (eps_min > eps_pls) {
	       float tmpSwap = eps_min;
	       eps_min = eps_pls;
	       eps_pls = tmpSwap;
	       tmpSignOfAlpha = -1;
	    }
	    else
	       tmpSignOfAlpha = 1;
	    tmpAlpha = getAlpha(eps_min, eps_pls, _theta);
	    tmpE = BaseLearner::getEnergy( eps_min, eps_pls, tmpAlpha, _theta );
	    if (tmpE < bestE && eps_pls > eps_min + _theta) {
	       _alpha = tmpAlpha;
	       _selectedIdx = j;
	       _signOfAlpha = tmpSignOfAlpha;
	       bestE = tmpE;
	    }
	 }
      }
   }
   else {
      if ( nor_utils::is_zero(_theta) ) {
	 for (int j = 0; j < _numBaseLearners; ++j) {
	    sumGamma = 0;
	    for (int i = 0; i < numExamples; ++i) {
	       vector<Label> labels = _pTrainingData->getLabels(i);
	       for (int l = 0; l < numClasses; ++l)
		  sumGamma += labels[l].weight * 
		     _baseLearners[j]->classify(_pTrainingData,i,l) * labels[l].y;
	    }
	    if (sumGamma > bestSumGamma) {
	       _selectedIdx = j;
	       bestSumGamma = sumGamma;
	    }
	 }
	 eps_pls = eps_min = 0;
	 for (int i = 0; i < numExamples; ++i) {
	    vector<Label> labels = _pTrainingData->getLabels(i);
	    for (int l = 0; l < numClasses; ++l) {
	       gamma = _baseLearners[_selectedIdx]->classify(_pTrainingData,i,l) *
		  labels[l].y;
	       if ( gamma > 0 )
		  eps_pls += labels[l].weight;
	       else if ( gamma < 0 )
		  eps_min += labels[l].weight;
	    }
	 }
	 _alpha = getAlpha(eps_min, eps_pls);
	 bestE = BaseLearner::getEnergy( eps_min, eps_pls );
      }
      else {
	 for (int j = 0; j < _numBaseLearners; ++j) {
	    eps_pls = eps_min = 0;
	    for (int i = 0; i < numExamples; ++i) {
	       vector<Label> labels = _pTrainingData->getLabels(i);
	       for (int l = 0; l < numClasses; ++l) {
		  gamma = _baseLearners[j]->classify(_pTrainingData,i,l) * labels[l].y;
		  if ( gamma > 0 )
		     eps_pls += labels[l].weight;
		  else if ( gamma < 0 )
		     eps_min += labels[l].weight;
	       }
	    }
	    tmpAlpha = getAlpha(eps_min, eps_pls, _theta);
	    tmpE = BaseLearner::getEnergy( eps_min, eps_pls, tmpAlpha, _theta );
	    if (tmpE < bestE && eps_pls > eps_min + _theta) {
	       _alpha = tmpAlpha;
	       _selectedIdx = j;
	       bestE = tmpE;
	    }
	    //cout << j << ": e- = " << eps_min << "\t e+ = " << eps_pls << "\t edge = " << (eps_pls - eps_min) << "\t energy = " << tmpE << "\t energy* = " << bestE << "\t alpha = " << tmpAlpha << endl << flush;
	 }
      }
   }
   return bestE;
   
}

// -----------------------------------------------------------------------

void ParasiteLearner::save(ofstream& outputStream, int numTabs)
{
   // Calling the super-class method
   BaseLearner::save(outputStream, numTabs);

   // we'll have to save the name of the pool file since it has to be loaded in the first
   // resume iteration, before run() is called
   outputStream << Serialization::standardTag("alphasign", _signOfAlpha, numTabs) << endl;
   outputStream << Serialization::standardTag("poolfile", _nameBaseLearnerFile, numTabs) << endl;
   outputStream << Serialization::standardTag("learneridx", _selectedIdx, numTabs) << endl;
}

// -----------------------------------------------------------------------

void ParasiteLearner::load(nor_utils::StreamTokenizer& st)
{
   //   cout << "Sorry, you can't load a ParasiteLearner" << endl << flush;
   //   exit(1);
   // Calling the super-class method
   BaseLearner::load(st);

   _signOfAlpha = UnSerialization::seekAndParseEnclosedValue<int>(st, "alphasign");
   _nameBaseLearnerFile = UnSerialization::seekAndParseEnclosedValue<string>(st, "poolfile");
   _selectedIdx = UnSerialization::seekAndParseEnclosedValue<int>(st, "learneridx");

   if (_baseLearners.size() == 0) {
      // load the base learners
      if (_verbose >= 2)
	 cout << "loading " << _nameBaseLearnerFile << ".." << flush;
      UnSerialization us;
      us.loadHypotheses( _nameBaseLearnerFile, _baseLearners, _pTrainingData, _verbose);
      if (_verbose >= 2)
	 cout << "finished " << endl << flush;
   }
}

// -----------------------------------------------------------------------

void ParasiteLearner::subCopyState(BaseLearner *pBaseLearner)
{
   BaseLearner::subCopyState(pBaseLearner);

   ParasiteLearner* pParasiteLearner =
      dynamic_cast<ParasiteLearner*>(pBaseLearner);

   pParasiteLearner->_signOfAlpha = _signOfAlpha;
   pParasiteLearner->_nameBaseLearnerFile = _nameBaseLearnerFile;
   pParasiteLearner->_selectedIdx = _selectedIdx;
}

// -----------------------------------------------------------------------

} // end of namespace shogun
