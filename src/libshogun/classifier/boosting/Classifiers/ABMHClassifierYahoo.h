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


/**
* \file ABMHClassifierYahoo.h Performs the classification with AdaBoostMH.
*/
#pragma warning( disable : 4786 )

#ifndef __ABMH_CLASSIFIER_YAHOO_H
#define __ABMH_CLASSIFIER_YAHOO_H

#include "classifier/boosting/Utils/Args.h"
#include "classifier/boosting/Classifiers/AdaBoostMHClassifier.h"

#include <string>
#include <cassert>

using namespace std;

namespace MultiBoost {

enum Scoring 
{
	EVAL_FOURTH_LABEL,
	EVAL_EXP_WEIGHT
};


//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

// Forward declarations.
class ExampleResults;
class InputData;
class BaseLearner;

/**
* Classify a dataset with AdaBoost.MH learner.
* Using the strong hypothesis file (shyp.xml by default) it builds the
* list of weak hypothesis (or weak learners), and use them to perform a classification over 
* the given data set. The strong hypothesis is the linear combination of the weak 
* hypotheses and their confidence alpha, and is defined as:
* \f[
* {\bf g}(x) = \sum_{t=1}^T \alpha^{(t)} {\bf h}^{(t)}(x),
* \f]
* where the bold defines a vector as returned value.
* To obtain a single class, we simply take the winning class that receives 
* the "most vote", that is:
* \f[
* f(x) = \mathop{\rm arg\, max}_{\ell} g_\ell(x).
* \f]
* \date 15/11/2005
*/
class ABMHClassifierYahoo : public AdaBoostMHClassifier
{
public:

   /**
   * The constructor. It initializes the variable and set them using the
   * information provided by the arguments passed. They are parsed
   * using the helpers provided by class Args.
   * \param args The arguments defined by the user in the command line.
   * \param verbose The level of verbosity
   * \see _verbose
   * \date 16/11/2005
   */
   ABMHClassifierYahoo(const nor_utils::Args& args, int verbose = 1);


   /**
   * Compute the results using the weak hypotheses.
   * This method is the one that effectively computes \f${\bf g}(x)\f$.
   * \param pData A pointer to the data to be classified.
   * \param weakHypotheses The list of weak hypotheses.
   * \param results The vector where the results will be stored.
   * \see ExampleResults
   * \date 16/11/2005
   */
   void computeResults(InputData* pData, vector<BaseLearner*>& weakHypotheses, 
                       vector< ExampleResults* >& results, vector< ExampleResults* >& normalizedResults, int numIterations );


   void run(const string& dataFileName, const string& shypFileName, 
            int numIterations, const string& outResFileName = "", 
	    int numRanksEnclosed = 2);

private:

   /**
   * Fake assignment operator to avoid warning.
   * \date 6/12/2005
   */
   ABMHClassifierYahoo& operator=( const ABMHClassifierYahoo& ) {return *this;}

   void getERR( vector<int>& l, vector<int>& r, float& err, float &ndcg, int k=10 );
   void normalizePredictedScores( vector<float>& r, vector<int>& ranks );
	/** 
	* Read query file
	**/
   void readQueries();

   string			_queryFile;
   vector<int>		_queryIDSBorders;
   vector<int>		_origLabels;
   vector<int>		_labels;

   Scoring			_scoring;
};

} // end of namespace MultiBoost

#endif // __ADABOOST_MH_CLASSIFIER_H
