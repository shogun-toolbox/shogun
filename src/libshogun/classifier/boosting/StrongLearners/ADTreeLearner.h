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


#ifndef __ADTREE_LEARNER_H
#define __ADTREE_LEARNER_H

#include "classifier/boosting/Utils/Args.h"
#include "classifier/boosting/StrongLearners/GenericStrongLearner.h"

#include <vector>
#include <fstream>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {

class BaseLearner;
class ADTreeWeakLearner;
class InputData;

/**
* ADTree learner.
* \date 20/3/2006
*/
class ADTreeLearner : public GenericStrongLearner
{
public:

   ADTreeLearner() 
      : _smallVal(1E-10) {}

   virtual void run(const nor_utils::Args& args);

   virtual void classify(const nor_utils::Args& /*args*/) {}

   virtual void doConfusionMatrix(const nor_utils::Args& /*args*/) {}

   virtual void doPosteriors(const nor_utils::Args& /*args*/) {}

protected:

   struct ADTreeArgs 
   {
      string trainFileName;
      string testFileName;

      string baseLearnerName;
      string shypFileName;

      string outputInfoFile;
      string resumeShypFileName;

      double theta;
      int    numIterations;
      int    maxTime;

      int    verbose;
   };

   ADTreeArgs getArgs(const nor_utils::Args& args);

   double     updateWeights(InputData* pTrainData, BaseLearner* pWeakHypothesis);

private:

   // TEMP here!!
   void createGraphvizDot(const string& outFileName, BaseLearner* pTreeHead, /*TEMP*/InputData* pData );
   void recursivePrintDotNode(ostream& dotFile, ADTreeWeakLearner* node, /*TEMP*/InputData* pData, bool isHead = false);

   vector<BaseLearner*>  _foundRules; //!< The list of the rules found.
   const double _smallVal; //!< A small value, to solve numeric issues

   /**
   * Fake assignment operator to avoid warning.
   * \date 6/12/2005
   */
   ADTreeLearner& operator=( const ADTreeLearner& ) {return *this;}

   BaseLearner* _pTreeHead; //!< The head of the ADTree. 
   vector< vector<double> > _ry;

};

} // end of namespace MultiBoost

#endif // __ADTREE_LEARNER_H
