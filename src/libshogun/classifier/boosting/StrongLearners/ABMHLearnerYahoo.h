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
* \file ABMHLearnerYahoo.h The meta-learner AdaBoostLearner.MH.
*/
#pragma warning( disable : 4786 )

#ifndef __ABMH_LEARNER_YAHOO_H
#define __ABMH_LEARNER_YAHOO_H

#include "classifier/boosting/StrongLearners/GenericStrongLearner.h"
#include "classifier/boosting/StrongLearners/AdaBoostMHLearner.h"

#include "classifier/boosting/Utils/Args.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace shogun {

class OutputInfo;
class BaseLearner;
class InputData;
class Serialization;

/**
* The AdaBoost learner. This class performs the meta-learning
* by calling the weak learners and updating the weights.
* \date 12/11/2005
*/
class ABMHLearnerYahoo : public AdaBoostMHLearner
{
public:

   /**
   * The constructor. It initializes the variables and sets them using the
   * information provided by the arguments passed. They are parsed
   * using the helpers provided by class Args.
   * \date 13/11/2005
   */
   ABMHLearnerYahoo()
      : AdaBoostMHLearner() {}

   /**
   * Performs the classification using the ABMHClassifierYahoo.
   * \param args The arguments provided by the command line with all
   * the options for classification.
   */
   virtual void classify(const nor_utils::Args& args);

};

} // end of namespace shogun

#endif // __ADABOOST_MH_LEARNER_H
