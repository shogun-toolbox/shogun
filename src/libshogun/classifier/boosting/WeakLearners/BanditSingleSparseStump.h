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
* \file BanditSingleSparseStump.h A single threshold decision stump learner. 
*/

#ifndef __BANDIT_SINGLE_SPARSE_STUMP_H
#define __BANDIT_SINGLE_SPARSE_STUMP_H

//#include "WeakLearners/ClasswiseLearner.h"
#include "classifier/boosting/WeakLearners/FeaturewiseLearner.h"
#include "classifier/boosting/WeakLearners/BanditSingleStumpLearner.h"
#include "classifier/boosting/Utils/Args.h"
#include "classifier/boosting/IO/InputData.h"
#include "classifier/boosting/IO/SortedData.h"
#include "classifier/boosting/Bandits/Exp3G2.h"

#include <vector>
#include <fstream>
#include <cassert>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace shogun {

/**
* A \b single threshold decision stump learner. 
* There is ONE and ONE ONLY threshold here.
*/
class BanditSingleSparseStump : public BanditSingleStumpLearner
{
public:
	BanditSingleSparseStump() : BanditSingleStumpLearner() {}

   /**
   * The destructor. Must be declared (virtual) for the proper destruction of 
   * the object.
   */
   virtual ~BanditSingleSparseStump() {}

   /**
   */
   virtual void init();


   /**
   * Returns itself as object.
   * \remark It uses the trick described in http://www.parashift.com/c++-faq-lite/serialization.html#faq-36.8
   * for the auto-registering classes.
   * \date 14/11/2005
   */
   virtual BaseLearner* subCreate() { 
	   BaseLearner* retLearner = new BanditSingleSparseStump();
	   static_cast< BanditSingleStumpLearner* >(retLearner)->setBanditAlgoObject( _banditAlgo );
	   return retLearner;
   }

   /**
   * Run the learner to build the classifier on the given data.
   * \param pData The pointer to the data.
   * \see BaseLearner::run
   * \date 11/11/2005
   */
   virtual float run();

   virtual float run( int colIdx );

protected:

};

//////////////////////////////////////////////////////////////////////////

} // end of namespace shogun

#endif // __SINGLE_STUMP_LEARNER_H
