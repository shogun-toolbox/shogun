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
* \file ConstantLearner.h A single threshold decision stump learner. 
*/

#ifndef __CONSTANT_LEARNER_H
#define __CONSTANT_LEARNER_H

#include "classifier/boosting/WeakLearners/AbstainableLearner.h"
#include "classifier/boosting/Utils/Args.h"
#include "classifier/boosting/IO/InputData.h"

#include <vector>
#include <fstream>
#include <cassert>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {

/**
* A \b single threshold decision stump learner. 
* There is ONE and ONE ONLY threshold here.
*/
class ConstantLearner : public AbstainableLearner
{
public:

   /**
   * The destructor. Must be declared (virtual) for the proper destruction of 
   * the object.
   */
   virtual ~ConstantLearner() {}

   /**
   * Returns itself as object.
   * \remark It uses the trick described in http://www.parashift.com/c++-faq-lite/serialization.html#faq-36.8
   * for the auto-registering classes.
   * \date 14/11/2005
   */
   virtual BaseLearner* subCreate() { return new ConstantLearner(); }

   /**
   * Run the learner to build the classifier on the given data.
   * \param pData The pointer to the data.
   * \see BaseLearner::run
   * \date 11/11/2005
   */
   virtual float run();
   virtual float run( int colNum );

   /**
   * Returns a vector of float holding any data that the specific weak learner can generate
   * using the given input dataset. Right now just a single case is contemplated, therefore
   * "reason" is not used. The returned vector of data correspond to:
   * \f[
   * \left\{ v^{(t)}_1, v^{(t)}_2, \cdots, v^{(t)}_K, \phi^{(t)}({\bf x}_1), 
   * \phi^{(t)}({\bf x}_2), \cdots, \phi^{(t)}({\bf x}_n) \right\}
   * \f]
   * where \f$v\f$ is the alignment vector, \f$\phi\f$ is the discriminative function and
   * \f$t\f$ is the current iteration (to which this instantiation of weak learner belongs to).
   * \param data The vector with the returned data.
   * \param pData The pointer to the data points (the examples) previously loaded.
   * \remark This particular method has been created for analysis purposes (research). If you
   * need to get similar analytical information from this weak learner, add it to the function
   * and uncomment the parameter "reason".
   * \see BaseLearner::getStateData
   * \see Classifier::saveSingelStumpFeatureData
   * \date 10/2/2006
   */
   virtual void getStateData( vector<float>& data, const string& /*reason = ""*/, InputData* pData = 0 );

protected:

   /**
   * A discriminative function. 
   * \remarks Positive or negative do NOT refer to positive or negative classification.
   * This function is equivalent to the phi function in my thesis.
   * \param val The value to discriminate
   * \param classIdx The class index, used by MultiStumpLearner when phi depends on the class
   * \return +1 always 
   * \date 19/09/2006
   * \see classify
   */
   virtual float phi(InputData* pData, int idx, int classIdx) const { return 1; }
};

//////////////////////////////////////////////////////////////////////////

} // end of namespace MultiBoost

#endif // __CONSTANT_LEARNER_H
