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
* \file HaarSingleStumpLearner.h A learner that uses the Haar-like features for
* input and the SingleStumpLearner for training.
* \date 17/12/2005
*/

#ifndef __HAAR_SINGLE_STUMP_LEARNER_H
#define __HAAR_SINGLE_STUMP_LEARNER_H

#include "classifier/boosting/WeakLearners/Haar/HaarLearner.h"
#include "classifier/boosting/WeakLearners/SingleStumpLearner.h"

#include "classifier/boosting/Utils/Utils.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {

class HaarData;

/**
* A learner that uses the Haar-like features for input and the SingleStumpLearner 
* for training.
* \see HaarLearner
* \see SingleStumpLearner
* \date 17/12/2005
*/
class HaarSingleStumpLearner : public HaarLearner, public SingleStumpLearner
{
public:

   /**
   * The destructor. Must be declared (virtual) for the proper destruction of 
   * the object.
   */
   virtual ~HaarSingleStumpLearner() {}

   /**
   * Declare weak-learner-specific arguments.
   * These arguments will be added to the list of arguments under 
   * the group specific of the weak learner. It is called
   * automatically in main, when the list of arguments is built up.
   * Use this method to declare the arguments that belongs to
   * the weak learner only.
   * \param args The Args class reference which can be used to declare
   * additional arguments.
   * \date 28/11/2005
   */
   virtual void declareArguments(nor_utils::Args& args);

   /**
   * Set the arguments of the algorithm using the standard interface
   * of the arguments. Call this to set the arguments asked by the user.
   * \param args The arguments defined by the user in the command line.
   * \date 14/11/2005
   */
   virtual void initLearningOptions(const nor_utils::Args& args);

   /**
   * Return {+1, -1} for the given class and value using the learned classifier.
   * \param pData The pointer to the data
   * \param idx The index of the example to classify
   * \param classIdx The index of the class
   * \remark This override classify of StumpLearner, as the data must be
   * transformed to haar-like feature space.
   * \remark Passing the data and the index to the example is not nice at all.
   * This will soon be replace with the passing of the example itself in some
   * form (probably a structure to the example).
   * \return +1 if the classifier thinks that \a val belongs to class 
   * \a classIdx, -1 if it does not and 0 if it abstain
   * \date 13/11/2005
   */
   virtual float classify(InputData* pData, int idx, int classIdx);

   /**
   * Returns itself as object.
   * \remark It uses the trick described in 
   * http://www.parashift.com/c++-faq-lite/serialization.html#faq-36.8
   * for the auto-registering classes.
   * \date 14/11/2005
   */
   virtual BaseLearner* subCreate() { return new HaarSingleStumpLearner(); }

   /**
   * Creates an InputData object that it is good for the
   * weak learner. Overridden to return HaarData.
   * \see InputData
   * \see BaseLearner::createInputData()
   * \see SortedData
   * \warning The object \b must be destroyed by the caller.
   * \date 21/11/2005
   */
   virtual InputData* createInputData();

   /**
   * Run the learner to build the classifier on the given data.
   * \param pData The pointer to the data
   * \see SingleStumpLearner::run
   * \date 11/11/2005
   */
   virtual float run();

   /**
   * Save the current object information needed for classification.
   * \param outputStream The stream where the data will be saved
   * \param numTabs The number of tabs before the tag. Useful for indentation
   * \remark To fully save the object it is \b very \b important to call
   * also the super-class method.
   * \see StumpLearner::save()
   * \date 13/11/2005
   */
   virtual void save(ofstream& outputStream, int numTabs = 0);

   /**
   * Load the xml file that contains the serialized information
   * needed for the classification and that belongs to this class.
   * \param st The stream tokenizer that returns tags and values as tokens
   * \see save()
   * \date 13/11/2005
   */
   virtual void load(nor_utils::StreamTokenizer& st);

   /**
   * Copy all the info we need in classify().
   * pBaseLearner was created by subCreate so it has the correct (sub) type.
   * Usually one must copy the same fields that are loaded and saved. Don't 
   * forget to call the parent's subCopyState().
   * \param pBaseLearner The sub type pointer into which we copy.
   * \see save
   * \see load
   * \see classify
   * \see ProductLearner::run()
   * \date 25/05/2007
   */
   virtual void subCopyState(BaseLearner *pBaseLearner);

   virtual void getStateData( vector<float>& data, const string& /*reason = ""*/, InputData* pData = 0 );

private:

   /**
   * A discriminative function. 
   * \remarks Positive or negative do NOT refer to positive or negative classification.
   * This function is equivalent to the phi function in my thesis.
   * \param val The value to discriminate
   * \return +1 if \a val is on one side of the border for \a classIdx and -1 otherwise
   * \date 11/11/2005
   * \see classify
   */
   virtual float phi(float val, int /*classIdx*/) const;

};

} // end of namespace MultiBoost

#endif // __HAAR_SINGLE_STUMP_LEARNER_H
