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
* \file ParasiteLearner.h A single threshold decision stump learner. 
* \date 24/04/2007
*/

#ifndef __PARASITE_LEARNER_H
#define __PARASITE_LEARNER_H

#include "classifier/boosting/WeakLearners/BaseLearner.h"
#include "classifier/boosting/Utils/Args.h"
#include "classifier/boosting/IO/InputData.h"

#include <vector>
#include <fstream>
#include <string>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace shogun {

/**
* A learner that loads a set of base learners, and boosts on the top of them. 
*/
class ParasiteLearner : public BaseLearner
{
public:

   /**
   * The constructor. It initializes sign of alpha to +1, and _closed to true
   * \date 28/04/2007
   */
   ParasiteLearner() 
       : _signOfAlpha(1),_closed(0) {}

   /**
   * The destructor. Must be declared (virtual) for the proper destruction of 
   * the object.
   */
   virtual ~ParasiteLearner() {}

   /**
   * Declare weak-learner-specific arguments.
   * adding --pool
   * The name of the shyp.xml file containing the base learners
   * The number of base learners
   * \param args The Args class reference which can be used to declare
   * additional arguments.
   * \date 24/04/2007
   */
   virtual void declareArguments(nor_utils::Args& args);

   /**
   * Set the arguments of the algorithm using the standard interface
   * of the arguments. Call this to set the arguments asked by the user.
   * \param args The arguments defined by the user in the command line.
   * \date 24/04/2007
   */
   virtual void initLearningOptions(const nor_utils::Args& args);

   /**
   * Returns itself as object.
   * \remark It uses the trick described in http://www.parashift.com/c++-faq-lite/serialization.html#faq-36.8
   * for the auto-registering classes.
   * \date 14/11/2005
   */
   virtual BaseLearner* subCreate() { return new ParasiteLearner(); }

   /**
   * Run the learner to build the classifier on the given data.
   * \see BaseLearner::run
   * \date 24/04/2007
   */
   virtual float run();

   /**
   * Return the classification using the learned classifier.
   * \param pData The pointer to the data
   * \param idx The index of the example to classify
   * \param classIdx The index of the class
   * \remark Passing the data and the index to the example is not nice at all.
   * This will soon be replace with the passing of the example itself in some
   * form (probably a structure to the example).
   * \return the classification using the learned classifier.
   * \date 24/04/2007
   */
   virtual float classify(InputData* pData, int idx, int classIdx);

   /**
   * Save the current object information needed for classification,
   * that is the single threshold.
   * \param outputStream The stream where the data will be saved
   * \param numTabs The number of tabs before the tag. Useful for indentation
   * \remark To fully save the object it is \b very \b important to call
   * also the super-class method.
   * \see BaseLearner::save()
   * \date 24/04/2007
   */
   virtual void save(ofstream& outputStream, int numTabs = 0);

   /**
   * Load the xml file that contains the serialized information
   * needed for the classification and that belongs to this class.
   * \param st The stream tokenizer that returns tags and values as tokens
   * \see save()
   * \date 24/04/2007
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

   /**
   * Return the index of the selected weak learner.
   * \date 24/04/2007
   */
   int getSelectedIndex() { return _selectedIdx; }

   /**
   * Return the sign of the coefficient.
   * We keep _alpha always positive, if --closed is specified, we do 
   * consider -1 times the weak learners, and if the coefficient of a weak
   * learner is negative, we set _signOfAlpha to -1.
   * \date 24/04/2007
   */
   int getSignOfAlpha() { return _signOfAlpha; }

   /**
   * Return the static vector containing the weak learners.
   * _baseLearners
   * \date 24/04/2007
   */
   const vector<BaseLearner*>& getBaseLearners() const { return _baseLearners; }

   virtual const char* get_name() const { return "ParasiteLearner"; }

protected:

   static int _numBaseLearners; //!< the user specified number of base learners
   static string _nameBaseLearnerFile; //!< the name of the shyp file with the pool
   static vector<BaseLearner*> _baseLearners; //!< the pool of base learners


   int _selectedIdx; //!< the index of the selected base learner
   int _signOfAlpha; //!< to close the set over multiplication by -1
   int _closed; //!< to indicate whether the user wants to close the set (default = true)

};

//////////////////////////////////////////////////////////////////////////

} // end of namespace shogun

#endif // __PARASITE_LEARNER_H
