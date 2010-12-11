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
* \file FeaturewiseLearner.h It represents all weak learners that search all or
* or a subset of features. and implement phi(x,l) as phi(x[selectedFeature],l)
*/

#ifndef __FEATUREWISE_LEARNER_H
#define __FEATUREWISE_LEARNER_H

#include "classifier/boosting/WeakLearners/AbstainableLearner.h"
#include "classifier/boosting/Utils/Args.h"
#include "classifier/boosting/IO/InputData.h"

#include <vector>
#include <fstream>
#include <limits>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {

/**
* A generic featurewise learner. It represents all weak learners that search all or
* or a subset of features, and implement phi(x,l) as phi(x[selectedFeature],l)
*
* \date 19/07/2006
*/
class FeaturewiseLearner : public AbstainableLearner
{
public:

   /**
   * The constructor. It initializes the selected column to -1.
   * \date 19/07/2006
   */
   FeaturewiseLearner() : _selectedColumn(-1), _maxNumOfDimensions( numeric_limits<int>::max() ) {}

   /**
   * The destructor. Must be declared (virtual) for the proper destruction of 
   * the object.
   */
   virtual ~FeaturewiseLearner() {}

   // this interfaces aren't implemented for all featurewise learner
   virtual float run( int colNum ) { return 0.0; } ;
   virtual float run( vector<int>& colIndexes ) {return 0.0; };
   /**
   * Declare weak-learner-specific arguments.
   * These arguments will be added to the list of arguments under 
   * the group specific of the weak learner. It is called
   * automatically in main, when the list of arguments is built up.
   * Use this method to declare the arguments that belongs to
   * the weak learner only.
   * 
   * This class declares the argument "rsample" only.
   * \param args The Args class reference which can be used to declare
   * additional arguments.
   * \date 19/07/2006
   */
   virtual void declareArguments(nor_utils::Args& args);

   /**
   * Set the arguments of the algorithm using the standard interface
   * of the arguments. Call this to set the arguments asked by the user.
   * \param args The arguments defined by the user in the command line.
   * \date 14/11/2005
   * \remark These options are used for training only!
   */
   virtual void initLearningOptions(const nor_utils::Args& args);

   /**
   * Save the current object information needed for classification,
   * that is: \a _selectedColumn, the column of the 
   * data with that yielded the lowest error
   * \param outputStream The stream where the data will be saved
   * \param numTabs The number of tabs before the tag. Useful for indentation
   * \remark To fully save the object it is \b very \b important to call
   * also the super-class method.
   * \see AbstainableLearner::save()
   * \date 13/11/2005
   */
   virtual void save(ofstream& outputStream, int numTabs = 0);

   /**
   * Load the xml file that contains the serialized information
   * needed for the classification and that belongs to this class
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

protected:

   /**
   * A discriminative function. 
   * \param pData The data
   * \param idx The index of the data point
   * \param classIdx The class index
   * \return The output of the discriminant function 
   * \date 19/07/2006
   */
   virtual float phi(InputData* pData, int idx, int classIdx) const;

   /**
   * A scalar discriminative function. 
   * \param val The value to discriminate, it is interpreted in the _selectedColumn'th feature
   * \param classIdx The class index
   * \return 
   * \date 11/11/2005
   */
   virtual float phi(float val, int classIdx) const = 0;

   int    _selectedColumn; //!< The column of the training data with the lowest error.
   int    _maxNumOfDimensions; //!< limit on the number of searched dimensions in run()
};

// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

} // end of namespace MultiBoost

#endif // __FEATUREWISE_LEARNER_H
