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
* \file ExtendableData.h Input data with the possibility of adding
* new features.
*/

#ifndef __EXTENDABLE_DATA_H
#define __EXTENDABLE_DATA_H

#include "classifier/boosting/IO/SortedData.h"
#include "classifier/boosting/WeakLearners/HierarchicalStumpLearner.h"

#include <vector>

using namespace std;

namespace MultiBoost {

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

/**
* Overloading of the SortedData class to support adding (and perhaps deleting)
* features (columns in the data matrix). Experimental stage.
* \date 15/02/2006
*/
class ExtendableData : public SortedData
{
public:

   /**
   * The constructor. It does noting but initializing some variables.
   * \date 17/02/2006
   */
   ExtendableData() : _numColumnsOriginal(0), _numWeakLearners(0) {}

   /**
   * The destructor. Must be declared (virtual) for the proper destruction of 
   * the object.
   */
   virtual ~ExtendableData() {}

   /**
   * Overloading of the load function to remeber the number of columns in the original data.
   * \param fileName The name of the file to be loaded.
   * \param inputType The type of input.
   * \param verboseLevel The level of verbosity.
   * \see InputData::load()
   * \see SortedData::load()
   * \date 17/02/2006
   */
   virtual void load(const string& fileName, eInputType inputType = IT_TRAIN, int verboseLevel = 1);

   /**
   * Get the first and last elements of the (sorted) column of the data.
   * \param colIdx The column index
   * \return A pair containing the iterator to the first and last elements of the column
   * \remark The second is the end() iterator, so it does not point to anything!
   * \see SortedData::getSortedBeginEnd()
   * \date 03/17/2006
   */
   virtual pair<vpIterator,vpIterator> getSortedBeginEnd(int colIdx);

   /**
   * Add new column (feature) to the data matrix.
   * \param newColumn The vector of the new feature values.
   * \param parentIdx1 The index of its first parent
   * \param parentIdx2 The index of its second parent
   * \date 15/02/2006
   */
   void addColumn(const vector<double>& newColumnVector, int parentIdx1, int parentIdx2);

   /**
   * Add new weak learner
   * \param weakLearner The weak learner to be added.
   * \date 16/02/2006
   */
   void addWeakLearner(HierarchicalStumpLearner* pWeakLearner);

   /**
   * Get idx'th weak learner
   * \param idx The index of the weak learner to be added.
   * \date 17/02/2006
   */
   const HierarchicalStumpLearner* getWeakLearner(int idx) const { return _weakLearners[idx]; }
   const HierarchicalStumpLearner* getParent1WeakLearner(int idx) const { 
      // for debugging:
      cout << "returning parent1 of [" << idx << "] :" <<  _parentIndices1[idx] << endl;
      return getWeakLearner(_parentIndices1[idx]); }
   const HierarchicalStumpLearner* getParent2WeakLearner(int idx) const {  
      // for debugging:
      cout << "returning parent2 of [" << idx << "] :" <<  _parentIndices2[idx] << endl;
      return getWeakLearner(_parentIndices2[idx]); }

   int getNumWeakLearners() { return _numWeakLearners; }  //!< Returns the number of weak learners.
   int getNumColumnsOriginal()  const { return _numColumnsOriginal; }   //!< Returns the number of columns in the original data set.

protected:

   vector< vector<double> > _unsortedExtendedData;

   int   _numColumnsOriginal;   //!< The number of columns in the original data set.
   vector<const HierarchicalStumpLearner*>   _weakLearners;   //!< Previously learned weak learners
   vector<int>   _parentIndices1;   //!< Indices of first parent of combined features
   vector<int>   _parentIndices2;   //!< Indices of second parent of combined features
   int   _numWeakLearners;   //!< Number of previously learned weak learners
   int _numAttributes; //MBFIXME
   column _tempSortedColumn;

};

} // end of namespace MultiBoost

#endif // __EXTENDABLE_DATA_H
