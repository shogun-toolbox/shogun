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



#include "ExtendableData.h"

#include "classifier/boosting/Defaults.h" // for STABLE_SORT declaration
#include "classifier/boosting/Utils/Utils.h" // for comparePairOnSecond
#include <algorithm> // for sort

// ------------------------------------------------------------------------
namespace MultiBoost {

void ExtendableData::load(const string& fileName, eInputType inputType, int verboseLevel)
{
   SortedData::load(fileName, inputType, verboseLevel);
   _numColumnsOriginal = _numAttributes;

   // initialize parent indices of existing columns to -1
   _parentIndices1.reserve(_numAttributes);
   _parentIndices2.reserve(_numAttributes);

   for (int j = 0; j < _numAttributes; ++j) 
   {
      _parentIndices1.push_back(-1);
      _parentIndices2.push_back(-1);
   }
}

// ------------------------------------------------------------------------
pair<vpIterator,vpIterator> ExtendableData::getSortedBeginEnd(int colIdx)
{
   if (colIdx < _numColumnsOriginal)
      return SortedData::getSortedBeginEnd(colIdx);
   else
   {
//       const HierarchicalStumpLearner* pWeakLearner1 = getParent1WeakLearner(colIdx);
//       const HierarchicalStumpLearner* pWeakLearner2 = getParent2WeakLearner(colIdx);

//       vector<double> tempColumnVector;

//       tempColumnVector.reserve(_numExamples);

//       for (int i = 0; i < _numExamples; ++i) 
//       {
// 	 char value1 = pWeakLearner1->phi(this, i);
// 	 char value2 = pWeakLearner2->phi(this, i);
// 	 double newValue;
// 	 if (value1 == value2)
// 	    newValue = 0.0;
// 	 else
// 	    newValue = static_cast<double>(value1);
// 	 tempColumnVector.push_back(newValue);
//       }

      _tempSortedColumn.resize(_numExamples);

      for (int i = 0; i < _numExamples; ++i)
         _tempSortedColumn[i] = make_pair( i, getValue(i,colIdx) );

#if STABLE_SORT
      stable_sort( _tempSortedColumn.begin(), _tempSortedColumn.end(), 
		   nor_utils::comparePairOnSecond< int, double, less<double> > );
#else
      sort( _tempSortedColumn.begin(), _tempSortedColumn.end(), 
            nor_utils::comparePair<2, int, double, less<double> >() );
#endif
       
      return make_pair(_tempSortedColumn.begin(), _tempSortedColumn.end());
   }

}
   // BUG: We refer to _sortedData somewhere 
// ------------------------------------------------------------------------
void ExtendableData::addColumn(const vector<double>& newColumnVector, int parentIdx1, int parentIdx2)
{
   // increment the number of columns for the stored data 
   ++_numAttributes;
//    _sortedData.resize(_numColumns);

   //////////////////////////////////////////////////////////////////////////
   // Fill the new column of the sorted data vector.
   // The data is stored column-wise. The pair represents the index 
   // of the example with the value.
//    column& newSortedColumn = _sortedData[_numColumns-1]; 
//    newSortedColumn.reserve(_numExamples);

//   addDataColumn(newColumnVector);
   
   //////////////////////////////////////////////////////////////////////////
   // Now sort the new column.  todo: for certain kind of features (2-valued or
   // 3-valued) sorting could be much more efficient with bucket sort, I might
   // want to signal it in a parameter and change the sorting action
   // accordingly.

// #if STABLE_SORT
//    stable_sort( newSortedColumn.begin(), newSortedColumn.end(), 
// 		nor_utils::comparePairOnSecond< int, double, less<double> > );
// #else
//    sort( newSortedColumn.begin(), newSortedColumn.end(), 
//             nor_utils::comparePairOnSecond< int, double, less<double> > );
// #endif

   // save parent indices
   _parentIndices1.push_back(parentIdx1);
   _parentIndices2.push_back(parentIdx2);
}

// ------------------------------------------------------------------------

void ExtendableData::addWeakLearner(HierarchicalStumpLearner* pWeakLearner)
{
   _weakLearners.push_back(pWeakLearner);
   ++_numWeakLearners;
}

// ------------------------------------------------------------------------

} // end of namespace MultiBoost
