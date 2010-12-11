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
* \file SortedData.h Input data which has the column sorted.
*/

#ifndef __SORTED_DATA_H
#define __SORTED_DATA_H

#include "classifier/boosting/IO/InputData.h"

#include <vector>
#include <utility> // for pair

using namespace std;

namespace MultiBoost {

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

/**
* Overloading of the InputData class to support sorting of the column.
* This is particularly useful for stump-based learner, because
* they work column-by-column (dimension-by-dimension), looking for a threshold
* that minimizes the error, and sorting the data it's mandatory.
* The connection between this class and the weak learner that implements
* decision stump, is done with the overriding of method BaseLearner::createInputData()
* which will return the desired InputData type (and which might depend on
* the arguments of the command line too).
* \see BaseLearner::createInputData()
* \see StumpLearner::createInputData()
* \date 21/11/2005
*/
class SortedData : public InputData
{
public:

   /**
   * The destructor. Must be declared (virtual) for the proper destruction of 
   * the object.
   */
   virtual ~SortedData() {}

   /**
   * Overloading of the load function to support sorting.
   * \param fileName The name of the file to be loaded.
   * \param inputType The type of input.
   * \param verboseLevel The level of verbosity.
   * \see InputData::load()
   * \date 21/11/2005
   * \todo Erase the original memory once sorted?
   */
   virtual void load(const string& fileName, eInputType inputType = IT_TRAIN, int verboseLevel = 1);

   /**
   * Get the first and last elements of the (sorted) column of the data.
   * \param colIdx The column index
   * \return A pair containing the iterator to the first and last elements of the column
   * \remark The second is the end() iterator, so it does not point to anything!
   * \remark Replaces getSortedEnd() and getSortedBegin() in earlier versions
   * \date 03/17/2006
   */

   virtual bool isAttributeEmpty( int idx ) {
	   return _sortedData[idx].empty();
   }

   virtual bool isFilteredAttributeEmpty() {
	   return _filteredColumn.empty();
   }

   virtual bool isFilteredAttributeHasOneValue() {
	   return ( _filteredColumn[0].second == _filteredColumn[_filteredColumn.size()-1].second );
   }


protected:
   virtual pair<vpIterator,vpIterator> getSortedBeginEnd(int colIdx) { 
	   if ( this->isFiltered() ) {
		   return this->getFileteredBeginEnd( colIdx );
	   } else {
		return make_pair(_sortedData[colIdx].begin(),_sortedData[colIdx].end()); }
   }
public: 
   virtual pair<vpIterator,vpIterator> getFileteredBeginEnd(int colIdx);
   virtual pair<vpReverseIterator,vpReverseIterator> getFileteredReverseBeginEnd(int colIdx);
protected:

   /**
   * A column of the data.
   * The pair represents the index of the example and the value of the column.
   * The index of the column is the index of the vector itself.
   * \remark I am storing both the index and the value because it is a trade off between
   * speed in a key part of the code (finding the threshold) and the memory consumption.
   * In case of very large databases, this could be turned into a index only vector.
   * \date 11/11/2005
   */
   typedef vector< pair<int, float> > column;

   vector<column>    _sortedData; //!< the sorted data.

   column _filteredColumn;

};


} // end of namespace MultiBoost

#endif // __SORTED_DATA_H
