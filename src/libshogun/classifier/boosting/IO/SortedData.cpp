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


#include "SortedData.h"

#include "classifier/boosting/Defaults.h" // for STABLE_SORT declaration
#include "classifier/boosting/Utils/Utils.h" // for comparePairOnSecond
#include <algorithm> // for sort

#include <limits>

// ------------------------------------------------------------------------
namespace MultiBoost {

void SortedData::load(const string& fileName, eInputType inputType, int verboseLevel)
{
   InputData::load(fileName, inputType, verboseLevel);

   // Test does not need sorting
   if (inputType == IT_TEST)
      return;

   if (verboseLevel > 0)
      cout << "Sorting data..." << flush;

   // set the number of columns for the stored data
   _sortedData.resize(_pData->getNumAttributes()); 
   int i = 0;

   if ( _pData->getDataRep() == DR_DENSE )
   {
	   for ( i = 0; i < _pData->getNumAttributes(); ++i )
         _sortedData[i].reserve(_numExamples);
   }
   
   //////////////////////////////////////////////////////////////////////////
   // Fill the sorted data vector.
   // The data is stored column-wise. The index [j] is the column
   // and the pair represent the index of the example with the value
   vector<Example>::iterator eIt;

   i = 0;
   // for each example
   for ( eIt = _pData->rawBegin(); eIt != _pData->rawEnd(); ++eIt, ++i )
   {
      vector<float>& values = eIt->getValues();
      const vector<int>& valIdx = eIt->getValuesIndexes();
      vector<float>::iterator vIt;
      int j = 0;

      if ( valIdx.empty() ) // dense data!
      {
         //_sortedData[j].reserve(values.size()); // <-- this should be done before, and only of dense!
         // for each attribute of the example
         for (vIt = values.begin(); vIt != values.end(); ++vIt, ++j )
            _sortedData[j].push_back( make_pair(i, *vIt) ); // store the index of the example and the value
      }
      else // sparse data
      {
         // for each attribute of the example
         for (vIt = values.begin(); vIt != values.end(); ++vIt, ++j )
            _sortedData[ valIdx[j] ].push_back( make_pair(i, *vIt) ); // store the index of the example and the value
      }
   }

   //////////////////////////////////////////////////////////////////////////
   // Now sort the data.
   
   // For each column
   for (int j = 0; j < _pData->getNumAttributes(); ++j)
   {

#if STABLE_SORT
      stable_sort( _sortedData[j].begin(), _sortedData[j].end(), 
                   nor_utils::comparePair<2, int, float, less<float> >() );
#else
      sort( _sortedData[j].begin(), _sortedData[j].end(), 
            nor_utils::comparePair< 2, int, float, less<float> >() );
#endif
   }

   if (verboseLevel > 0)
      cout << "Done!" << endl;
}

// ------------------------------------------------------------------------

pair<vpIterator,vpIterator> SortedData::getFileteredBeginEnd(int colIdx) {
	_filteredColumn.clear();
	for( column::iterator it = _sortedData[colIdx].begin(); it != _sortedData[colIdx].end(); it ++ ) {
		set<int>::iterator setIt = this->_usedIndices.find( (*it).first ); 
		if ( ( setIt != this->_usedIndices.end() ) && ( it->second == it->second ) ) {
			int i = this->getOrderBasedOnRawIndex( it->first );
			_filteredColumn.push_back( pair<int, float>(i, it->second) );
		}
	}
	return make_pair(_filteredColumn.begin(),_filteredColumn.end());
}

// ------------------------------------------------------------------------

pair<vpReverseIterator,vpReverseIterator> SortedData::getFileteredReverseBeginEnd(int colIdx) {
	_filteredColumn.clear();
	for( column::iterator it = _sortedData[colIdx].begin(); it != _sortedData[colIdx].end(); it ++ ) {
		set<int>::iterator setIt = this->_usedIndices.find( (*it).first ); 
		if ( ( setIt != this->_usedIndices.end() ) && ( it->second == it->second )  ) {
			int i = this->getOrderBasedOnRawIndex( it->first );
			_filteredColumn.push_back( pair<int, float>(i, it->second) );
		}
	}
	return make_pair(_filteredColumn.rbegin(),_filteredColumn.rend());
}


/*
pair<vpIterator,vpIterator> SortedData::getFileteredBeginEnd(int colIdx) {
	if ( _pData->getDataRep() == DR_DENSE ) {
		_filteredColumn.clear();
		for( column::iterator it = _sortedData[colIdx].begin(); it != _sortedData[colIdx].end(); it ++ ) {
			set<int>::iterator setIt = this->_usedIndices.find( (*it).first ); 
			if ( setIt != this->_usedIndices.end() ) {
				int i = this->getOrderBasedOnRawIndex( it->first );
				_filteredColumn.push_back( pair<int, float>(i, it->second) );
			}
		}
	} else if ( _pData->getDataRep() == DR_SPARSE ) {
		//this solution is temporary, because this implementation convert dense data from the sparse one
		_filteredColumn.clear();
		_filteredColumn.resize( _usedIndices.size() );
		set< int > tmpUsedIndices = this->_usedIndices;
		int i;
		column::reverse_iterator it;

		for( i = _filteredColumn.size()-1, it = _sortedData[colIdx].rbegin(); it != _sortedData[colIdx].rend(); it++, i-- ) {
			set<int>::iterator setIt = tmpUsedIndices.find( (*it).first ); 
			if ( setIt != tmpUsedIndices.end() ) {
				tmpUsedIndices.erase( *setIt );
				int order = this->getOrderBasedOnRawIndex( it->first );
				_filteredColumn[ i ] =  make_pair<int, float>(order, it->second);
			}
			
		}

		//put the zero elements into the column
		for( set<int>::iterator setIt = tmpUsedIndices.begin(); setIt != tmpUsedIndices.end(); setIt++, i-- ){
			int order = this->getOrderBasedOnRawIndex( *setIt );
			_filteredColumn[ i ] =  make_pair<int, float>(order, 0);
		}

	}
	return make_pair(_filteredColumn.begin(),_filteredColumn.end());
}
*/

// ------------------------------------------------------------------------

} // end of namespace MultiBoost
