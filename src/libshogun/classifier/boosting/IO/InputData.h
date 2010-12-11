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
* \file InputData.h The input of the training and testing data.
*/

#ifndef __INPUT_pData_H
#define __INPUT_pData_H

#include <vector>
#include <map> // for class mappings
#include <utility> // for pair
#include <iosfwd> // for I/O
#include <limits> 

#include "classifier/boosting/Utils/Args.h"
#include "classifier/boosting/Defaults.h" // for MB_DEBUG

//#include "GenericParser.h"
#include "NameMap.h"
#include "classifier/boosting/Others/Example.h"
#include "RawData.h"

//#include "Parser.h"
#include <cassert>

using namespace std;

namespace MultiBoost {

	//////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////

	// A couple of useful typedefs
	typedef vector< pair<int, float> >::iterator			vpIterator; //!< Iterator on pair 
	typedef vector< pair<int, float> >::reverse_iterator	vpReverseIterator; //!< Iterator on pair 
	typedef vector< pair<int, float> >::const_iterator		cvpIterator; //!< Const iterator on pair 


	/**
	* Handles the data.
	* This class not just holds the data information but also the weights on examples
	* and labels.
	* It also stores the sorted data (for decision stump algorithms) if necessary.
	*
	* \date 05/11/2005
	*/
	class InputData
	{
		// these will be moved soon
	protected:
		bool   _hasExampleName, _classInLastColumn;
		vector< int >	_indirectIndices;
		set< int >		_usedIndices; // this  will contain the indeces we use
		map< int, int > _rawIndices;
	public:

		/**
		* The constructor. It does noting but initializing some variables.
		* \date 12/11/2005
		*/
		InputData() : _hasExampleName(false), _classInLastColumn(false), _numExamples(0) { _pData = new RawData(); }


		virtual int getOrderBasedOnRawIndex( int rawIndex ) {
			map<int, int>::iterator it = this->_rawIndices.find( rawIndex );
			if ( it == this->_rawIndices.end() ) return -1;
			else return it->second;
		}

		virtual bool isSamplesFromOneClass() {
			for( int i = 0; i < _pData->getNumClasses(); i++ ) {
				if ( ( _nExamplesPerClass[i] > 0 ) && ( _nExamplesPerClass[i] < this->_numExamples ) ) return false;
				if ( _nExamplesPerClass[i] == this->_numExamples ) return true;
			}
			return true;
		}

		/**
		* Set the arguments of the algorithm using the standard interface
		* of the arguments. Call this to set the arguments asked by the user.
		* \param args The arguments defined by the user in the command line.
		* on the derived classes.
		* \warning It does not have a declareArguments because it is 
		* dealt by the weak learner responsible for the input data
		* (so that the option goes under its own group).
		* \date 14/11/2005
		*/
		virtual void initOptions(const nor_utils::Args& args) {
			_pData->initOptions( args );
		}

		/**
		* Load the given file.
		* \param fileName The name of the file to be loaded.
		* \param inputType The type of input.
		* \param verboseLevel The level of verbosity.
		* \see eInputType
		* \date 08/11/2005
		*/
		virtual void load( const string& fileName, 
			eInputType inputType = IT_TRAIN, 
			int verboseLevel = 1) {
				// load the raw data
				int i;
				_pData = _pData->load( fileName, inputType, verboseLevel );
				//set the num of examples of the derived class (at the beginning the whole dataset is used by the InputData object )
				_numExamples = _pData->getNumExample();
				// need to set the indirect indices
				for( i=0; i < _numExamples; i++ ) {
					_indirectIndices.push_back( i );
					_usedIndices.insert( i );
					_rawIndices[i] = i;
				}
				
				_nExamplesPerClass = _pData->getExamplesPerClass();				
		}

		/**
		* Gets the labels of the given example.
		* \param idx The index of the example
		* \return The labels of the example [idx].
		* \date 10/11/2005
		*/
		inline const vector<Label>& getLabels(const int idx) const { return _pData->getLabels( _indirectIndices[idx] ); }
		inline       vector<Label>& getLabels(const int idx)       { return _pData->getLabels( _indirectIndices[idx] ); }

		inline const bool  hasLabel(const int idx, const int labelIdx) const 
		{ return _pData->hasLabel( _indirectIndices[idx], labelIdx); }

		inline const bool  hasPositiveLabel(const int idx, const int labelIdx) const 
		{ return _pData->hasPositiveLabel( _indirectIndices[idx], labelIdx); }

		/**
		* Get the values of the example \a idx
		* \param idx The index of the example.
		* \date 11/11/2005
		*/   
		inline const vector<float>& getValues(int idx) const 
		{ return _pData->getValues( _indirectIndices[idx] ); }
		inline       vector<float>& getValues(int idx) 
		{ return _pData->getValues( _indirectIndices[idx] ); }

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// IMPORTANT: This is a temporary version that assumes all the data
		// is dense!!!
		float   getValue(int idx, int columnIdx) const 
		{ return _pData->getValue( _indirectIndices[idx], columnIdx); }
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////

		inline const Example& getExample(int idx)
		{ return _pData->getExample( _indirectIndices[idx] ); }

		virtual inline const vector<Example>& getExamples() { 
			if ( ! this->isFiltered() ) {
				return _pData->getExamples(); 
			} else {		
				_subset.clear();
				for ( set<int>::iterator it = _usedIndices.begin(); it != _usedIndices.end(); it++ ) {
					//Example e = this->getExample( *it );
					_subset.push_back( this->getExample( *it ) );
				}
				return _subset;
			}
		}

		inline const NameMap& getClassMap()
		{ return _pData->getClassMap(); }

		inline const NameMap& getAttributeNameMap()
		{ return _pData->getAttributeNameMap(); }

		inline const NameMap& getEnumMap(int j)
		{ return _pData->getEnumMap( j ); }

		/**
		* Get the label of the example.
		* \param idx The index of the example.
		* \return A string with the label of the example, if this has been specified with
		* --examplename argument.
		* \date 14/2/2006
		*/
		const string& getExampleName(const int idx) { return _pData->getExampleName( _indirectIndices[idx] ); }

		int      getNumAttributes()  const { return _pData->getNumAttributes(); }   //!< Returns the number of attributes.
		int      getNumExamples()    const { return _numExamples; } //!< Returns the number of examples.
		int      getNumClasses()    const { return _pData->getNumClasses(); } //!< Returns the number of classes.

		//void     addDataColumn(const vector<float>& col);

		/**
		* Get the number of examples per class.
		* \param classIdx The index of the class.
		* \date 11/11/2005
		*/
		int      getNumExamplesPerClass(int classIdx) const 
		{ return _nExamplesPerClass[classIdx]; }

		/**
		* Set the indices of subset we use
		* \param the set which contains the indices
		* \data 12/10/2009
		*/
		virtual int		loadIndexSet( set< int > ind );	
		
		virtual void getIndexSet( set< int >& ind )
		{
			ind = _usedIndices;
		}


		/**
		* Clear the indices of subset we use, the whole dataset containing _pData will be used 
		* 	* \data 12/10/2009
		*/
		void	clearIndexSet( void );

		inline bool isFiltered() { return _numExamples != _pData->getNumExample(); }

		inline int getRawIndex( int i ) { return _indirectIndices[i]; }

		float getFeaturewiseMax( int idx ) {
			float max = numeric_limits<float>::min();
			for( int i = 0; i < this->getNumExamples(); i++ ) {
				Example e = getExample( i );
				vector<float> v = e.getValues();
				if ( max > v[idx] ) max = v[idx];
			}
			return max;
		}

		float getFeaturewiseMin( int idx ) {
			float min = numeric_limits<float>::max();
			for( int i = 0; i < this->getNumExamples(); i++ ) {
				Example e = getExample( i );
				vector<float> v = e.getValues();
				if ( min < v[idx] ) min = v[idx];
			}
			return min;
		}

	protected:
		int           _numExamples;  //!<  The number of examples.
		vector<int>   _nExamplesPerClass;   //!< The number of examples per class.

		RawData*		_pData;

		vector<Example> _subset;
	};

} // end of namespace MultiBoost

#endif // __INPUT_pData_H
