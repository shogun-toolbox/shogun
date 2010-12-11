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

#ifndef __RAW_DATA_H
#define __RAW_DATA_H

#include <vector>
#include <map> // for class mappings
#include <utility> // for pair
#include <iosfwd> // for I/O

#include "classifier/boosting/Utils/Args.h"
#include "classifier/boosting/Defaults.h" // for MB_DEBUG

//#include "GenericParser.h"
#include "NameMap.h"
#include "classifier/boosting/Others/Example.h"

//#include "Parser.h"
#include <cassert>

using namespace std;

namespace MultiBoost {

	/////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////
	enum eFileFormat
	{
		FF_SIMPLE,
		FF_ARFF,
		FF_ARFFBZIP,
		FF_LSHTC,
		FF_SVMLIGHT
		// FF_BINARY, // To come next!
	};

	/**
	* Defines the type of input. Used in case
	* train and test differs in any way.
	* \date 21/11/2005
	*/
	enum eInputType
	{
		IT_TRAIN, //!< If the input is train-type.
		IT_TEST //!< If the input is test-type. 
	};



	class RawData {
	 // these will be moved soon
	protected:
			bool   _hasExampleName, _classInLastColumn;
			string _sepChars;
			//vector< int >	indirectIndeces;

	public:

	   /**
	   * The constructor. It does noting but initializing some variables.
	   * \date 12/11/2005
	   */
	   RawData() : _hasExampleName(false), _classInLastColumn(false), _sepChars(" \t\n"),
					 _numAttributes(0), _numExamples(0), _fileFormat(FF_SIMPLE) {  }


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
		virtual void initOptions(const nor_utils::Args& args);


		/**
		* Load the given file.
		* \param fileName The name of the file to be loaded.
		* \param inputType The type of input.
		* \param verboseLevel The level of verbosity.
		* \see eInputType
		* \date 08/11/2005
		*/
		virtual RawData* load( const string& fileName, 
			eInputType inputType = IT_TRAIN, 
			int verboseLevel = 1);



		/**
		* Gets the labels of the given example.
		* \param idx The index of the example
		* \return The labels of the example [idx].
		* \date 10/11/2005
		*/
		inline const vector<Label>& getLabels(const int idx) const { return _data[idx].getLabels(); }
		inline       vector<Label>& getLabels(const int idx)       { return _data[idx].getLabels(); }

		const bool  hasLabel(const int idx, const int labelIdx) const 
		{ return _data[idx].hasLabel(labelIdx); }

		const bool  hasPositiveLabel(const int idx, const int labelIdx) const 
		{ return _data[idx].hasPositiveLabel(labelIdx); }

		/**
		* Get the values of the example \a idx
		* \param idx The index of the example.
		* \date 11/11/2005
		*/   
		inline const vector<float>& getValues(int idx) const 
		{ return _data[idx].getValues(); }
		inline       vector<float>& getValues(int idx) 
		{ return _data[idx].getValues(); }

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// IMPORTANT: This is a temporary version that assumes all the data
		// is dense!!!
		// the old version:
		//float   getValue(int idx, int columnIdx) const 
		//{ return _data[idx].getValues()[columnIdx]; }


		inline float   getValue(int idx, int columnIdx) const { 
			if ( _dataRep == DR_DENSE )	return _data[idx].getValues()[columnIdx]; 
			else {
				map<int,int>::iterator it = ((Example &)_data[idx]).getValuesIndexesMap().find( columnIdx );
				if ( it == ((Example &)_data[idx]).getValuesIndexesMap().end() ) return 0;
				else return _data[idx].getValues()[it->second];
			}
		}
			
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////

		/**
		*/
		void clearRawData() { _data.clear(); _numExamples = 0; } 

		void addExample( Example example ) { 
			_data.push_back( example ); 
			_numExamples++; 
		} 


		inline const Example& getExample(int idx)
		{ return _data[idx]; }

		inline const vector<Example>& getExamples() 
		{ return _data; }

		inline const NameMap& getClassMap()
		{ return _classMap; }

		inline const NameMap& getAttributeNameMap()
		{ return _attributeNameMap; }

		inline const NameMap& getEnumMap(int j)
		{ return _enumMaps[j]; }

		/**
		* Get the label of the example.
		* \param idx The index of the example.
		* \return A string with the label of the example, if this has been specified with
		* --examplename argument.
		* \date 14/2/2006
		*/
		const string& getExampleName(const int idx) { return _data[idx].getName(); }

		const eDataRep   getDataRep()  const { return _dataRep; }
		const eLabelRep  getLabelRep() const { return _labelRep; }

		int      getNumAttributes()  const { return _numAttributes; }   //!< Returns the number of attributes.
		int      getNumClasses()    const { return _classMap.getNumNames(); } //!< Returns the number of classes.
		int		 getNumExample() const { return _numExamples; }
		//bool    isSingleLabel() const { return true; }
		//bool    isDenseLabel() const { return false; }
		//bool    isSparseLabel() const { return !_isSingleLabel && !_isDenseLabel; }

		enum eAttributeType
		{
			ATTRIBUTE_NUMERIC, // eq1
			ATTRIBUTE_ENUM, // eq2 
		};

		enum eWeightInitType
		{
			WIT_SHARE_POINT, // eq1
			WIT_SHARE_LABEL, // eq2 
			WIT_PROP_ONLY,    // eq3
			WIT_BALANCED
		};

		const string getSepChars() { return _sepChars; }
		const eDataRep getDataRep() { return _dataRep; }
		vector<Example>::iterator rawBegin() {return _data.begin(); }
		vector<Example>::iterator rawEnd() {return _data.end();}

		vector< int >&	getExamplesPerClass() { return _nExamplesPerClass; }
	protected:
		int           _numAttributes;   //!< The number of columns (dimensions).
		int           _numExamples;  //!<  The number of examples.
		int           _numClasses;  //!<  The number of classes.
		vector<int>   _nExamplesPerClass;   //!< The number of examples per class.

		eFileFormat   _fileFormat;

		eWeightInitType _weightInitType;

		/**
		* Initialize the weights. 
		* The weights initialization formula is defined as:
		* \f[
		* w_{i,\ell}^{(1)} =  \begin{cases}
		*     \frac{1}{2n}  & \mbox{ if $\ell$ is the correct class (if $y_{i,\ell} = 1$),} \\
		*     \frac{1}{2n(k-1)} & \mbox{ otherwise (if $y_{i,\ell} = -1$).}
		*  \end{cases} 
		* \f]
		* where \f$n\f$ is the number of examples and \f$k\f$ the number of classes.
		* \see Example
		* \see _data
		* \date 11/11/2005
		*/
		virtual void  initWeights();

#if MB_DEBUG
		void checkVariances(); //!< Print a warning if there is no variance in a column. 
#endif

		// --------------------------------------------------------------------

		vector<Example> _data; //!< The vector of the data for the examples. 

		eDataRep       _dataRep;
		eLabelRep      _labelRep;

		NameMap         _classMap; //!< The map of class names 
		vector<NameMap> _enumMaps; //!< The vector of name maps of enum type attributes. 
		NameMap         _attributeNameMap; //!< The map of attribute names 

		vector<eAttributeType> _attributeTypes; //!< The vector of attribute types. 
		
		//for LSHTC challenge		
		string			_hierarchyFile;
		string			_labelingType;
		int				_labelingParameter;
	};

}

#endif
