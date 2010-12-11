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


#include <iostream>
#include <cmath> // for abs

#include "classifier/boosting/IO/SVMLightParser.h"
#include "classifier/boosting/Utils/Utils.h"

namespace shogun {

	// ------------------------------------------------------------------------

	SVMLightParser::SVMLightParser(const string& fileName)
		: GenericParser(fileName), _hasName(false)
	{
		_denseLocale  = locale(locale(), new nor_utils::white_spaces(": "));
		_sparseLocale = locale(locale(), new nor_utils::white_spaces(": "));
	}

	// ------------------------------------------------------------------------

	void SVMLightParser::readData( vector<Example>& examples, NameMap& classMap, 
		vector<NameMap>& enumMaps, NameMap& attributeNameMap,
		vector<RawData::eAttributeType>& attributeTypes )
	{
		// open file
		ifstream inFile(_fileName.c_str());
		if ( !inFile.is_open() )
		{
			cerr << "\nERROR: Cannot open file <" << _fileName << ">!!" << endl;
			exit(1);
		}

		_dataRep = DR_SPARSE;
		_labelRep = LR_DENSE;
		
		if ( ! _headerFileName.empty() ) // there is no file name
			readHeader( classMap, enumMaps, attributeNameMap, attributeTypes);
		readData(inFile, examples, classMap, enumMaps, attributeNameMap, attributeTypes);
	}


	// ------------------------------------------------------------------------
	void SVMLightParser::readHeader( NameMap& classMap, 
		vector<NameMap>& enumMaps, NameMap& attributeNameMap,
		vector<RawData::eAttributeType>& attributeTypes )
	{
		cout << "Reading header file (" << _headerFileName << ")...";
		ifstream inHeaderFile(_headerFileName.c_str());
		if ( !inHeaderFile.is_open() )
		{
			cerr << "\nERROR: Cannot open file <" << _headerFileName << ">!!" << endl;
			exit(1);
		}

		string tmpLine;

		getline(inHeaderFile, tmpLine);
		istringstream ss;
		ss.imbue(_sparseLocale);
		ss.clear();
		ss.str(tmpLine);
		//read the class labels
		while (!ss.eof())
		{
			//read the name of the label
			string tmpLab;
			ss >> tmpLab;

			if ( tmpLab.empty() ) continue;
			nor_utils::trim( tmpLab );
			if ( tmpLab.empty() ) continue;

			//add the class name to the namemap if it doesn't exist
			classMap.addName( tmpLab );
		}

		// read the feature names
		getline(inHeaderFile, tmpLine);
		ss.clear();
		ss.str(tmpLine);

		while (!ss.eof())
		{
			//read the name of the label
			string tmpFeatName;
			ss >> tmpFeatName;
			
			if ( tmpFeatName.empty() ) continue;
			nor_utils::trim( tmpFeatName );
			if ( tmpFeatName.empty() ) continue;

			//add the feat name to the namemap if it doesn't exist
			attributeNameMap.addName( tmpFeatName );
		}

		attributeTypes.resize( attributeNameMap.getNumNames() );
		fill( attributeTypes.begin(), attributeTypes.end(), RawData::ATTRIBUTE_NUMERIC );

		// read the label weighting if it is given
		getline(inHeaderFile, tmpLine);
		if ( ! tmpLine.empty() )
		{
			cout << "Read weighting...";
			_weightOfClasses.clear();

			ss.clear();
			ss.str(tmpLine);
			for ( int i=0; i < classMap.getNumNames(); i++ )
			{
				//read the name of the label
				float tmpWeight;
				ss >> tmpWeight;
				
				_weightOfClasses.insert( make_pair<int, float>( i, tmpWeight ) );					
			}

		}

		inHeaderFile.close();

		cout << "Done." << endl;
	}
	// ------------------------------------------------------------------------

	void SVMLightParser::readData( ifstream& in, vector<Example>& examples,
		NameMap& classMap, vector<NameMap>& enumMaps, NameMap& attributeNameMap,
		vector<RawData::eAttributeType>& attributeTypes )
	{
		char firstChar = 0;
		string tmpLine;
		int maxColumnIdx = 0;;

		istringstream ss;
		ss.imbue(_sparseLocale);

		cout << "Counting rows.." << flush;
		size_t numRows = nor_utils::count_rows(in);
		cout << "Allocating.." << flush;
		try {
			examples.resize(numRows);
		} 
		catch(...) {
			cerr << "ERROR: Cannot allocate memory for storage!" << endl;
			exit(1);
		}
		cout << "Done!" << endl;

		cout << "Now reading file.." << flush;
		int i;
		vector<int> tmpLabelIdxs( numRows );

		size_t currentSize = 0;
		for (i = 0; i < numRows; ++i)
		{
			while ( isspace(firstChar = in.get()) && !in.eof() );

			if (in.eof())
				break;

			

			//if ( i == 7492 ) {
				//cout << i << endl;
			//}


			//////////////////////////////////////////////////////////////////////////
			// first read the data
			if ( firstChar == '#' ) // comment!
			{
				getline(in, tmpLine);
				continue;
			}

			// read the name if specified
			/*
			if ( _hasName )
			{
				in.putback(firstChar);
				currExample.setName( readName(in) );
				while ( !in.eof() )
				{
					// skip spaces and the comma
					firstChar = in.get();
					if ( !isspace(firstChar) && firstChar != ',' )
						break;
				}
			}
			*/

			in.putback(firstChar);

			// now read the labels
			getline(in, tmpLine);
			ss.clear();
			ss.str(tmpLine);

			//set the labels
			string currLab;
			ss >> currLab;
			int tmpLabIDX = classMap.addName( currLab );
			tmpLabelIdxs[currentSize] = tmpLabIDX;

			Example& currExample = examples[currentSize++];
			//now read values
			readSparseValues(ss, currExample.getValues(), currExample.getValuesIndexes(), currExample.getValuesIndexesMap(),
				enumMaps, attributeTypes, attributeNameMap );
		}

		if ( attributeTypes.empty() )
		{
			attributeTypes.resize( attributeNameMap.getNumNames() );
			fill( attributeTypes.begin(), attributeTypes.end(), RawData::ATTRIBUTE_NUMERIC );
		}
		if ( currentSize != numRows )
		{
			// last row was empty!
			examples.resize( currentSize );
		}

		for( int i=0; i<examples.size(); i++ )
		{
			Example& currExample = examples[i];
			allocateSimpleLabels( tmpLabelIdxs[i], currExample.getLabels(), classMap );
		}

		cout << "Done!" << endl;

		// set the number of attributes, it can be problem, that the test dataset contains such attributes which isn't presented in the training data
		_numAttributes = attributeNameMap.getNumNames();

		// sparse representation always set the weight!
		if ( _labelRep == LR_SPARSE )
			_hasWeigthInit = true;
	}



	// ------------------------------------------------------------------------

	void SVMLightParser::readDenseValues(ifstream& in, vector<float>& values,
		vector<NameMap>& enumMaps, 
		const vector<RawData::eAttributeType>& attributeTypes )
	{
		const locale& originalLocale = in.imbue(_denseLocale); 

		values.reserve(_numAttributes);
		string tmpVal;

		for ( int j = 0; j < _numAttributes; ++j )
		{
			in >> tmpVal;
			if ( attributeTypes[j] == RawData::ATTRIBUTE_NUMERIC ) 
				if ( ( ! tmpVal.compare( "NaN" ) ) || ( ! tmpVal.compare( "?" ) ) )
					values.push_back( numeric_limits<float>::infinity() );
				else
					values.push_back(atof(tmpVal.c_str()));
			else //if ( attributeTypes[i] == RawData::ATTRIBUTE_ENUM ) 
				values.push_back( enumMaps[j].getIdxFromName(tmpVal) );
		}

		in.imbue(originalLocale);
	}

	// -----------------------------------------------------------------------------

	void SVMLightParser::readSparseValues(istringstream& ss, vector<float>& values, 
		vector<int>& idxs, map<int, int>& idxmap, vector<NameMap>& enumMaps, 
		const vector<RawData::eAttributeType>& attributeTypes, NameMap& attributeNameMap)
	{
		float tmpFeatVal;
		string tmpFeatName;
		int i = 0;
		while (!ss.eof())
		{
			//read the name of the next feature name and its value
			ss >> tmpFeatName;
			ss >> tmpFeatVal;

			//add the feature to the namemap if it doesn't exist
			int tmpIdx = attributeNameMap.addName( tmpFeatName );
			
			// add the feature value and its index
			idxs.push_back(tmpIdx);
			idxmap[ tmpIdx ] = i++;
			values.push_back( tmpFeatVal );
		}
	}

	// -----------------------------------------------------------------------------

	SVMLightParser::eTokenType SVMLightParser::getNextTokenType( ifstream& in )
	{
		char firstChar = 0;

		// skip white space at the beginning
		while ( isspace(firstChar = in.get()) && !in.eof() );

		if ( in.eof() )
			return TT_EOF;

		if ( firstChar == '%' )
			return TT_COMMENT;

		if ( firstChar != '@' )
			return TT_UNKNOWN;

		string str;
		in >> str;

		if ( nor_utils::cmp_nocase(str, "attribute") )
			return TT_ATTRIBUTE;
		else if ( nor_utils::cmp_nocase(str, "relation") )
			return TT_RELATION;
		else if ( nor_utils::cmp_nocase(str, "data") )
			return TT_DATA;

		return TT_UNKNOWN;
	}

	// ------------------------------------------------------------------------

	void SVMLightParser::allocateSimpleLabels( int labelIdx, vector<Label>& labels,
		NameMap& classMap )
	{

		const int numClasses = classMap.getNumNames();
		labels.resize(numClasses);

		for ( int i = 0; i < numClasses; ++i )
		{
			labels[i].idx = i;
			labels[i].y = -1;
		}

		// now set the declared labels
		labels[ labelIdx ].y = +1;

		if ( ! _weightOfClasses.empty() ) // weighting
			labels[ labelIdx ].weight = _weightOfClasses[ labelIdx ];		
	}

} // end of namespace shogun
