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

#include "classifier/boosting/IO/ArffParserBzip2.h"
#include "classifier/boosting/Utils/Utils.h"

namespace shogun {

	// ------------------------------------------------------------------------

	ArffParserBzip2::ArffParserBzip2(const string& fileName)
		: GenericParser(fileName), _hasName(false)
	{
		_denseLocale  = locale(locale(), new nor_utils::white_spaces(", "));
		_sparseLocale = locale(locale(), new nor_utils::white_spaces(", "));
	}

	// ------------------------------------------------------------------------

	void ArffParserBzip2::readData( vector<Example>& examples, NameMap& classMap, 
		vector<NameMap>& enumMaps, NameMap& attributeNameMap,
		vector<RawData::eAttributeType>& attributeTypes )
	{

		// open file
		Bzip2WrapperReader inFile( _fileName.c_str() );
		if ( ! inFile.is_open() )
		{
			cerr << "\nERROR: Cannot open file <" << _fileName << ">!!" << endl;
			exit(1);
		}
		inFile.setDelim( " " );

		_dataRep = DR_UNKNOWN;
		_labelRep = LR_UNKNOWN;

		readHeader(inFile, classMap, enumMaps, attributeNameMap, attributeTypes);
		readData(inFile, examples, classMap, enumMaps, attributeTypes);

	}

	// ------------------------------------------------------------------------
	// ------------------------------------------------------------------------

	void ArffParserBzip2::readHeader( Bzip2WrapperReader& in, NameMap& classMap, 
		vector<NameMap>& enumMaps, NameMap& attributeNameMap,
		vector<RawData::eAttributeType>& attributeTypes )
	{
		bool isData = false;
		string tmpStr;
		string tmpAttrType;

		char firstChar = 0;

		locale labelsLocale = locale(locale(), new nor_utils::white_spaces("{,}"));

		while ( !isData )
		{
			switch ( getNextTokenType(in) )
			{
			case TT_DATA:
				isData = true;
				break;

			case TT_COMMENT:
				getline(in, tmpStr); // ignore line
				break;

			case TT_RELATION:
				in >> _headerFileName;
				break;

			case TT_ATTRIBUTE:
				in >> tmpStr;

				if ( nor_utils::cmp_nocase(tmpStr, "class") )
				{
					// It's a class!!
					char firstChar = 0;
					while ( isspace(firstChar = in.get()) && !in.eof() );
					in.putback(firstChar);

					getline(in, tmpStr);
					stringstream ss(tmpStr);
					ss.imbue(labelsLocale); 

					// read the classes
					for (;;)
					{
						ss >> tmpStr;
						if ( ss.eof() )
							break;
						tmpStr = nor_utils::trim(tmpStr);
						if (!tmpStr.empty())
							classMap.addName(tmpStr);
					}
				}
				else
				{
					NameMap enumMap;
					in >> tmpAttrType;
					if ( nor_utils::cmp_nocase(tmpAttrType, "numeric") || 
						nor_utils::cmp_nocase(tmpAttrType, "real") || 
						nor_utils::cmp_nocase(tmpAttrType, "integer") )
					{
						attributeNameMap.addName(tmpStr);
						attributeTypes.push_back(RawData::ATTRIBUTE_NUMERIC);
					}
					else if ( nor_utils::cmp_nocase(tmpAttrType, "string") )
					{
						if (attributeNameMap.getNumNames() == 0)
							_hasName = true;
						else
						{
							cerr << "ERROR: One can specify the name of an example only as the first attribute, otherwise string types are not supported!!" << endl;
							exit(1);
						}               
					}
					else 
					{
						// enum attributeTypes
						// For the time being the enumeration cannot contain spaces, we should
						// correct it.
						if (tmpAttrType[0] == '{') 
						{
							attributeNameMap.addName(tmpStr);
							attributeTypes.push_back(RawData::ATTRIBUTE_ENUM);
							stringstream ss(tmpAttrType);
							ss.imbue(labelsLocale);

							for (;;)
							{
								ss >> tmpAttrType;
								if ( ss.eof() )
									break;
								tmpAttrType = nor_utils::trim(tmpAttrType);
								if (!tmpAttrType.empty())
									enumMap.addName(tmpAttrType);
							}
						}
						else
						{
							cerr << "ERROR: Unknown attribute type " << tmpAttrType[0] << endl;
							exit(1);
						}               
					}
					// We create an enumMap for every attribute, non enum type ones, too, so
					// the jth attribute's enumMap is always in enumMaps[j]
					enumMaps.push_back(enumMap);
				}
				// skip to the end of the line
				//while ( in.get() != '\n' && !in.eof() );

				break;

			case TT_UNKNOWN:
				cerr << "ERROR: Unknown token in the input file!" << endl;
				exit(1);
				break;

			case TT_EOF:
				cerr << "ERROR: End of File before reading any data! Separate header/data is not supported yet!" << endl;
				exit(1);
				break;

			}
		}

		_numAttributes = attributeNameMap.getNumNames();
	}

	// ------------------------------------------------------------------------

	void ArffParserBzip2::readData( Bzip2WrapperReader& in, vector<Example>& examples,
		NameMap& classMap, vector<NameMap>& enumMaps,
		const vector<RawData::eAttributeType>& attributeTypes )
	{
		char firstChar = 0;
		string tmpLine;

		istringstream ssDense;
		ssDense.imbue(_denseLocale);
		istringstream ssSparse;
		ssSparse.imbue(_sparseLocale);

		cout << "Counting rows.." << flush;
		//size_t numRows = nor_utils::count_rows(in);
		/////////////////////////////////////////////////////
		size_t numRows = in.remainingRowNum();
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
		size_t i;
		for (i = 0; i < numRows; ++i)
		{

			while ( isspace(firstChar = in.get()) && !in.eof() );

			if (in.eof())
				break;

			//if ( i == 7493 ) {
				//cout << i << endl;
			//}


			Example& currExample = examples[i];

			//////////////////////////////////////////////////////////////////////////
			// first read the data
			if ( firstChar == '%' ) // comment!
			{
				getline(in, tmpLine);
				continue;
			}

			// read the name if specified
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

			if ( firstChar == '{' ) // sparse data!
			{
				if ( _dataRep == DR_DENSE )
				{
					cerr << "ERROR: Cannot have dense and sparse data at the same time!" << endl;
					exit(1);
				}
				else if ( _dataRep == DR_UNKNOWN )
					_dataRep = DR_SPARSE;

				getline(in, tmpLine, '}');
				ssSparse.clear();
				ssSparse.str(tmpLine);

				readSparseValues(ssSparse, currExample.getValues(), currExample.getValuesIndexes(), currExample.getValuesIndexesMap(),
					enumMaps, attributeTypes );
			}
			else // dense!
			{
				if ( _dataRep == DR_SPARSE )
				{
					cerr << "ERROR: Cannot have dense and sparse data at the same time!" << endl;
					exit(1);
				}
				else if ( _dataRep == DR_UNKNOWN )
					_dataRep = DR_DENSE;

				

				in.putback(firstChar);
				readDenseValues(in, currExample.getValues(), enumMaps, attributeTypes);
				/*
				vector <float > vals = currExample.getValues();
				for( int i= 0; i < vals.size(); i++ ) 
				{
					cout << vals[ i ] << ",";
				}
				cout << endl;
				*/
			}

			//////////////////////////////////////////////////////////////////////////
			while ( !in.eof() )
			{
				// skip spaces and the last comma
				firstChar = in.get();
				if ( !isspace(firstChar) && firstChar != ',' )
					break;
			}

			// now read the labels
			if ( firstChar == '{' ) // weight is specified!
			{
				if ( _labelRep == LR_DENSE )
				{
					cerr << "ERROR: Labels cannot be formatted both in dense and sparse format!" << endl;
					exit(1);
				}
				_labelRep = LR_SPARSE;

				getline(in, tmpLine);
				ssSparse.clear();
				ssSparse.str(tmpLine);

				readExtendedLabels(ssSparse, currExample.getLabels(), classMap);
			}
			else // dense!
			{
				if ( _labelRep == LR_SPARSE )
				{
					cerr << "ERROR: Labels were declared sparse, but they are not formatted correctly (with {}!)!" << endl;
					exit(1);
				}
				_labelRep = LR_DENSE;

				in.putback(firstChar);

				getline(in, tmpLine);
				ssDense.clear();
				ssDense.str(tmpLine);

				readSimpleLabels(ssDense, currExample.getLabels(), classMap);
			}
		}

		cout << "Done!" << endl;

		if ( i != numRows )
		{
			// last row was empty!
			examples.resize(i);
		}

		// sparse representation always set the weight!
		if ( _labelRep == LR_SPARSE )
			_hasWeigthInit = true;
	}

	// ------------------------------------------------------------------------

	void ArffParserBzip2::readSimpleLabels( istringstream& ss, vector<Label>& labels,
		NameMap& classMap )
	{

		string strLabel;
		const int numClasses = classMap.getNumNames();
		labels.resize(numClasses);

		for ( int i = 0; i < numClasses; ++i )
		{
			labels[i].idx = i;
			labels[i].y = -1;
		}

		// now get the declared labels
		while (!ss.eof())
		{
			ss >> strLabel;
			strLabel = nor_utils::trim(strLabel);
			labels[ classMap.getIdxFromName(strLabel) ].y = +1;
		}

	}

	// ------------------------------------------------------------------------

	void ArffParserBzip2::readExtendedLabels( istringstream& ss, vector<Label>& labels,
		NameMap& classMap )
	{
		string strLabel;
		float weight;

		// now get the declared labels
		while (!ss.eof())
		{
			Label tmpLabel;
			ss >> strLabel;

			if ( strLabel == "}" || ss.eof() )
				break;

			ss >> weight;

			tmpLabel.y = nor_utils::sign(weight);
			tmpLabel.weight = abs(weight); // this will be used later in RawData to set the weights

			strLabel = nor_utils::trim(strLabel);
			tmpLabel.idx = classMap.getIdxFromName(strLabel);
			labels.push_back(tmpLabel);
		}
	}

	// ------------------------------------------------------------------------

	void ArffParserBzip2::readDenseValues(Bzip2WrapperReader& in, vector<float>& values,
		vector<NameMap>& enumMaps, 
		const vector<RawData::eAttributeType>& attributeTypes )
	{
		//const locale& originalLocale = in.imbue(_denseLocale); 

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

		//in.imbue(originalLocale);
	}


	// -----------------------------------------------------------------------------

	void ArffParserBzip2::readSparseValues(istringstream& ss, vector<float>& values, 
		vector<int>& idxs, map<int,int>& idxmap, vector<NameMap>& enumMaps, 
		const vector<RawData::eAttributeType>& attributeTypes)
	{
		string tmpVal;
		int tmpIdx;
		int i = 0;
		while (!ss.eof())
		{
			ss >> tmpIdx;
			idxs.push_back(tmpIdx);
			idxmap[ tmpIdx ] = i++;
			ss >> tmpVal;
			if ( attributeTypes[tmpIdx] == RawData::ATTRIBUTE_NUMERIC ) 
				values.push_back( atof(tmpVal.c_str()) );
			else // if ( attributeTypes[tmpIdx] == InputData::ATTRIBUTE_ENUM) 
				values.push_back( enumMaps[tmpIdx].getIdxFromName(tmpVal) );
		}
	}

	// -----------------------------------------------------------------------------

	ArffParserBzip2::eTokenType ArffParserBzip2::getNextTokenType( Bzip2WrapperReader& in )
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

} // end of namespace shogun
