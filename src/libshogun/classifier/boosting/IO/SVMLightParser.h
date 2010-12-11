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
* \file SVMLightParser.h A parser for ARFF file format
*/
#pragma warning( disable : 4786 )


#ifndef __SVMLIGHT_PARSER_H
#define __SVMLIGHT_PARSER_H

#include <fstream>
#include <sstream>
#include "GenericParser.h"
#include "NameMap.h"
#include "InputData.h"
#include "classifier/boosting/Utils/ClassHierarchy.h"

using namespace std;

namespace MultiBoost {

	class SVMLightParser : public GenericParser
	{
	public:
		SVMLightParser(const string& fileName);

		virtual void readData(vector<Example>& examples, NameMap& classMap, 
			vector<NameMap>& enumMaps, NameMap& attributeNameMap,
			vector<RawData::eAttributeType>& attributeTypes);

		virtual int  getNumAttributes() const
		{ return _numAttributes; }
		
		void setHeaderFileName(const string& headerFileName) 
		{ _headerFileName = headerFileName; }

	protected:
		/*
		Read the header in the following form:
		First line: 1 2 4 6 # class labels
		Second line: id1 id2 id3 # feature names
		Third line: 0.2 0.2 0.1 0.5 # label weights (optional )
		*/
		void readHeader( NameMap& classMap, 
			vector<NameMap>& enumMaps, NameMap& attributeNameMap,
			vector<RawData::eAttributeType>& attributeTypes );

		void readData(ifstream& in, vector<Example>& examples, NameMap& classMap, 
			vector<NameMap>& enumMaps, NameMap& attributeNameMap,
			vector<RawData::eAttributeType>& attributeTypes);

		string readName(ifstream& in);


		void readDenseValues(ifstream& in, vector<float>& values,
			vector<NameMap>& enumMaps, 
			const vector<RawData::eAttributeType>& attributeTypes);

		void readSparseValues(istringstream& ss, vector<float>& values, vector<int>& idxs, map<int, int>& idxmap, 
			vector<NameMap>& enumMaps, 
			const vector<RawData::eAttributeType>& attributeTypes, NameMap& attributeNameMap);

		void allocateSimpleLabels( int labelIdx, vector<Label>& labels,
			NameMap& classMap );


		enum eTokenType
		{
			TT_EOF,
			TT_COMMENT,
			TT_RELATION,
			TT_ATTRIBUTE,
			TT_DATA,
			TT_UNKNOWN
		};

		eTokenType getNextTokenType(ifstream& in);

		int            _numAttributes;

		locale         _denseLocale;
		locale         _sparseLocale;
		bool           _hasName;
		string		   _headerFileName;
	    map< int, float> _weightOfClasses;

	public:

	};

	// -----------------------------------------------------------------------------

	inline string SVMLightParser::readName(ifstream& in)
	{
		const locale& originalLocale = in.imbue(_denseLocale); 
		string name;
		in >> name;
		in.imbue(originalLocale);
		return name;
	}

	// -----------------------------------------------------------------------------


} // end of namespace MultiBoost

#endif // __ARFF_PARSER_H
