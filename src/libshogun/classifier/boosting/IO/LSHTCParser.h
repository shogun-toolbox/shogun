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
* \file LSHTCParser.h A parser for ARFF file format
*/
#pragma warning( disable : 4786 )


#ifndef __LSHTC_PARSER_H
#define __LSHTC_PARSER_H

#include <fstream>
#include <sstream>
#include "GenericParser.h"
#include "NameMap.h"
#include "InputData.h"
#include "classifier/boosting/Utils/ClassHierarchy.h"

using namespace std;

namespace shogun {

	class LSHTCParser : public GenericParser
	{
	public:
		LSHTCParser(const string& fileName);

		void setHeaderFileName(const string& headerFileName) 
		{ _headerFileName = headerFileName; }
		void setHierarchyFileName( string fname ) 
		{ _hierarchyFile = fname; }

		virtual void readData(vector<Example>& examples, NameMap& classMap, 
			vector<NameMap>& enumMaps, NameMap& attributeNameMap,
			vector<RawData::eAttributeType>& attributeTypes);

		virtual int  getNumAttributes() const
		{ return _numAttributes; }

	protected:
		void readClassHierarchy();

		void readData(ifstream& in, vector<Example>& examples, NameMap& classMap, 
			vector<NameMap>& enumMaps, NameMap& attributeNameMap,
			vector<RawData::eAttributeType>& attributeTypes);

		string readName(ifstream& in);

		void setHierchicalSparseLabelLabels( int category, vector<Label>& labels, NameMap& classMap );
		void setHierchicalDenseLabelLabels( int category, vector<Label>& labels, NameMap& classMap );

		void readDenseValues(ifstream& in, vector<float>& values,
			vector<NameMap>& enumMaps, 
			const vector<RawData::eAttributeType>& attributeTypes);

		void readSparseValues(istringstream& ss, vector<float>& values, vector<int>& idxs, map<int, int>& idxmap, 
			vector<NameMap>& enumMaps, 
			const vector<RawData::eAttributeType>& attributeTypes);


		enum eTokenType
		{
			TT_EOF,
			TT_COMMENT,
			TT_RELATION,
			TT_ATTRIBUTE,
			TT_DATA,
			TT_UNKNOWN
		};

		enum eLabelSetting 
		{
			LS_FULL,	// az osszes categoriat hasznaljuk
			LS_SUBCAT,	// csak egy belso kategoriat reszfajat tartjuk 
			LS_DEPTH,	// egy bizonyos melysegik hasznaljuk a kategoriakat
			LS_NDEPTH,  //csak egy adott az N-edik melysegben hasznaljuk a kategoriat
			LS_LEAF,	//csak a levelkategoriakat hasznaljuk
			LS_CHILDREN // egy adott kategoria gyerekeit osztalyozzunk
		};

		eTokenType getNextTokenType(ifstream& in);

		eLabelSetting	_labelSetting;
		int				_labelSettingParameter;

		int            _numAttributes;
		string         _headerFileName;

		locale         _denseLocale;
		locale         _sparseLocale;
		bool           _hasName;

		string			_hierarchyFile;
		ClassHierarchy	_hierarchy;
		ClassHierarchy	_originalHierarchy;
		set<int>		_usedCategories;
	public:
		void setLabeling( string type, int par ) 
		{
			//"full, subcat, depth, ndepth, leaf",
			_labelSettingParameter = par;

			if ( type == "full" ) {
				_labelSetting = LS_FULL;
			} else if ( type == "subcat" ) {
				_labelSetting = LS_SUBCAT;
			} else if ( type == "depth" ) {
				_labelSetting = LS_DEPTH;
			} else if ( type == "ndepth" ) {
				_labelSetting = LS_NDEPTH;
			} else if ( type == "leaf" ) {
				_labelSetting = LS_LEAF;
			} else if ( type == "children" ) {
				_labelSetting = LS_CHILDREN;
			} else {
				cout << "Unknown labelling (LSHTCParser)!!!" << endl;
				exit( -1 );
			}
		}

	};

	// -----------------------------------------------------------------------------

	inline string LSHTCParser::readName(ifstream& in)
	{
		const locale& originalLocale = in.imbue(_denseLocale); 
		string name;
		in >> name;
		in.imbue(originalLocale);
		return name;
	}

	// -----------------------------------------------------------------------------


} // end of namespace shogun

#endif // __ARFF_PARSER_H
