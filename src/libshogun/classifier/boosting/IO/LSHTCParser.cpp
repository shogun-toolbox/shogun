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

#include "classifier/boosting/IO/LSHTCParser.h"
#include "classifier/boosting/Utils/Utils.h"

namespace shogun {

	// ------------------------------------------------------------------------

	LSHTCParser::LSHTCParser(const string& fileName)
		: GenericParser(fileName), _hasName(false)
	{
		_denseLocale  = locale(locale(), new nor_utils::white_spaces(": "));
		_sparseLocale = locale(locale(), new nor_utils::white_spaces(": "));
	}

	// ------------------------------------------------------------------------

	void LSHTCParser::readData( vector<Example>& examples, NameMap& classMap, 
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
		_labelRep = LR_SPARSE;
		
		readClassHierarchy();
		
		//readHeader(inFile, classMap, enumMaps, attributeNameMap, attributeTypes);
		readData(inFile, examples, classMap, enumMaps, attributeNameMap, attributeTypes);
		
		_hierarchy.getClassNameMap( classMap );
	}

	// ------------------------------------------------------------------------
	void LSHTCParser::readClassHierarchy() {
		// ez csak a hasznalt kategoriakt fogja tartalmazni
		_hierarchy.load( _hierarchyFile );
		// ez a teljes hierarchiat
		_originalHierarchy.load( _hierarchyFile );
		// ez az osszes kategoriat, meg azokat is, amelyeket atvaltunk
		_usedCategories.clear();

		if ( _labelSetting == LS_FULL ) {
			_originalHierarchy.getCategorySet( _usedCategories );
		} else if ( _labelSetting == LS_SUBCAT ) {
			_hierarchy.subcat( _labelSettingParameter );
			_hierarchy.getCategorySet( _usedCategories );
		} else if ( _labelSetting == LS_DEPTH ) {
			_hierarchy.eraseTreeUptoDepth( _labelSettingParameter ); //pl ha 1, akkor csak az elso szintu kategoriak maradnak meg
			_originalHierarchy.getCategorySet( _usedCategories );
		} else if ( _labelSetting == LS_NDEPTH ) {
			_originalHierarchy.getCategorySet( _usedCategories );
		}else if ( _labelSetting == LS_LEAF ) {
			_originalHierarchy.getCategorySet( _usedCategories );
		}else if ( _labelSetting == LS_CHILDREN ) {
			//megszoritjuk a reszfat erre a kategoriara, de meg kell hagyni a gyerek kategoriakat is, hogy at tudjuk valtani a szulokategoriara
			//pl. adott egy 1 2 34 kategoriaju elem, akkor a ennek az osztalya 2 lesz
			// a vegen meg kell hivni a clearChildren fuggvenyt
			_hierarchy.keepTheChildrenOfACategory( _labelSettingParameter );

			vector<int> tmpVec;
			_originalHierarchy.getDescendants( tmpVec, _labelSettingParameter );

			for( vector<int>::iterator it = tmpVec.begin(); it != tmpVec.end(); it++ ) {
				_usedCategories.insert( *it );
			}

		}
			
	}

	// ------------------------------------------------------------------------


	void LSHTCParser::readData( ifstream& in, vector<Example>& examples,
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
			if ( firstChar == '%' ) // comment!
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

			int category;
			//get the category num (the number which can be found in the cat_hier.txt)			
			ss >> category;
			

			//ha nincs ez a kategoria, a hierachiaban, akkor be se olvassuk
			set<int>::iterator it = _usedCategories.find( category );
			
			if ( it == _usedCategories.end() ) continue;
			
			Example& currExample = examples[currentSize++];

			if ( _labelRep == LR_SPARSE ) setHierchicalSparseLabelLabels( category, currExample.getLabels(), classMap );
			//else if ( _labelRep == LR_DENSE ) setHierchicalDenseLabelLabels( category, currExample.getLabels(), classMap );
			else {
				cout << "Unknonw label represenation!!!" << endl;
				exit( -1 );
			}

			//now read values
			readSparseValues(ss, currExample.getValues(), currExample.getValuesIndexes(), currExample.getValuesIndexesMap(),
				enumMaps, attributeTypes );

			//determine the number of column			
			map<int,int> featIdxs = currExample.getValuesIndexesMap();
			for( map<int,int>::iterator it = featIdxs.begin(); it != featIdxs.end(); it++ ) {
				if ( it->first > maxColumnIdx ) maxColumnIdx = it->first;
			}

		}


		if ( _labelSetting == LS_SUBCAT ) {
		} else if ( _labelSetting == LS_DEPTH ) {
			_hierarchy.eraseTreeUptoDepth( _labelSettingParameter );
		} else if ( _labelSetting == LS_NDEPTH ) {
		}else if ( _labelSetting == LS_LEAF ) {
		}else if ( _labelSetting == LS_CHILDREN ) {
			//megszoritjuk a reszfat erre a kategoriara, de meg kell hagyni a gyerek kategoriakat is, hogy at tudjuk valtani a szulokategoriara
			//pl. adott egy 1 2 34 kategoriaju elem, akkor a ennek az osztalya 2 lesz
			// a vegen meg kell hivni a eraseUptoDepth fuggvenyt
			_hierarchy.eraseTreeUptoDepth(1);
		}


		for( int j = 0; j <= maxColumnIdx; j++ ) {
			string s;
			stringstream sstream;
			sstream << j;
			s = sstream.str();


			attributeNameMap.addName( s );
			attributeTypes.push_back( RawData::ATTRIBUTE_NUMERIC );
		}

		cout << "Done!" << endl;

		if ( currentSize != numRows )
		{
			// last row was empty!
			examples.resize( currentSize );
		}

		// set the number of attributes, it can be problem, that the test dataset contains such attributes which isn't presented in the training data
		_numAttributes = attributeNameMap.getNumNames();

		// sparse representation always set the weight!
		if ( _labelRep == LR_SPARSE )
			_hasWeigthInit = true;
	}

	// ------------------------------------------------------------------------

	void LSHTCParser::setHierchicalSparseLabelLabels( int category, vector<Label>& labels,
		NameMap& classMap )
	{
	
		//LS_LEAF meg nem mukodik
		if ( ( _labelSetting == LS_FULL	) || ( _labelSetting == LS_SUBCAT )) {
			
			labels.clear();
			
			vector<int> siblingCategories;
			vector<int> ancestorCategories;
			vector<int>::iterator itParent;
			vector<int>::iterator itSibling;

			//get its ancestor categories
			_hierarchy.getAncestors( ancestorCategories, category );
			


			for( itParent = ancestorCategories.begin(); itParent != ancestorCategories.end(); itParent++ ) {
				_hierarchy.getSiblings( siblingCategories, *itParent );
				
				for( itSibling = siblingCategories.begin(); itSibling != siblingCategories.end(); itSibling++ ) {
					Label tmpLabel;
					tmpLabel.idx = _hierarchy.convertCategoryToIdx( *itSibling );
					if ( *itParent == *itSibling ) {
						tmpLabel.y = +1;
						tmpLabel.weight = 1.0;
					} else { 
						tmpLabel.y = -1;
						tmpLabel.weight = 1.0;
					}

					// now get the declared labels
					labels.push_back( tmpLabel );

				}
			}
		} else if ( _labelSetting == LS_DEPTH ) {
			labels.clear();
			
			vector<int> siblingCategories;
			vector<int> ancestorCategories;
			vector<int>::iterator itParent;
			vector<int>::iterator itSibling;

			//get its ancestor categories
			_originalHierarchy.getAncestors( ancestorCategories, category );
			

			int depth;
			for( depth = 1, itParent = ancestorCategories.begin(); itParent != ancestorCategories.end(); itParent++, depth++ ) {
				_hierarchy.getSiblings( siblingCategories, *itParent );
				
				for( itSibling = siblingCategories.begin(); itSibling != siblingCategories.end(); itSibling++ ) {
					Label tmpLabel;
					tmpLabel.idx = _hierarchy.convertCategoryToIdx( *itSibling );
					if ( *itParent == *itSibling ) {
						tmpLabel.y = +1;
						tmpLabel.weight = 1.0;
					} else { 
						tmpLabel.y = -1;
						tmpLabel.weight = 1.0;
					}

					// now get the declared labels
					labels.push_back( tmpLabel );

				}
			}

		} else if ( _labelSetting == LS_CHILDREN ) {
			_labelRep = LR_DENSE;
			

			const int numClasses = _hierarchy.getNumOfCategories();
			labels.resize(numClasses);

			for ( int i = 0; i < numClasses; ++i )
			{
				labels[i].idx = i;
				labels[i].y = -1;
			}

			

			vector<int> ancestorCategories;
			vector<int>::iterator itParent;
			
			//get its ancestor categories
			_originalHierarchy.getAncestors( ancestorCategories, category );

			for( itParent = ancestorCategories.begin(); itParent != ancestorCategories.end(); itParent++ ) {
				if ( _hierarchy.existCategory( *itParent ) )
					labels[ _hierarchy.convertCategoryToIdx( *itParent ) ].y = +1;
			}

		} else {
			cout << "Unknown label setting or it hasn't implemented yet!!!!" << endl;
			exit( -1 );
		}


	}

	// ------------------------------------------------------------------------

	void LSHTCParser::setHierchicalDenseLabelLabels( int category, vector<Label>& labels,
		NameMap& classMap )
	{
		
		const int numClasses = classMap.getNumNames();
		labels.resize(numClasses);

		for ( int i = 0; i < numClasses; ++i )
		{
			labels[i].idx = i;
			labels[i].y = -1;
		}

		

		vector<int> siblingCategories;
		vector<int> ancestorCategories;
		vector<int>::iterator itParent;
		vector<int>::iterator itSibling;

		//get its ancestor categories
		_hierarchy.getAncestors( ancestorCategories, category );
		


		for( itParent = ancestorCategories.begin(); itParent != ancestorCategories.end(); itParent++ ) {
			labels[ _hierarchy.convertCategoryToIdx( *itParent ) ].y = +1;
		}



	}


	// ------------------------------------------------------------------------

	void LSHTCParser::readDenseValues(ifstream& in, vector<float>& values,
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

	void LSHTCParser::readSparseValues(istringstream& ss, vector<float>& values, 
		vector<int>& idxs, map<int, int>& idxmap, vector<NameMap>& enumMaps, 
		const vector<RawData::eAttributeType>& attributeTypes)
	{
		int tmpVal;
		int tmpIdx;
		int i = 0;
		while (!ss.eof())
		{
			ss >> tmpIdx;
			idxs.push_back(tmpIdx);
			idxmap[ tmpIdx ] = i++;
			ss >> tmpVal;

			values.push_back( tmpVal );
		}
	}

	// -----------------------------------------------------------------------------

	LSHTCParser::eTokenType LSHTCParser::getNextTokenType( ifstream& in )
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
