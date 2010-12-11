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


#include <fstream>
#include <iostream>
#include <sstream>

#include "classifier/boosting/IO/TxtParser.h"
#include "classifier/boosting/Utils/Utils.h"

namespace shogun {

// ------------------------------------------------------------------------

void TxtParser::readData( vector<Example>& examples, NameMap& classMap, 
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
   
   // set white spaces to consider tab as NOT whitespace
   // the white_tab will be erased automatically by fstream
   inFile.imbue( locale(locale(), new nor_utils::white_spaces(_sepChars) ) );

   _numAttributes = (int)nor_utils::count_columns(inFile);

   // if it has a filename for each example, don't count it
   if (_hasExampleName)
      --_numAttributes;

   // the class is not a data column
   --_numAttributes;

   string line;
   getline( inFile, line );
   if ( !checkInput( line, _numAttributes ) )
   {
      cerr << "\nERROR: Input file not correct, check file <" << _fileName << "> for errors," << endl
           << "or your separation option -d (if you are using it)" << endl;
      exit(1);
   }

   inFile.clear(); // reset position
   inFile.seekg(0);

   string tmpExampleName;
   string tmpClassName;

   cout << "Counting rows.." << flush;
   size_t numRows = nor_utils::count_rows(inFile);

   cout << "Allocating.." << flush;
   try {
      examples.resize(numRows);
   } 
   catch(...) {
      cerr << "ERROR: Cannot allocate memory for storage!" << endl;
      exit(1);
   }

   cout << "Done!" << endl;

   cout << "Reading file.." << endl;

   /////////////////////////
   size_t i;
   for (i = 0; i < numRows; ++i)
   {
      if (_hasExampleName)
         inFile >> tmpExampleName; // store file name

      if (!_hasClassEnd)
         inFile >> tmpClassName; // store class

      Example& currExample = examples[i];
      currExample.setName(tmpExampleName);
      
      vector<float>& values = currExample.getValues();
      values.resize(_numAttributes);

      for (int j = 0; j < _numAttributes; ++j)
         inFile >> values[j]; // store values

      // to avoid problems in the case of an empty line at the end
      // of the file
      if ( inFile.eof() )
         break;

      if (_hasClassEnd)
         inFile >> tmpClassName; // store class

      int classIdx = classMap.addName(tmpClassName);
      if ( classIdx > 1 )
      {
         cerr << "ERROR: Only binary labels are accepted (max 2!) for .txt parser!!" << endl;
         exit(1);
      }

      currExample.addBinaryLabel(classIdx);
   }

   if ( i != examples.size() )
   {
      cerr << "WARNING: Different number of read examples (" 
           << i << ") and lines (" 
           << examples.size() << ")!"  << endl;

      examples.resize(i);
   }

   cout << "Done!" << endl;
}

// ------------------------------------------------------------------------

bool TxtParser::checkInput(const string& line, int numColumns) const
{
   istringstream ss(line);
   ss.imbue( locale(locale(), new nor_utils::white_spaces(_sepChars) ) );

   string tmp;
   bool inputValid = true;

   if (_hasExampleName)
      ss >> tmp; // example name

   if (!_hasClassEnd)
      ss >> tmp; // class at the beginning

   for (int j = 0; j < numColumns; ++j)
   {
      if ( ss.eof() )
      {
         inputValid = false;
         break;
      }

      ss >> tmp;

      if ( !nor_utils::is_number(tmp) )
      {
         inputValid = false;
         break;
      }
   }

   if (_hasClassEnd)
      ss >> tmp; // class
   if ( tmp.empty() )
      inputValid = false;

   return inputValid;
}

// ------------------------------------------------------------------------

}

