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


#include "HaarData.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm> // for sort
#include <cmath> // for sqrt

// ------------------------------------------------------------------------
namespace MultiBoost {

short HaarData::_width = 0;
short HaarData::_height = 0;

// ------------------------------------------------------------------------

HaarData::~HaarData()
{
   // delete data
   vector<int*>::iterator it;
   for (it = _intImages.begin(); it != _intImages.end(); ++it)
      delete [] *it;
}

// ------------------------------------------------------------------------

void HaarData::initOptions(const nor_utils::Args& args)
{
   // call the super class
   InputData::initOptions(args);

   string regsString = HaarFeature::RegisteredFeatures().getRegString();

   if ( args.hasArgument("ftypes") )
      regsString = args.getValue<string>("ftypes", 0);

   string::const_iterator it;
   for (it = regsString.begin(); it != regsString.end(); ++it)
   {
      char argStr[3];
      argStr[0] = *it++;

      if (it != regsString.end())
         argStr[1] = *it;
      else
         argStr[1] = '\0';
      argStr[2] = '\0';

      if ( !HaarFeature::RegisteredFeatures().hasFeature(argStr) )
      {
         cerr << "ERROR: Haar feature declaration <" << argStr << "> not found!" << endl;
         exit(1);
      }

      HaarFeature* hf = HaarFeature::RegisteredFeatures().getFeature(argStr)->create();
      _loadedFeatures.push_back( hf );
   }

   string tmpVal = args.getValue<string>("iisize", 0);
	
   size_t divPos = tmpVal.find('x');
   if (divPos == string::npos)
   {
      cerr << "ERROR: syntax error for option -iisize!" << endl;
      exit(1);
   }
	
   istringstream ss(tmpVal);
   char val[10];
   ss.getline(val, 10, 'x');
   _width = static_cast<short>( atoi(val) );
	
   ss.getline(val, 10);
   _height = static_cast<short>( atoi(val) );
}

// ------------------------------------------------------------------------

void HaarData::load(const string& fileName, eInputType inputType, int verboseLevel)
{
   //if (verboseLevel > 0)
   //   cout << "Loading data..." << flush;

   //ifstream inFile(fileName.c_str());
   //if ( !inFile.is_open() )
   //{
   //   cerr << "\nERROR: Cannot open file <" << fileName << ">!!" << endl;
   //   exit(1);
   //}

   //// set white spaces to consider tab as NOT whitespace
   //// the white_tab will be erased automatically by fstream
   //inFile.imbue( locale(locale(), new nor_utils::white_spaces(_sepChars) ) );

   //int _numColumns = static_cast<int>(nor_utils::count_columns(inFile));

   //// if it has a filename for each example, don't count it
   //if (_hasExampleName) 
   //   --_numColumns;

   //// the class is not a data column
   //--_numColumns;

   //string line;
   //getline(inFile, line);
   //if ( !checkInput( line, _numColumns ) )
   //{
   //   cerr << "ERROR: Input file not correct, check file <" << fileName << "> for errors." << endl;
   //   exit(1);
   //}

   //inFile.clear(); // reset position
   //inFile.seekg(0);

   //if (_width == 0 || _height == 0)
   //{
   //   float square = sqrt(static_cast<float>(_numColumns));

   //   if ( !nor_utils::is_zero( square - static_cast<int>(square) ) )
   //   {
   //      cerr << "Error: the number of elements in file <" << fileName << "> for each example" << endl
   //           << "is not square." << endl
   //           << "Please make sure you are loading integral image data, or specify the size" << endl
   //           << "of it with -iisize." << endl;
   //      exit(1);
   //   }

   //   _width = static_cast<short>(square);
   //   _height = _width;
   //}

   //string tmpLine;

   //// this array will be filled with the values from the example.
   //// We need this to be sure we are not storing fake data because we reached
   //// the end of the file
   //int* pDataArray = NULL;

   //string tmpFileName;
   //string tmpClassName;

   //_numExamples = 0;

   //map<int, int> tmpPointsPerClass;

   ///////////////////////////
   //while( !inFile.eof() ) 
   //{
   //   if (_hasExampleName)
   //      inFile >> tmpFileName; // store file name

   //   if (!_classInLastColumn)
   //      inFile >> tmpClassName; // store class

   //   pDataArray = new int[_numColumns];
   //   if (!pDataArray)
   //   {
   //      cerr << "ERROR: Cannot allocate memory for storage!" << endl;
   //      exit(1);
   //   }

   //   for (int j = 0; j < _numColumns; ++j)
   //      inFile >> pDataArray[j]; // store values

   //   // to avoid problems in the case of an empty line at the end
   //   // of the file
   //   if ( inFile.eof() )
   //   {
   //      delete [] pDataArray;
   //      break;
   //   }

   //   if (_classInLastColumn)
   //      inFile >> tmpClassName; // store class

   //   int classIdx = ClassMappings::addClassName(tmpClassName);
   //   tmpPointsPerClass[ classIdx ]++;

   //   _intImages.push_back(pDataArray);
   //   _infoData.push_back( Example(classIdx, tmpFileName) );

   //   ++_numExamples;
   //} 

   ///////////////////////////

   //const int numClasses = ClassMappings::getNumClasses();

   //for (int l = 0; l < numClasses; ++l)
   //   _nExamplesPerClass.push_back( tmpPointsPerClass[l] );

   //// Initialize weights
   //initWeights();

   InputData::load(fileName, inputType, verboseLevel);

   // Test does not need sorting
   if (inputType == IT_TEST)
      return;

   if (verboseLevel > 0)
      cout << "Pre-compute configurations.." << flush;

   // precompute all the possible configurations for the features types
   // and the area of the integral image
   vector<HaarFeature*>::iterator it;
   int nPrecs = 0;
   for (it = _loadedFeatures.begin(); it != _loadedFeatures.end(); ++it)
   {
      nPrecs = (*it)->precomputeConfigs();
      if (verboseLevel > 1)
         cout << "(" << (*it)->getShortName() << ": " << nPrecs << ")" << flush;
   }

   //if (verboseLevel > 0)
   //   cout << endl;

   //if (verboseLevel > 1)
   //{
   //   cout << "Num Columns = " << _numColumns << endl;  

   //   for (int l = 0; l < numClasses; ++l)
   //      cout << "Of class '" << ClassMappings::getClassNameFromIdx(l) << "': " 
   //           << _nExamplesPerClass[l] << endl;

   //   cout << "Total: " << _numExamples << " examples read." << endl;
   //}

   if (verboseLevel > 0)
      cout << "Done!" << endl;

}

// ------------------------------------------------------------------------
bool HaarData::checkInput(const string& line, int numColumns)
{
   istringstream ss(line);
   ss.imbue( locale(locale(), new nor_utils::white_spaces(_pData->getSepChars()) ) );

   string tmp;
   bool inputValid = true;

   if (_hasExampleName)
      ss >> tmp; // filename

   if (!_classInLastColumn)
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

   if (_classInLastColumn)
      ss >> tmp; // class
   if ( tmp.empty() )
      inputValid = false;

   return inputValid;
}


} // end of namespace MultiBoost
