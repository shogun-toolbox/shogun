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
* \file ArffParser.h A parser for ARFF file format
*/
#pragma warning( disable : 4786 )


#ifndef __ARFF_PARSER_H
#define __ARFF_PARSER_H

#include <fstream>
#include <sstream>
#include "GenericParser.h"
#include "NameMap.h"
#include "InputData.h"

using namespace std;

namespace MultiBoost {

class ArffParser : public GenericParser
{
public:
   ArffParser(const string& fileName);

   void setHeaderFileName(const string& headerFileName) 
   { _headerFileName = headerFileName; }

   virtual void readData(vector<Example>& examples, NameMap& classMap, 
			 vector<NameMap>& enumMaps, NameMap& attributeNameMap,
			 vector<RawData::eAttributeType>& attributeTypes);

   virtual int  getNumAttributes() const
   { return _numAttributes; }

protected:
   
   void readHeader(ifstream& in, NameMap& classMap, 
		   vector<NameMap>& enumMaps, NameMap& attributeNameMap, 
		   vector<RawData::eAttributeType>& attributeTypes);
   void readData(ifstream& in, vector<Example>& examples, NameMap& classMap, 
		 vector<NameMap>& enumMaps, 
		 const vector<RawData::eAttributeType>& attributeTypes);

   string readName(ifstream& in);

   void readDenseValues(ifstream& in, vector<float>& values,
			vector<NameMap>& enumMaps, 
			const vector<RawData::eAttributeType>& attributeTypes);

   void readSparseValues(istringstream& ss, vector<float>& values, vector<int>& idxs, map<int, int>& idxmap, 
			 vector<NameMap>& enumMaps, 
			 const vector<RawData::eAttributeType>& attributeTypes);

   /**
   * Read labels declared in the standard arff format:
   * each class (label) is set to +1 if it's there, otherwise is -1
   * i.e.:
   * \verbatim
   @ATTRIBUTE sepallength  NUMERIC
   @ATTRIBUTE sepalwidth   NUMERIC
   @ATTRIBUTE petallength  NUMERIC
   @ATTRIBUTE petalwidth   NUMERIC
   @ATTRIBUTE class        {Iris-setosa, Iris-versicolor, Iris-virginica}
   @DATA
    4,  2,  4,  2, Iris-setosa
   25, 23,  1,  0, Iris-versicolor, Iris-virginica
    0,  1, 10, 12, Iris-virginica
   \endverbatim
   * In this case the labels the the example will respectively be:
   \verbatim
   +1, -1, -1
   -1, +1, +1
   -1, -1, +1
   \endverbatim
   */
   void readSimpleLabels(istringstream& ss, vector<Label>& labels, NameMap& classMap);

   /**
   * Read labels declared in a non-standard arff variant:
   * each class (label) is declared with a value that can be positive, or negative
   * left out labels are automatically considered zero (abstention!).
   * i.e.:
   * \verbatim
   @ATTRIBUTE sepallength  NUMERIC
   @ATTRIBUTE sepalwidth   NUMERIC
   @ATTRIBUTE petallength  NUMERIC
   @ATTRIBUTE petalwidth   NUMERIC
   @ATTRIBUTE class        {Iris-setosa, Iris-versicolor, Iris-virginica}
   @DATA
    4,  2,  4,  2, {Iris-setosa -2} 
   25, 23,  1,  0, {Iris-versicolor 1, Iris-virginica -1}
    0,  1, 10, 12, {Iris-setosa +2, Iris-versicolor -1, Iris-virginica -3}
   \endverbatim
   * The sign is used to set the value of y[l], and the magnitude to initialize the weights.
   * In this case the labels the the example will respectively be:
   \verbatim
   -1, 0, 0
   0, +1, -1
   +1, -1, -1
   \endverbatim
   * \remark Internally this type of label is stored as sparse. This will have
   * a small hit in terms of memory, but nothing in terms of performance.
   */
   void readExtendedLabels(istringstream& ss, vector<Label>& labels, NameMap& classMap);

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
   string         _headerFileName;
   
   locale         _denseLocale;
   locale         _sparseLocale;
   bool           _hasName;
};

// -----------------------------------------------------------------------------

inline string ArffParser::readName(ifstream& in)
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
