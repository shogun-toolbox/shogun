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
* \file TxtParser.h A parser for TXT file format: dense data and single class only here!
*/

#ifndef __TXT_PARSER_H
#define __TXT_PARSER_H

#include "classifier/boosting/IO/GenericParser.h"

using namespace std;

namespace MultiBoost {

/**
* Parse simple text data.
* Here is an example of valid data (note: in this case the argument \b --examplelabel has been provided!):
* \verbatim
/home/music/classical/classical.00078.au	classical	5.72939e+01	2.95128e+02	6.43395e+00
/home/music/disco/disco.00078.au	disco	1.98315e+02	1.31341e+03	-6.15398e+00
/home/music/reggae/reggae.00022.au	reggae	2.51418e+02	7.68241e+02	-5.66704e+00
/home/music/hiphop/hiphop.00080.au	hiphop	2.62773e+02	4.83971e+02	8.80924e-01
/home/music/rock/rock.00015.au	rock	2.03546e+02	9.31192e+02	-7.56387e+00	1.15847e+02
/home/music/hiphop/hiphop.00027.au	hiphop	2.37860e+02	1.03110e+03	2.50052e-01
/home/music/rock/rock.00094.au	rock	2.48359e+02	1.69432e+02	-1.66508e+01
\endverbatim 
* \remark The column can be separated by any white-space character. If a particular 
* separator needs to be specified (for instance when the class name contains spaces), 
* use the --d option.
* \remark The TxtParser is STRICTLY single class (can have many classes but just one per example).
*/
class TxtParser : public GenericParser
{
public:
   TxtParser(const string& fileName)
      : GenericParser(fileName) {}

   void setHasExampleName(bool hasExampleName)
   { _hasExampleName = hasExampleName; }

   void setClassEnd(bool hasClassEnd)
   { _hasClassEnd = hasClassEnd; }

   void setSepChars(const string& sepChars)
   { _sepChars = sepChars; }

   virtual void readData( vector<Example>& examples, NameMap& classMap, 
			  vector<NameMap>& enumMaps, NameMap& attributeNameMap, 
			  vector<RawData::eAttributeType>& attributeTypes );

   virtual int   getNumAttributes() const
   { return _numAttributes; }

private:

   bool checkInput(const string& line, int numColumns) const;

   bool     _hasExampleName;
   bool     _hasClassEnd;
   string   _sepChars;

   int      _numAttributes;
};

} // end of namespace MultiBoost

#endif // __TXT_PARSER_H
