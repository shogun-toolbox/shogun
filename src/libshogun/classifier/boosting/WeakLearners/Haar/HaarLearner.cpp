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


#include "HaarLearner.h"
#include "classifier/boosting/WeakLearners/Haar/HaarFeatures.h"
#include "classifier/boosting/Utils/StreamTokenizer.h"

#include "classifier/boosting/IO/HaarData.h"
#include "classifier/boosting/IO/Serialization.h"
#include "classifier/boosting/Utils/Args.h"

#include <limits> // for numeric_limits
#include <ctime> // for time

namespace shogun {

// ------------------------------------------------------------------------------

void HaarLearner::declareArguments(nor_utils::Args& args)
{
   args.declareArgument("ftypes", 
                        "Specify the list with the type of Haar-like features. This list can contain "
                        "all the types requested at once. The types allowed are:\n"
                        "  2v: 2 blocs vertical feature\n"
                        "  2h: 2 blocs horizontal feature\n"
                        "  3v: 3 blocs vertical feature\n"
                        "  3h: 3 blocs horizontal feature\n"
                        "  4q: 4 blocs squared feature\n"
                        "For instance if you need all the 2 and 3 blocs type, just write:\n"
                        "-ftypes 2v2h3v3h\n"
                        "(DEFAULT: ALL of them will be selected)"
                        , 1, "<list>");

   args.declareArgument("csample",
                        "Instead of computing all the possible combinations, select a set of "
                        "random configurations. It is possible to specify the number of combinations "
                        "or the amount of time per feature type. The options are:\n"
                        "  num: The number of combination per type per iteration.\n"
                        "  time: The amount of time (in seconds) per type per iteration.\n"
                        "Example: -csample num 4 -> Set the number of random configurations to 4.\n"
                        "Example: -csample time 120 -> Set the time to 2 minutes per type per iteration.",
                        2, "<opt> <#|sec>");

   args.declareArgument("iisize", 
                        "Specify the size of the integral image representation for each example,\n"
                        "Example: -iisize 128x64", 
                        1, "<width>x<height>");

}

// ------------------------------------------------------------------------------

void HaarLearner::initOptions(const nor_utils::Args& args)
{
   // If the sampling is required
   if (args.hasArgument("csample"))
   {
      string randSearch = args.getValue<string>("csample", 0);
      // if the type is ST_NUM this is the number of samplings per feature per iteration
      // if the type is ST_TIME this is the amount of time per feature per iteration
      _samplingVal = args.getValue<int>("csample", 1);

      if (randSearch == "num")
         _samplingType = ST_NUM;
      else if (randSearch == "time")
         _samplingType = ST_TIME;
   }

}

// ------------------------------------------------------------------------------

void HaarLearner::save(ofstream& outputStream, int numTabs)
{
   // save the type of feature
   outputStream << Serialization::standardTag("type", _pSelectedFeature->getShortName(), numTabs) << endl;

   // Save the coordinates and size of the feature
   outputStream << Serialization::getTabs(numTabs) << "<rect>" << endl;
      outputStream << Serialization::standardTag("x", _selectedConfig.x, numTabs+1) << endl;
      outputStream << Serialization::standardTag("y", _selectedConfig.y, numTabs+1) << endl;
      outputStream << Serialization::standardTag("width", _selectedConfig.width, numTabs+1) << endl;
      outputStream << Serialization::standardTag("height", _selectedConfig.height, numTabs+1) << endl;
   outputStream << Serialization::getTabs(numTabs) << "</rect>" << endl;

}

// -----------------------------------------------------------------------

void HaarLearner::load(nor_utils::StreamTokenizer& st)
{
   // get the type of feature
   string shortName = UnSerialization::seekAndParseEnclosedValue<string>(st, "type");
   // get it from the registered features
   _pSelectedFeature = HaarFeature::RegisteredFeatures().getFeature(shortName);

   // Load the coordinates and size of the feature
   _selectedConfig.x = UnSerialization::seekAndParseEnclosedValue<short>(st, "x");
   _selectedConfig.y = UnSerialization::seekAndParseEnclosedValue<short>(st, "y");
   _selectedConfig.width = UnSerialization::seekAndParseEnclosedValue<short>(st, "width");
   _selectedConfig.height = UnSerialization::seekAndParseEnclosedValue<short>(st, "height");

}

// -----------------------------------------------------------------------

void HaarLearner::subCopyState(HaarLearner *pHaarLearner)
{
   pHaarLearner->_pSelectedFeature = _pSelectedFeature;
   pHaarLearner->_selectedConfig.x = _selectedConfig.x;
   pHaarLearner->_selectedConfig.y = _selectedConfig.y;
   pHaarLearner->_selectedConfig.width = _selectedConfig.width;
   pHaarLearner->_selectedConfig.height = _selectedConfig.height;
   pHaarLearner->_samplingType = _samplingType;
   pHaarLearner->_samplingVal = _samplingVal;
}

// ------------------------------------------------------------------------------

} // end of MultiBoost namespace
