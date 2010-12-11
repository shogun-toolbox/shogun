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
* \file HaarData.h Input data which loads integral image representations.
* \date 17/12/2005
*/

//
// IMPORTANT!! HAARDATA WILL BE SOON REPLACED WITH A SIMPLER CLASS THAT USES float!!
//

#ifndef __HAAR_DATA_H
#define __HAAR_DATA_H

#include "classifier/boosting/IO/InputData.h"
#include "classifier/boosting/WeakLearners/Haar/HaarFeatures.h"
#include "classifier/boosting/Utils/Utils.h" // for Rect

#include <string>
#include <map>
#include <cassert>

using namespace std;

namespace shogun {

// A couple of useful typedefs
typedef vector< pair<int, int> >::iterator       vpIntIterator; //!< Iterator on pair 
typedef vector< pair<int, int> >::const_iterator cvpIntIterator; //!< Const iterator on pair 

/**
* Overloading of the InputData class to load data which has been already
* transformed into integral image representation. To get more information about
* this format, please see the work of Viola and Jones "Robust Real-time Object Detection".
* To convert normal images (or any 1 or 2 dimensional data) you can use this simple
* matlab script:
\code
function iimage = computeIntegralImage(data, width, height)

   % Uncomment this if your data is not an array
   % data = data(:);

   % fill an array of size "width" with zeros
   cumulativeRowSum = zeros(width);
   arrayIdx = 1;
   iiImageIdx = 1;

   for y = 1:height
      prevIntImValue = 0;

      for x = 1:width
         cumulativeRowSum(x) = cumulativeRowSum(x) + data(arrayIdx);
         iimage(iiImageIdx) = prevIntImValue + cumulativeRowSum(x);
         prevIntImValue = iimage(iiImageIdx);

         iiImageIdx = iiImageIdx + 1;
         arrayIdx =  arrayIdx + 1;
      end
   end
\endcode
* \date 17/12/2005
*/
class HaarData : public InputData
{
public:

   /**
   * The destructor. Erases the integral image data.
   * \date 17/12/2005
   */
   virtual ~HaarData();

   /**
   * Set the arguments of the algorithm using the standard interface
   * of the arguments. Call this to set the arguments asked by the user.
   * \param args The arguments defined by the user in the command line.
   * on the derived classes.
   * \warning It does not have a declareArguments because it is 
   * dealt by the weak learner responsible for the input data
   * (so that the option goes under its own group).
   * \date 14/11/2005
   */
   virtual void initOptions(const nor_utils::Args& args);

   /**
   * Overloading of the load function to read integral images files.
   * \param fileName The name of the file to be loaded.
   * \param inputType The type of input.
   * \param verboseLevel The level of verbosity.
   * \see InputData::load()
   * \date 21/11/2005
   */
   virtual void load(const string& fileName, 
                     eInputType inputType = IT_TRAIN, int verboseLevel = 1);

   /**
   * Overload the standard method for getting data and returns an error if called.
   * This function cannot be overloaded for the integral image type, as it is
   * integer.
   * \date 11/11/2005
   */
   float getValue(int /*idx*/, int /*columnIdx*/) const 
      { assert(!"This function should not be called with HaarData!!"); return 0; }


   /**
   * Returns the whole integral image of example \a idx.
   * \param idx The index of the example.
   * \date 17/12/2005
   */
   //const int* getIntImage(int idx) const { return _intImages[idx]; }

   /**
   * Get the vector that contains the integral images for all the examples.
   * \date 17/12/2005
   */
   //const vector<int*>& getIntImageVector() const { return _intImages; }

   /**
   * Get the vector of the features types that have been loaded (or declared)
   * by the user.
   * \date 17/12/2005
   */
   vector<HaarFeature*>& getLoadedFeatures() { return _loadedFeatures; }

   //////////////////////////////////////////////////////////////////////////

   /**
   * Return the width of the integral image.
   * \date 17/12/2005
   */
   static short areaWidth() { return _width; }

   /**
   * Return the height of the integral image.
   * \date 17/12/2005
   */
   static short areaHeight() { return _height; }

protected:
    bool checkInput(const string& line, int numColumns);

   vector< int* >   _intImages;       //!< the data of the examples.

   static short   _width;  //!< The width of the integral image.
   static short   _height; //!< The height of the integral image. 

   /**
   * The list of features that have been requested by the user.
   */
   vector<HaarFeature*> _loadedFeatures; //!< 
};

} // end of namespace Multiboost

#endif // __HAAR_DATA_H
