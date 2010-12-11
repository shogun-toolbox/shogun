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
* \file HaarFeatures.h Holds the Haar-like features that are computed by the HaarLearner.
*/

#ifndef __HAAR_FEATURES_H
#define __HAAR_FEATURES_H

#include "classifier/boosting/Defaults.h" // for MB_DEBUG
#include "classifier/boosting/Utils/Utils.h" // for Rect
#include "classifier/boosting/Others/Example.h"

#include <vector>
#include <map>
#include <set>

using namespace std;

namespace MultiBoost {

//////////////////////////////////////////////////////////////////////////

/**
* The type of features.
*/
enum eFeatureType 
{
   FEATURE_NO_TYPE, //!< No type specified. 
   FEATURE_2H_RECT, //!< Two horizontal.
   FEATURE_2V_RECT, //!< Two vertical. 
   FEATURE_3H_RECT, //!< Three horizontal.
   FEATURE_3V_RECT, //!< Three vertical.
   FEATURE_4SQUARE_RECT //!< Four squares feature 
};

/**
* The type of access to the configurations. This access can be full,
* that is all the possible configurations of the features (position and
* shape) are explored. The other access type creates a random sampling
* of the configurations, because the full search would be too expensive.
* The random sampling is guaranteed to be homogeneous among the space of
* configurations.
* \date 27/12/2005
*/ 
enum eAccessType
{
   AT_FULL, //!< Full search among all the configurations. 
   AT_RANDOM_SAMPLING //!< Random sampling of the configurations. 
};

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

/**
* This class holds a Haar-like feature type. Each type will have a set
* of possible "configurations", that is its position and size
* inside a window of width x height size. For instance, here is an
* example of 3 different configurations of a "2 vertical feature" inside
* a window of 24x24 pixels:
\verbatim
                24 pixels
   +--------------------------------+
   |                   +-+          |
   |   +--------+      | |          |
   |   |________|      |_|          |
   |   |        |      | |          |
   |   +--------+      | |          |
   |                   +-+          |  24 pixels
   |    +-------------------------+ |
   |    |                         | |
   |    |_________________________| |
   |    |                         | |
   |    |                         | |
   |    +-------------------------+ |
   +--------------------------------+
\endverbatim
* The size of the window depend on the integral image loaded by HaarData.
* For more information on Haar-like features, please see Viola and
* Jones 2001 paper (Robust Real-time Object Detection).
* \see HaarLearner
* \see HaarData
* \date 27/12/2005
*/
class HaarFeature
{
public:

   //////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////

   /**
   * Holds the information about the registered Haar features. Works pretty
   * much like a class factory.
   * \see REGISTER_HAAR_FEATURE
   */
   class FeaturesRegs
   {
   public:

      /**
      * Add a feature type to the list of the registered features.
      * \param shortName The short name of the feature type. It will be used
      * to retrieve it when necessary.
      * \param pFeatureToRegister The pointer to the object that contain the feature
      * to be registered. The object will be an instantiation of the derived type from
      * which, thanks to the method \a HaarFeature::create() it will be possible
      * to obtain other objects of the same derived type.
      * \date 27/12/2005
      */
      void addFeature(const string& shortName, HaarFeature* pFeatureToRegister)
      { _features[shortName] = pFeatureToRegister; }

      /**
      * Ask if a feature with the given shortName has been registered.
      * \param shortName The short name of the feature type.
      * \return true if the feature with the given short name exists,
      * false otherwise.
      * \see addFeature
      * \date 27/12/2005
      */
      bool hasFeature(const string& shortName)
      { return ( _features.find(shortName) != _features.end() ); }

      /**
      * Get the registered object with the given shortName.
      * \param shortName The short name of the feature type.
      * \return The object of the registered feature.
      * \see addFeature
      * \date 27/12/2005
      */
      HaarFeature* getFeature(const string& shortName)
      { return _features[shortName]; }

      /**
      * Get a string with the list of the registered Haar-like features.
      * \return The string with the list of registered Haar-like features
      * (Note: this list will be created with the short names used for
      * the registration).
      * \date 27/12/2005
      */
      string getRegString()
      {
         string regList;
         map<string, HaarFeature*>::const_iterator it;
         for (it = _features.begin(); it != _features.end(); ++it)
            regList += it->first;
         return regList;
      }

   private:
      map<string, HaarFeature*> _features; //!< The map of the registered haar features. 
   };

   /**
   * Map of the registered Haar-like features.
   * This data is updated statically just by adding the macro
   * REGISTER_HAAR_FEATURE(X, Y) where X is the short name of 
   * the feature and Y is the enum type (see eFeatureType)
   * Example (in file HaarFeatures.cpp):
   * \code
   * REGISTER_HAAR_FEATURE(2h, HaarFeature_2H);
   * \endcode
   * \remark To prevent the "static initialization order fiasco"
   * I am using the trick described in http://www.parashift.com/c++-faq-lite/ctors.html#faq-10.13
   * \see create()
   * \see FeaturesRegs
   * \date 27/12/2005
   */
   static FeaturesRegs& RegisteredFeatures()
   {
      // Construct On First Use Idiom:
      // Since static local objects are constructed the first time control flows 
      // over their declaration (only), the new HaarFeaturesRegs() statement will only 
      // happen once: the first time RegisteredHaarFeatures() is called. Every subsequent 
      // call will return the same LearnersRegs object (the one pointed to by ans). 
      static FeaturesRegs* regFeatures = new FeaturesRegs();
      return *regFeatures;
   }

   //////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////

   /**
   * The constructor. It just initializes the variables.
   * \param width The width of the Haar-like feature.
   * \param height The height of the Haar-like feature.
   * \param shortName The short name of the feature. This name will be used for serialization
   * and should be short and without spaces.
   * \param name The name of the feature.
   * \param type The type of feature.
   * \date 27/12/2005
   */
   HaarFeature(short width = 0, short height = 0, const string& shortName = "", 
               const string& name = "", eFeatureType type = FEATURE_NO_TYPE);

   virtual ~HaarFeature(){}

   /**
   * Returns a new object of the derived type.
   * For instance the overriding of this method in HaarFeature_2H
   * will be:
   * \code
   * return new HaarFeature_2H(_shortName)
   * \endcode
   * \remark It uses the trick described in http://www.parashift.com/c++-faq-lite/serialization.html#faq-36.8
   * for the auto-registering classes.
   * \date 27/12/2005
   */
   virtual HaarFeature* create() = 0;

   /**
   * Get the short name of the feature type.
   * \return The short name of the feature type.
   * \date 27/12/2005
   */
   string       getShortName()  const { return _shortName; }

   /**
   * Get the full name of the feature type.
   * \return The full name of the feature type.
   * \date 27/12/2005
   */
   string       getName()       const { return _name; }

   /**
   * Get the type of the feature.
   * \return The type of the feature.
   * \see eFeatureType
   * \date 27/12/2005
   */
   eFeatureType getType()       const { return _type; }

   /**
   * Set the type of access for the configurations.
   * \param accessType The type of access, which can be full or random.
   * \see eAccessType
   * \date 27/12/2005
   */
   void setAccessType(eAccessType accessType) { _accessType = accessType; }

   /**
   * Perform the pre-computation of all the possible configurations of
   * the feature (position and size within the window). 
   * Once they will be ready, an iterator is positioned at the beginning 
   * of the list.
   * \see HaarFeature
   * \see moveToNextConfig
   * \see hasConfigs
   * \see resetConfigIterator
   * \see getCurrentConfig
   * \date 27/12/2005
   */
   int precomputeConfigs();

   /**
   * Return the configuration on the current position of the iterator
   * over the list of pre-computed configurations.
   * \return The rectangle with the position and size of the current
   * configuration.
   * \see HaarFeature
   * \date 27/12/2005
   */
   const nor_utils::Rect& getCurrentConfig() const { return *_configIt; }

   /**
   * Move the iterator to the next configuration. If the access type is
   * set to AT_FULL, the next configuration will be simply the next on the list.
   * If the access type is AT_RANDOM_SAMPLING, the next configuration will be
   * a random one which has \b not been visited yet.
   * \date 27/12/2005
   * \todo Replace the "visited" list with something less awful.
   */
   void moveToNextConfig();

   /**
   * Check if there is another configuration available.
   * \return true if there is another configuration, otherwise false.
   * \date 27/12/2005
   */
   bool hasConfigs() const;

   /**
   * Reset the iterator over the list of configurations.
   * \see moveToNextConfig
   * \see getCurrentConfig
   * \see precomputeConfigs
   * \date 27/12/2005
   */
   void resetConfigIterator();

   /**
   * Convert the vector of all the examples (in integral image format) into
   * a vector of features outputs, using the current configuration.
   * The pair represent the original index of the example, where the second
   * element is the feature's output.
   * \remark This conversion could be done example-by-example with the 
   * getValue method, but a virtual call for each example would be too
   * slow. Instead the type of Haar-like feature is resolved within this
   * method, and then the filling is performed statically by _fillHaarData.
   * \param intImages The vector with the examples to be converted.
   * \param haarData The returned vector of features outputs.
   * \see getValue
   * \see _fillHaarData
   * \see getCurrentConfig
   * \date 27/12/2005
   */
   void fillHaarData( const vector<Example>& intImages, // in
                      vector< pair<int, float> >& haarData ); // out

   /**
   * Get the feature output given a single example (in integral image format).
   * It will be overridden by the derived classes. 
   * \remark Because it is a virtual function it is quite slow. If the purpose
   * is to "convert" a whole vector of examples, please use fillHaarData.
   * \param pIntImage The array with the integral image data of the single example.
   * \param r The configuration that will be used to compute the value of the
   * feature.
   * \see HaarFeature
   * \date 27/12/2005
   */
   virtual float getValue(const vector<float>& intImage, const nor_utils::Rect& r) = 0;

   virtual int getLoadedConfigIndex() { return _loadedConfigIndex; }
   virtual void loadConfigByNum( int idx ); 

protected:

   /**
   * Return the integral image sum of an area which starts at 0,0 and ends
   * at x,y. 
   * \param pIntImage The pointer to the array of data which represent the integral
   * image of the example.
   * \param x The x coordinate.
   * \param y The y coordinate.
   * \see HaarData
   * \date 27/12/2005
   */
   float getSumAt(const vector<float>& intImage, int x, int y);

   string       _shortName; //!< The short name of the Haar-like feature. 

private:

   short _width;  //!< The width of the Haar-like feature. 
   short _height; //!< The height of the Haar-like feature. 

   string       _name; //!< The full name of the Haar-like feature. 
   eFeatureType _type; //!< The type of the Haar-like feature. 


   /**
   * The type of access for the configurations.
   * \see eAccessType
   * \date 27/12/2005
   */
   eAccessType _accessType; 

   vector<nor_utils::Rect> _precomputedConfigs; //!< The precomputed sizes and locations of the features.

   /**
   * The list of the configurations that have already been used when the access type is
   * AT_RANDOM_SAMPLE.
   * \see moveToNextConfig
   * \see eAccessType
   * \date 27/12/2005
   */
   vector<char> _visitedConfigs;

   /**
   * Keep the number of configurations that have already been used when the access type
   * is AT_RANDOM_SAMPLE.
   * \see moveToNextConfig
   * \see eAccessType
   * \date 4/1/2006
   */
   size_t       _numVisited;

   /**
   * The iterator of the current position in the list of configurations.
   * \remark Used only with "full search".
   * \see getNextFeature
   * \date 3/12/2005
   */
   vector<nor_utils::Rect>::const_iterator _configIt;

   int _loadedConfigIndex; 

   /**
   * Convert the vector of all the examples (in integral image format) into
   * a vector of features outputs, using the current configuration.
   * The pair represent the original index of the example, where the second
   * element is the feature's output. This method statically call the derived
   * method to avoid the virtual calling.
   * \param intImages The vector with the examples to be converted.
   * \param haarData The returned vector of features outputs.
   * \see fillHaarData.
   * \date 27/12/2005
   */
   template <typename TDeriv>
   void _fillHaarData( const vector<Example>& intImages, // input 
                       vector< pair<int, float> >& haarData ) // output
   {
      int i;
      vector<Example>::const_iterator iiIt;
      const vector<Example>::const_iterator iiEnd = intImages.end();

      vector< pair<int, float> >::iterator hIt;
      const nor_utils::Rect& currConfig = getCurrentConfig();

      for (iiIt = intImages.begin(), hIt = haarData.begin(), i = 0;
           iiIt != iiEnd; ++iiIt, ++hIt, ++i)
      {
         hIt->first = i;
         hIt->second = static_cast<TDeriv&>(*this).getValue(iiIt->getValues(), currConfig);
      }
   }
};

// ------------------------------------------------------------------------------

/**
* Type "2 Blocs Horizontal" feature.
\verbatim
 For white:
 1      2   
  +-----++-----+
  | [W] ||  B  |
 3+-----4+-----+
        ^-- xHalfPos
 For black:
        1      2
  +-----++-----+
  |  W  || [B] |
  +-----3+-----4
         ^-- xHalfPos
\endverbatim
* (W = White, B = Black)
* The formula is: 4+1 - (2+3)
* \date 27/12/2005
*/
class HaarFeature_2H : public HaarFeature
{
public:

   /**
   * The constructor. Defines the name, type and shape of the feature.
   * \param shortName The short name of the feature.
   * \date 27/12/2005
   */
   HaarFeature_2H(const string& shortName) 
      : HaarFeature(2, 1, shortName, "2 Blocs Horizontal", FEATURE_2H_RECT) {}

   /**
   * Returns itself as object.
   * \remark It uses the trick described in http://www.parashift.com/c++-faq-lite/serialization.html#faq-36.8
   * for the auto-registering classes.
   * \see FeaturesRegs
   * \date 27/12/2005
   */
   HaarFeature* create() 
   { return new HaarFeature_2H(_shortName); }

   /**
   * Transform the example in integral image format, into the scalar
   * output of the feature.
   * \param pIntImage The array with the integral image data of the single example.
   * \param r The configuration that will be used to compute the value of the
   * feature.
   * \date 27/12/2005
   */
   virtual float getValue(const vector<float>& intImage, const nor_utils::Rect& r);
};

// ------------------------------------------------------------------------------

/**
* Type "2 Blocs Vertical" feature.
\verbatim
 For white:
   1        2
    +-------+
    |  [W]  |
   3+-------4
    |   B   |
    +-------+
 For black:
    +-------+
   1|   W   2
    +-------+
    |  [B]  |
   3+-------4
\endverbatim
* (W = White, B = Black)
* The formula is: 4+1 - (2+3)
* \date 27/12/2005
*/
class HaarFeature_2V : public HaarFeature
{
public:

   /**
   * The constructor. Defines the name, type and shape of the feature.
   * \param shortName The short name of the feature.
   * \date 27/12/2005
   */
   HaarFeature_2V(const string& shortName) 
      : HaarFeature(1, 2, shortName, "2 Blocs Vertical", FEATURE_2V_RECT) {}

   /**
   * Returns itself as object.
   * \remark It uses the trick described in http://www.parashift.com/c++-faq-lite/serialization.html#faq-36.8
   * for the auto-registering classes.
   * \see FeaturesRegs
   * \date 27/12/2005
   */
   HaarFeature* create() 
   { return new HaarFeature_2V(_shortName); }

   /**
   * Transform the example in integral image format, into the scalar
   * output of the feature.
   * \param pIntImage The array with the integral image data of the single example.
   * \param r The configuration that will be used to compute the value of the
   * feature.
   * \date 27/12/2005
   */
   virtual float getValue(const vector<float>& intImage, const nor_utils::Rect& r);
};

// ------------------------------------------------------------------------------

/**
* Type "3 Blocs Horizontal" feature.
\verbatim
 For white left:
   1      2
    +-----+-----+-----+
    | [W] |  B  |  W  |
   3+-----4-----+-----+
 For black:
         1      2
    +-----+-----+-----+
    |  W  | [B] |  W  |
    +----3+-----4-----+
 For white right:
               1      2
    +-----+-----+-----+
    |  W  |  B  | [W] |
    +-----+----3+-----4
\endverbatim
* (W = White, B = Black)
* The formula is: 4+1 - (2+3)
* \date 27/12/2005
*/
class HaarFeature_3H : public HaarFeature
{
public:

   /**
   * The constructor. Defines the name, type and shape of the feature.
   * \param shortName The short name of the feature.
   * \date 27/12/2005
   */
   HaarFeature_3H(const string& shortName) 
      : HaarFeature(3, 1, shortName, "3 Blocs Horizontal", FEATURE_3H_RECT) {}

   /**
   * Returns itself as object.
   * \remark It uses the trick described in http://www.parashift.com/c++-faq-lite/serialization.html#faq-36.8
   * for the auto-registering classes.
   * \see FeaturesRegs
   * \date 27/12/2005
   */
   HaarFeature* create() 
   { return new HaarFeature_3H(_shortName); }

   /**
   * Transform the example in integral image format, into the scalar
   * output of the feature.
   * \param pIntImage The array with the integral image data of the single example.
   * \param r The configuration that will be used to compute the value of the
   * feature.
   * \date 27/12/2005
   */
   virtual float getValue(const vector<float>& intImage, const nor_utils::Rect& r);
};

// ------------------------------------------------------------------------------

/**
* Type "3 Blocs Vertical" feature.
\verbatim
 For white top               For white bottom
  1      2       For black 
   +-----+        +-----+        +-----+
   | [W] |       1|  W  2        |  W  |
  3+-----4        +-----+        +-----+
   |  B  |        | [B] |       1| [B] 2
   +-----+       3+-----4        +-----+
   |  W  |        |  W  |        |  W  |
   +-----+        +-----+       3+-----4
\endverbatim
* (W = White, B = Black)
* The formula is: 4+1 - (2+3)
* \date 27/12/2005
*/
class HaarFeature_3V : public HaarFeature
{
public:

   /**
   * The constructor. Defines the name, type and shape of the feature.
   * \param shortName The short name of the feature.
   * \date 27/12/2005
   */
   HaarFeature_3V(const string& shortName) 
      : HaarFeature(1, 3, shortName, "3 Blocs Vertical", FEATURE_3V_RECT) {}

   /**
   * Returns itself as object.
   * \remark It uses the trick described in http://www.parashift.com/c++-faq-lite/serialization.html#faq-36.8
   * for the auto-registering classes.
   * \see FeaturesRegs
   * \date 27/12/2005
   */
   HaarFeature* create() 
   { return new HaarFeature_3V(_shortName); }

   /**
   * Transform the example in integral image format, into the scalar
   * output of the feature.
   * \param pIntImage The array with the integral image data of the single example.
   * \param r The configuration that will be used to compute the value of the
   * feature.
   * \date 27/12/2005
   */
   virtual float getValue(const vector<float>& intImage, const nor_utils::Rect& r);
};

// ------------------------------------------------------------------------------

/**
* Type "4 Blocs Square" feature.
\verbatim
 For white top-left:     For black top-right:
   1      2                     1      2
    +-----+-----+          +-----+-----+
    | [W] |  B  |          |  W  | [B] |
   3+-----4-----+          +----3+-----4
    |  B  |  W  |          |  B  |  W  |
    +-----+-----+          +-----+-----+

 For black bottom-left:  For white bottom-right:
           
    +-----+-----+          +-----+-----+
   1|  W  2  B  |          |  W 1|  B  2
    +-----+-----+          +-----+-----+
    | [B] |  W  |          |  B  | [W] |
   3+-----4-----+          +----3+-----4
\endverbatim
* (W = White, B = Black)
* The formula is: 4+1 - (2+3)
* \date 27/12/2005
*/
class HaarFeature_4SQ : public HaarFeature
{
public:

   /**
   * The constructor. Defines the name, type and shape of the feature.
   * \param shortName The short name of the feature.
   * \date 27/12/2005
   */
   HaarFeature_4SQ(const string& shortName) 
      : HaarFeature(2, 2, shortName, "4 Blocs Square", FEATURE_4SQUARE_RECT) {}

   /**
   * Returns itself as object.
   * \remark It uses the trick described in http://www.parashift.com/c++-faq-lite/serialization.html#faq-36.8
   * for the auto-registering classes.
   * \see FeaturesRegs
   * \date 27/12/2005
   */
   HaarFeature* create() 
   { return new HaarFeature_4SQ(_shortName); }

   /**
   * Transform the example in integral image format, into the scalar
   * output of the feature.
   * \param pIntImage The array with the integral image data of the single example.
   * \param r The configuration that will be used to compute the value of the
   * feature.
   * \date 27/12/2005
   */
   virtual float getValue(const vector<float>& intImage, const nor_utils::Rect& r);
};

// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

/**
* The macro that \b must be declared by all the Derived classes that can be used
* as Haar-like features.
* \see HaarFeature::FeaturesRegs
*/
#define REGISTER_HAAR_FEATURE(SHORTNAME, X) \
struct RegisterHF_##X \
{ RegisterHF_##X() { HaarFeature::RegisteredFeatures().addFeature(#SHORTNAME, new X(#SHORTNAME)); } }; \
        static RegisterHF_##X rHF_##X;

// ------------------------------------------------------------------------------

} // end of namespace Multiboost

#endif // __HAAR_FEATURES_H
