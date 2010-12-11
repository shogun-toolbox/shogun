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
* \file Utils.h Some useful functions.
*/

#pragma warning( disable : 4786 )


#ifndef __UTILS_H
#define __UTILS_H

#include <cstring>
#include <locale>
#include <string>
#include <functional>
#include <limits>
#include <vector>
#include <iostream>
#include <sstream>

using namespace std;


//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace nor_utils
{

// ----------------------------------------------------------------

/**
* Simple structure of a rectangle.
* \date 27/12/2005
*/
struct Rect
{
   /**
   * The constructor that set everything to 0.
   */
   Rect() : x(0), y(0), width(0), height(0) {}
   /**
   * The constructor that ask for the sizes.
   */
   Rect(short x, short y, short width, short height) 
        : x(x), y(y), width(width), height(height) {}
   short x; //!< x position.
   short y; //!< y position.
   short width; //!< Width 
   short height; //!< Height 
};

// ----------------------------------------------------------------

/**
* Set the white_spaces to anything given in the constructor and newlines only.
* Use it with: 
* \code 
* "ifstream".imbue( locale(locale(), new nor_utils::white_spaces("\t ,.") ); 
* \endcode
* \date 11/11/2005
*/
struct white_spaces : ctype<char>
{
   /**
   * Create the white_spaces object, and initialize the table.
   * \remark If no argument is provided, the only white space allowed will be "\n"!
   * \date 12/11/2005
   */
   white_spaces(const string& sepChars = "", const bool doNewLine = true) 
      : ctype<char>( init(sepChars, doNewLine) ) {}
   ~white_spaces() { if (_rc) delete _rc; } //!< The destructor. Clear the table. 

   /**
   * Initialize the table only with the give sepChars as whitespaces.
   * \return The modified table.
   * \date 12/11/2005
   */
   ctype_base::mask const* init(const string& sepChars, const bool doNewLine)
   {
      _rc = new ctype_base::mask[ctype<char>::table_size];

      // not using fill_n because I don't wanna get bothered by the warning
      // of the VC8 compiler. This does not need to be secured! :P
      for (size_t i = 0; i < ctype<char>::table_size; ++i)
         _rc[i] = ctype_base::mask();

      //_rc['\t'] = ctype_base::space;
      for (string::const_iterator it = sepChars.begin(); it != sepChars.end(); ++it)
         _rc[*it] = ctype_base::space;

      if (doNewLine)
         _rc['\n'] = ctype_base::space;

      return _rc;
   }

   /**
   * Get the local table.
   * \return The local table.
   * \date 12/11/2005
   * \see init()
   */
   ctype_base::mask const* get_table() {  return _rc;  }

private:
   ctype_base::mask* _rc; //!< An array containing the local table 
};

// ----------------------------------------------------------------

/**
* Convert the string into escape sequences. For instance if a string
* contains "\t\n", it will return another string with a tab and a 
* new line. This is useful in case of command line input.
* \param inStr The string to be converted.
* \return The new string with the escape sequences.
* \remark Only simple sequences are converted. \\x \\o and some others are not
* supported.
* \date 15/1/2006
*/
string getEscapeSequence(const string& inStr);

// ----------------------------------------------------------------

/**
* Skip \a nLine lines of a given stream.
* \param inFile The input stream.
* \param nLines The number of lines to skip (default = 1).
* \date 11/11/2005
*/
void skip_line(istream& inFile, int nLines = 1);

// ----------------------------------------------------------------

/**
* Returns the file with the extension. If the given extension already exist
* in the file name, then ignore it.
* Examples:
* \code
* addAndCheckExtension("hello.dat", "dat"); // -> "hello.dat"
* addAndCheckExtension("hello", "dat"); // -> "hello.dat"
* addAndCheckExtension("hello.txt", "dat"); // -> "hello.txt.dat"
* \endcode
* \param file The fileName.
* \param extension The extension to append.
* \return The string with filename and extension.
* \date 11/11/2005
*/
string addAndCheckExtension(const string& file, const string& extension);

// ----------------------------------------------------------------

/**
* Trim a string on the left and on the right.
* \param str The string to be trimmed.
* \return The trimmed string.
* \date 11/11/2005
*/
string trim(const string& str);

// ----------------------------------------------------------------

string int2string( const int i );


// ----------------------------------------------------------------

// 
/**
* Case insensitive comparison.
* \param s1 The first string.
* \param s2 The second string.
* \return True if the two string are equal (regardless of the case), false otherwise.
* \date 16/11/2005
*/
bool cmp_nocase(const string &s1, const string &s2);

// ----------------------------------------------------------------

/**
* Count the number of columns from a stream.
* \param in The stream to be examined.
* \return The number of columns found.
* \remark The stream position will \b not be modified
* \date 11/11/2005
*/
size_t count_columns(istream& in);

// ----------------------------------------------------------------

/**
* Count the number of rows from a stream.
* \param in The stream to be examined.
* \param from_start If true, it will start from the beginning of the file
* \return The number of columns found.
* \remark The stream position will \b not be modified
* \date 11/11/2005
*/
size_t count_rows(istream& in, bool from_start = false);

// ----------------------------------------------------------------

/**
* Checks if a number is equal to zero, between a positive and negative
* limit of \a smallVal. Should be used with floating numbers to overcome
* numerical problems.
* \param val The value to check.
* \param smallVal An arbitrary small value that defines the positive and negative limits.
* \return true if it is between the limits, false otherwise.
* \remark The type \b must be float (float, float, etc).
* \date 11/11/2005
*/
template < typename T >
bool is_zero( T val, T smallVal = 1E-10 )
{
   if ( val <= smallVal &&
      val >= -smallVal)
      return true;
   else
      return false;
}

// ----------------------------------------------------------------

/**
* Check if the given string is a legal number.
* \param str The string to check.
* \return \a true if the string contains a legal number, \a false otherwise.
* \date 10/2/2006
*/
bool is_number( const string& str );

// ----------------------------------------------------------------

/**
* Convert the number from base ten to base 26 that uses only letters, 
* that is the alphanumeric range A-Z. 
* Examples:
* \code
* getAlphanumeric(1); // -> "A"
* getAlphanumeric(21); // -> "V"
* getAlphanumeric(3458); // -> "ADF"
* \endcode
* \param num The number to be converted.
* \return The same number in base 26.
* \date 10/2/2006
*/
string getAlphanumeric(int num);

// ----------------------------------------------------------------

/**
* Compute the size of a file (in bytes).
* \param in The stream of the file.
*/
size_t getFileSize(ifstream& in);

// ----------------------------------------------------------------

/**
* Simply returns the sign of a number.
*/
template <typename T>
char sign(T val)
{
   if ( val == 0 )
      return 0;
   else
      return val < 0 ? -1: +1;
}

// ----------------------------------------------------------------
// ----------------------------------------------------------------

/**
* Sort pairs on the first or second element. The predicate gives the sorting direction.
* 
* To sort from small to big:
* \code
* vector< pair<int, float> > v;
* sort( v.begin(), v.end(), nor_utils::comparePair< 2, float, less<float> > );
* \endcode
* From big to small:
* \code
* vector< pair<int, float> > v;
* sort( v.begin(), v.end(), nor_utils::comparePair< 2, float, greater<float> > );
* \endcode
* \remark Don't forget to include \<functional\> if you use the standard predicates
* \remark I could have done a predicate for the element of the pair too,
* but they are just two, and I don't really need the first.. :P
* \remark T1 The type of the first element of the pair
* \remark T2 The type of the second element of the pair
* \remark Pred The predicate used for the comparison
* \param el1 The first pair
* \param el2 The second pair
* \date 11/11/2005
*/
template <int N, typename T1, typename T2, typename Pred>
struct comparePair;

// specialization
// sadly VC does not support template specialization with a default argument yet.. :P
template <typename T1, typename T2, typename Pred/* = greater<T1>*/ >
struct comparePair<1, T1, T2, Pred> : public binary_function<T1, T1, bool>
{
   bool operator()(const pair<T1, T2>& el1, const pair<T1, T2>& el2 ) const
   {
      return Pred()(el1.first, el2.first);
   }
};

template <typename T1, typename T2, typename Pred/* = greater<T2>*/ >
struct comparePair<2, T1, T2, Pred> : public binary_function<T2, T2, bool>
{
   bool operator()(const pair<T1, T2>& el1, const pair<T1, T2>& el2 ) const
   {
      return Pred()(el1.second, el2.second);
   }
};


template <int N, typename T1, typename T2, typename Pred>
struct comparePairP;


template <typename T1, typename T2, typename Pred/* = greater<T1>*/ >
struct comparePairP<1, T1, T2, Pred> : public binary_function<T1, T1, bool>
{
   bool operator()(const pair<T1, T2>* el1, const pair<T1, T2>* el2 ) const
   {
      return Pred()(el1->first, el2->first);
   }
};

template <typename T1, typename T2, typename Pred/* = greater<T2>*/ >
struct comparePairP<2, T1, T2, Pred> : public binary_function<T2, T2, bool>
{
   bool operator()(const pair<T1, T2>* el1, const pair<T1, T2>* el2 ) const
   {
      return Pred()(el1->second, el2->second);
   }
};


//template < typename T1, typename T2, typename Pred >
//bool comparePairOnSecond( const pair<T1, T2>& el1, const pair<T1, T2>& el2 )
//{ return Pred()(el1.second, el2.second); }


// ----------------------------------------------------------------


} // end of namespace nor_utils

#endif // __UTILS_H

