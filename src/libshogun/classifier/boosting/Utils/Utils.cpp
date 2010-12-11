
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

#include "Utils.h"
#include <cctype> // for isspace
#include <algorithm>
#include <sstream>
#include <fstream>

namespace nor_utils {

// ----------------------------------------------------------------


string getEscapeSequence(const string& inStr)
{
   string result;
   insert_iterator<string> resIt(result, result.begin());

   string::const_iterator it;

   for (it = inStr.begin(); it != inStr.end(); ++it)
   {
      if (*it == '\\')
      {
         // escape char!
         ++it;
         switch ( *it )
         { 
         case 'a': // Bell (alert)
            *resIt++ = '\a';
            break;
         case 'b': // Backspace
            *resIt++ = '\t';
            break;
         case 'f': // Formfeed
            *resIt++ = '\f';
            break;
         case 'n': // New line
            *resIt++ = '\f';
            break;
         case 'r': // Carriage return
            *resIt++ = '\f';
            break;
         case 't':
            *resIt++ = '\t'; // Horizontal Tab
            break;
         case 'v': // Vertical tab
            *resIt++ = '\f';
            break;
         case '\'': // Single quotation mark
            *resIt++ = '\'';
            break;
         case '\"': // float quotation mark
            *resIt++ = '"';
            break;
         case '\\': // Backslash
            *resIt++ = '\\';
            break;
         case '\?': // Literal question mark
            *resIt++ = '\?';
            break;
         }

      }
      else
         *resIt++ = *it;
   }

   return result;
}

// ----------------------------------------------------------------

void skip_line(istream& inFile, int nLines)
{
   for (int i = 0; i < nLines; ++i)
      while ( inFile.get() != '\n' && !inFile.eof() );
}

// ----------------------------------------------------------------

string addAndCheckExtension(const string& file, const string& extension)
{
   // if file smaller than the extension
   // or
   // if the file has no extension whatsoever
   // then return it with the extension
   if ( file.length() <= extension.length() ||
      file.rfind('.') == string::npos )
      return file + "." + extension;

   // get the extension
   size_t pos = file.rfind('.') + 1;
   string fileExt = file.substr(pos, file.length() - pos);

   if (fileExt == extension)
      return file; // extension is the same. Just return the file name
   else
      return file + "." + extension; // return filename plus extension
}

// ----------------------------------------------------------------

string trim(const string& str)
{
   size_t beg, end;

   string::const_iterator fIt = str.begin();
   string::const_reverse_iterator rIt = str.rbegin();

   for (beg = 0; isspace( *(fIt++) ); ++beg);
   for (end = str.length(); isspace( *(rIt++) ); --end);

   return str.substr(beg, end-beg);  
}

// ----------------------------------------------------------------

string int2string( const int i ){
	string s;
	stringstream ss;
	ss << i;
	s = ss.str();
	return s;
}


// ----------------------------------------------------------------

bool cmp_nocase(const string &s1, const string &s2)
{
   if (s1.length() != s2.length())
      return false;

   string::const_iterator p1 = s1.begin();
   string::const_iterator p2 = s2.begin();

   const string::const_iterator s1End = s1.end();
   const string::const_iterator s2End = s2.end();

   while (p1 != s1End && p2 != s2End )
   {
      if (toupper(*p1) != toupper(*p2) )
         return false;

      // just for clearity
      ++p1;
      ++p2;
   }

   return true;
}

// ----------------------------------------------------------------

size_t count_columns(istream& in)
{
   size_t nCols = 0;
   bool inCol = false;
   int c = 0;

   // Remember current position
   ios::pos_type currPos = in.tellg();

   while( !in.eof() )
   {
      c = in.get();

      if ( c == '\n' || c == '\r' )
         break;

      if (isspace(c))
      {
         if (inCol)
            inCol = false;
      }
      else
      {
         if (!inCol)
         {
            inCol = true;
            ++nCols;
         }
      }

   }

   // Put read pointer back to where it was
   in.seekg(currPos, ios::beg);
   in.clear();

   return nCols;
}

// ----------------------------------------------------------------

size_t count_rows( istream& in, bool from_start /*= false*/ )
{
   size_t nRows = 0;

   // Remember current position
   ios::pos_type currPos = in.tellg();
   if ( from_start )
      in.seekg(0, ios::beg);

   nRows = std::count( std::istreambuf_iterator<char>(in),
                       std::istreambuf_iterator<char>(),
                       '\n');

   // Put read pointer back to where it was
   in.seekg(currPos, ios::beg);
   in.clear();

   return nRows;
}

// ----------------------------------------------------------------

bool is_number(const string& str)
{
   string::const_iterator it;
   const string::const_iterator endIt = str.end();

   it = str.begin();

   // empty string!
   if (it == endIt)
      return false;

   // check for a sign
   if (*it == '+' || *it == '-')
      ++it;

   // just a sign!
   if (it == endIt)
      return false;

   // check for a string of digits
   while ( it != endIt && isdigit(*it) )
      ++it;

   // no decimal point nor anything else: just numbers!
   if (it == endIt)
      return true;

   // check for a decimal point
   if (*it == '.') 
   {
      ++it;
      while ( it != endIt && isdigit(*it) )
         ++it;
   }
   else
      return false;

   // no exponent
   if (it == endIt)
      return true;

   // check for an exponent
   if ( (*it == 'E' || *it == 'e') )
   {
      ++it;

      // check for a sign
      if ( it != endIt && (*it == '+' || *it == '-') )
         ++it;

      for ( ; it != endIt; ++it)
      {
         // everything in the exponent must be a number
         if ( !isdigit(*it) )
            return false;
      }
   }
   else
      return false;

   return true;
}

// ----------------------------------------------------------------

string getAlphanumeric(int num)
{
   // convert the number from base ten to base 26, that is the
   // alphanumeric range A-Z

   const int numAscii = 26;
   const int startAscii = 65;

   if (num == 0)
      return "A";

   string res;
   // perform the conversion and build the string
   while (num > 0)
   {
      res += static_cast<char>( startAscii + num % numAscii );
      num /= numAscii;
   }
   
   return res;
}

// ----------------------------------------------------------------

size_t getFileSize(ifstream& in)
{
   // Remember current position
   ios::pos_type currPos = in.tellg();

   // Seek to end of file, get position
   in.seekg(0, ios::end);
   ios::pos_type endPos = in.tellg();

   // Put read pointer back to where it was
   in.seekg(currPos, ios::beg);

   return static_cast<size_t>( endPos );
}

// ----------------------------------------------------------------



} // end of namespace nor_utils

