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
* \file StreamTokenizer.h Tokenize a stream.
*/

#ifndef __STREAM_TOKENIZER_H
#define __STREAM_TOKENIZER_H

#include <string>
#include <iterator>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace nor_utils {

/**
* Can tokenize a stream.
* It takes a stream, and defined the delimiters, it returns the tokens between these
* delimiters.
* \date 16/11/2005
*/
class StreamTokenizer
{
   typedef istream_iterator<char> sIt; //!< A stream iterator typedef

public:

   /**
   * The constructor. It defines the delimiters, and use the normal
   * white-spaces on the stream.
   * \param is The input stream.
   * \param delim The delimiters.
   * \date 16/11/2005
   */
   StreamTokenizer(istream& is, const string delim = " \t")
      : p(is), end_of_stream(), delimiters(delim) 
   { is.unsetf(std::ios::skipws); }

   /**
   * Get the next token.
   * \return The string of the next token in the list.
   * \date 16/11/2005
   */
   string next_token()
   {
      string result;
      char ch;
      insert_iterator<string> iIt(result, result.begin());

      if ( p != end_of_stream)
      {
         // move until the delimiter
         while ( p != end_of_stream && is_delimiter(*p) )
         {
            ch = *p;
            p++;
         }

         // move until the next delimiter, and build
         // the string.
         while ( p != end_of_stream && !is_delimiter(*p) )
         {
            ch = *p;
            *iIt++ = *p++;
         }
      }

      return result;
   }

   /**
   * Ask if the stream has other tokens.
   * \return True if it is at the end of the stream, false
   * otherwise.
   * \date 16/11/2005
   * \bug In fact this method does not know if there are other tokens
   * awaiting. If there are characters after the last delimiter
   * at the end of the stream, they will be considered a token
   * and therefore, this method will return true. This happens
   * also if there is just a newline (in the case it is not a delimiter).
   * It is possible to avoid this problem, but will affect efficency.
   * Because a "spurious" token will just be ignored by the parser
   * I decided to just leave it as it is.
   */
   bool has_token()
   { return (! (p == end_of_stream) ); }

private:

   /**
   * Fake assignment operator to avoid warning.
   * \date 6/12/2005
   */
   StreamTokenizer& operator=( const StreamTokenizer& ) {return *this;}

   sIt p; //!< The current position in the stream.
   sIt end_of_stream; //!< The end of the stream.

   const string delimiters; //!< The delimiters. 

   /**
   * Checks if the given char is a delimiter.
   * \param c The char to be checked
   * \return True if \a c is a delimiter, false otherwise.
   * \date 16/11/2005
   */
   bool is_delimiter(char c)
   { 
      return delimiters.find(c) != string::npos; 
   }
};

} // end of namespace nor_utils

#endif //__STREAM_TOKENIZER_H

