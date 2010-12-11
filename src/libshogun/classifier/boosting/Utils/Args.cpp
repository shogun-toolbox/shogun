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



#include "Args.h"
#include <stdlib.h>

namespace nor_utils {

// -----------------------------------------------------------------------

Args::~Args()
{
   map< string, vector<Argument*> >::iterator mIt;
   vector<Argument*>::iterator vIt;

   for (mIt = _groupedList.begin(); mIt != _groupedList.end(); ++mIt)
   {
      for (vIt = mIt->second.begin(); vIt != mIt->second.end(); ++vIt)
         delete (*vIt);
   }
}

// -----------------------------------------------------------------------

void Args::declareArgument( const string& name )
{
   _declArgs.insert( pair<string, Argument*>(name, new Argument(name)) );
}

// -----------------------------------------------------------------------

void Args::declareArgument( const string& name, const string& description, int numValues,
                            const string& valuesNamesList )
{
   // check if the discriminator has been set
   if ( _argDiscriminator.empty() )
   {
      cerr << "WARNING: declaration of arguments with values when NO discriminator has" << endl;
      cerr << "         been set, can be very dangerous! Please use setArgumentDiscriminator()!" << endl;
   }


   bool alreadyIn = false;
   // checks if it is not already in the list
   if ( _declArgs.find(name) != _declArgs.end() )
   {
      pair<mm_iterator, mm_iterator> range = _declArgs.equal_range(name);

      // check if the number of options is the same
      for (mm_iterator it = range.first; it != range.second; ++it)
      {
         if (it->second->numValues == numValues)
         {
            alreadyIn = true;
            break;
         }
      }
   }

   if (!alreadyIn)
   {
      Argument* pArg = new Argument(name, description, numValues, valuesNamesList);
      // not very efficient but speed is not an issue here
      _declArgs.insert( pair<string, Argument*>(name, pArg) );
      _groupedList[_currentGroup].push_back(pArg);
   }

}

// -----------------------------------------------------------------------

void Args::eraseDeclaration(const string& name)
{
   // checks if the argument has been declared
   mm_iterator declIt = _declArgs.find(name);

   if ( declIt == _declArgs.end() )
   {
      cerr << "ERROR: Cannot erase the argument <" << name << ">, as it has not been declared yet!" << endl;
      exit(1);
   }

   // checks if there are more declared with the same name
   if ( _declArgs.count(name) > 1 )
   {
      cerr << "ERROR: Args::eraseDeclaration called on an argument which has been declared multiple times." << endl
           << "       Please use Args::eraseDeclaration(string, int, bool) instead and provide the right" << endl
           << "       number of values to identify it." << endl;
      exit(1);
   }

   _declArgs.erase(declIt);

   // find it in the argument in the group map and erase it
   map< string, vector<Argument*> >::iterator gIt;
   for (gIt = _groupedList.begin(); gIt != _groupedList.end(); ++gIt)
   {
      vector<Argument*>& args = gIt->second;
      vector<Argument*>::iterator it;
      for (it = args.begin(); it != args.end(); ++it)
      {
         if ((*it)->name == name)
         {
            // delete the allocated space of the Argument object
            delete (*it);

            // erase the entry in the vector. Quite inefficient! :P
            args.erase(it);
            break;
         }
      }

   }

}

// -----------------------------------------------------------------------

void Args::eraseDeclaration(const string& name, const int numValues)
{
   bool argFound = false;
   
   pair<mm_iterator, mm_iterator> range = _declArgs.equal_range(name);
   for (mm_iterator it = range.first; it != range.second; ++it)
   {
      if ( it->second->numValues == numValues )
      {
         _declArgs.erase(it);
         argFound = true;
         break;
      }
   }

   if (!argFound)
   {
      cerr << "ERROR: Args::eraseDeclaration called on an argument which has either not been declared or declared" << endl
           << "       with a number of values different from the ones provided with the method call." << endl;
      exit(1);
   }

   // find it in the argument in the group map and erase it
   map< string, vector<Argument*> >::iterator gIt;
   for (gIt = _groupedList.begin(); gIt != _groupedList.end(); ++gIt)
   {
      vector<Argument*>& args = gIt->second;
      vector<Argument*>::iterator it;
      for (it = args.begin(); it != args.end(); ++it)
      {
         if ((*it)->name == name && (*it)->numValues == numValues)
         {
            // delete the allocated space of the Argument object
            delete (*it);

            // erase the entry in the vector. Quite inefficient! :P
            args.erase(it);
            break;
         }
      }

   }


}

// -----------------------------------------------------------------------

void Args::printGroup(const string& groupName, ostream& out, int indSpaces) const
{
   out << "\n" << groupName << ":" << endl;

   vector<Argument*>& args = _groupedList[groupName];

   vector<Argument*>::iterator it;
   for (it = args.begin(); it != args.end(); ++it)
   {
      for (int i = 0; i < indSpaces; ++i)
         out << ' ';

      string argName = _argDiscriminator + (*it)->name;
      out << argName;
      
      if (!(*it)->valuesNamesList.empty())
         out << " " << getWrappedString( (*it)->valuesNamesList, indSpaces + static_cast<int>(argName.length()) + 1, false);
      
      out << ":" << endl;
      out << getWrappedString((*it)->description, indSpaces+3) << endl;

   }

}

// -----------------------------------------------------------------------

ArgsOutType Args::readArguments( int argc, const char* argv[] )
{
   if (argc < 2 )
      return AOT_NO_ARGUMENTS;

   for (int i = 1; i < argc; ++i)
   {
      // check if the string is a legal argument
      if ( !hasArgumentDiscriminator(argv[i]) )
      {
         cerr << "ERROR: Expected argument, got value: " << argv[i] << endl;
         return AOT_UNKOWN_ARGUMENT;
      }

      string argName = getArgumentString(argv[i]);

      if ( _declArgs.find(argName) == _declArgs.end() )
      {
         cerr << "ERROR: Unknown argument " << _argDiscriminator << argName << endl;
         return AOT_UNKOWN_ARGUMENT;
      }

      // if the arg discriminator is empty (no arg discrimination), then
      // there is no way to determine if an option has the right number
      // of values. 
      if ( _argDiscriminator.empty() )
      {
         _resArgs[argName].push_back("");
         continue;
      }

      int numVals = 0; 

      // find the number of values the user has provided
      for (int j = i+1; j < argc && !hasArgumentDiscriminator(argv[j]); ++j, ++numVals);

      bool argFound = false;      

      pair<mm_iterator, mm_iterator> range = _declArgs.equal_range(argName);
      for (mm_iterator it = range.first; it != range.second; ++it)
      {
         if ( it->second->numValues == numVals )
         {
            argFound = true;
            break;
         }
      }

      if (!argFound)
      {
         cerr << "ERROR: The number of values for argument <" << argName << "> is incorrect!\n"
              << "Got:\n"
              << " " << _argDiscriminator << argName;
         for (int j = i+1; j <= i+numVals; ++j)
            cerr << " " << argv[j];

         cerr << "\nExpected:\n";
         mm_iterator lastElIt = range.second;
         --lastElIt;

         for (mm_iterator it = range.first; it != range.second; ++it)
         {
            cerr << " " << _argDiscriminator << argName << " "
                 << it->second->valuesNamesList << endl;

            if ( it != lastElIt )
               cout << "or" << endl;
         }

         return AOT_INCORRECT_VALUES_NUMBER; 
      }

      if (numVals == 0)
         _resArgs[argName].push_back("");
      for (int j = 0; j < numVals; ++j)
         _resArgs[argName].push_back(argv[++i]);
   }

   return AOT_OK;
}

// -----------------------------------------------------------------------

const vector<string>&  Args::getValuesVector(const string& argument) const
{ 
   if ( _resArgs.find(argument) == _resArgs.end() )
   {
      cout << "ERROR: Looking for an argument (" << argument << ") that do not exists!" << endl;
      exit(1);
   }

   return _resArgs[argument]; 
}  

// -----------------------------------------------------------------------

bool Args::checkValueRange(const string& argument, int index) const
{
   if ( _resArgs.find(argument) == _resArgs.end() )
   {
      cerr << "ERROR (getValue): Looking for an argument (" << argument << ") that does not exists!" << endl;
      exit(1);
   }

   if ( index < 0 || index >= static_cast<int>(_resArgs[argument].size()) )
   {
      cerr << "ERROR (getValue): trying to access a value index that does not exists!" << endl;
      exit(1);
   }

   return true;
}

// -----------------------------------------------------------------------

bool Args::hasArgumentDiscriminator(const string& str) const
{
   if (str.length() <= _argDiscriminator.length())
      return false;

   string::const_iterator discrIt;
   string::const_iterator strIt = str.begin();

   const string::const_iterator itBegin = _argDiscriminator.begin();
   const string::const_iterator itEnd = _argDiscriminator.end();

   // checks for negative numbers
   if ( *strIt == '-' )
   {
      const string::const_iterator itNext = strIt + 1;
      if (itNext != str.end() && isdigit(*itNext) )
         return false; // a number is not an argument name
   }

   // checks for the argument discriminator (i.e. the "-")
   for (discrIt = itBegin; discrIt != itEnd; ++discrIt, ++strIt)
   {
      if ( *discrIt != *strIt )
         return false;
   }
   
   return true;
}

// -----------------------------------------------------------------------

string Args::getArgumentString(const string& str) const
{
   if ( !hasArgumentDiscriminator(str) )
      return "";

   return str.substr( _argDiscriminator.length() );
}

// -----------------------------------------------------------------------

string Args::getWrappedString(const string& str, int leftSpace, bool spacesInFirstLine) const
{
   string result;
   insert_iterator<string> iIt(result, result.begin());
   string::const_iterator sIt;

   int col = leftSpace;

   if (spacesInFirstLine)
   {
      for (int i = 0; i < leftSpace; ++i)
         *iIt++ = ' ';
   }

   for (sIt = str.begin(); sIt != str.end(); ++sIt)
   {
      if ( (*sIt == ' ' && col > _maxColumns) || *sIt == '\n' )
      {
         col = leftSpace;
         *iIt++ = '\n';
         for (int i = 0; i < leftSpace; ++i)
            *iIt++ = ' ';
      }
      else
      {
         ++col;
         *iIt++ = *sIt;
      }
   }

   return result;
}

// -----------------------------------------------------------------------

} // end of namespace nor_utils


