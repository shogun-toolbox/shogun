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
* \file Args.h Handles the command line arguments.
*/
#pragma warning( disable : 4786 )

#ifndef __ARGS_H
#define __ARGS_H

#include <map>
#include <vector>
#include <set>
#include <string>
#include <iostream>
#include <sstream>
#include <stdlib.h> //for exit function when we are using gcc

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace nor_utils {

/**
* The enum returned by readArguments(). It gives informations
* about the status of the read arguments.
* \date 10/11/2005
*/
enum ArgsOutType
{
   AOT_OK, //!< Everything went ok.
   AOT_NO_ARGUMENTS, //!< No argument in the command line.
   AOT_UNKOWN_ARGUMENT, //!< Unknown argument (not registered argument).
   AOT_INCORRECT_VALUES_NUMBER //!< The number of values for the given argument does not match the declaration.
};

/**
* Handle the command line arguments.
* Provide an easy (but simple) way to handle parameters in a command line program.
* It works by first declaring the arguments via declareArgument, for instance:
* \code
* nor_utils::Args myArgs;
* // define the character(s) that define what an option is. By default there is no
* // argument discriminator, but without it no multiple arguments with the same name
* // are allowed.
* myArgs.setArgumentDiscriminator("-")
* // declare a simple argument
* myArgs.declareArgument("help"); 
* // declare an argument -test which takes two following values.
* myArgs.declareArgument("test", "Test the model.", 2, "<dataFile> <shypFile>");
* // declare an argument with the same same but 3 values.
* myArgs.declareArgument("test", "Test the model, and save the results.", 3, "<dataFile> <shypFile> <outFile>");
* \endcode
* 
* Once the arguments are declared, readArguments() takes the standard C
* arguments (that is \a argv and \a argc) which were passed by the user:
* \verbatim myProgram -test file.dat shyp.xml \endverbatim
* and fill a map with the ones selected by the user.
* If they do not follow the declaration's constraints, readArguments 
* returns an error.
* When the user needs the values of an argument, it just calls the
* getValue() or getValuesVector().
*
* It is also possible to defines groups of arguments, using setGroup
* before the call of any declareArgument. The name passed, will be used
* to print a nicely formatted list of the group's arguments using 
* printGroup().
* \date 10/11/2005
*/
class Args
{
public:

   /**
   * The constructor. Set the current group to "general".
   * \date 10/11/2005
   */
   Args() : _currentGroup("general"), _maxColumns(60), _argDiscriminator("") {}

   /**
   * The destructor. Deallocates the memory of each Argument object.
   * \date 14/2/2006
   */
   ~Args();

   /**
   * Defines the current groups of arguments. The group
   * defines which argument has to be printed together.
   * To be used before the subsequent call of declareArgument.
   * \param groupName The name of the group.
   * \see printGroup
   * \see declareArgument(const string&)
   * \see declareArgument(const string&, const string&, int, const string&)
   * \date 28/11/2005
   */
   void setGroup( const string& groupName ) { _currentGroup = groupName; }

   /**
   * Defines an argument discriminator. It is helpful to discriminate
   * an argument from a value of the argument. For instance if in the commandline
   * it the user writes:
   * \verbatim myProgram -arg 1 3 hello -anotherarg 10 -smallarg \endverbatim
   * the first argument (arg) has 3 values, the second 1 and the third none.
   * They are discriminated from the values thanks to the string "-" which can be
   * defined here.
   * \param argDiscriminator The string which is used to discriminate arguments from
   * values.
   * \date 14/2/2006
   */
   void setArgumentDiscriminator( const string& argDiscriminator ) 
                                            { _argDiscriminator = argDiscriminator; }

   void setMaxColumns(const int maxCols) { _maxColumns = maxCols; }

   /**
   * Declare a simple argument.
   * Just declare an argument without following values, nor descriptions.
   * \param name The name of the argument.
   * \date 28/11/2005
   */
   void declareArgument( const string& name );

   /**
   * Declare an argument.
   * This type of argument has a description, and might have values. For instance
   * an argument such as -file filename.txt is declared as
   * \code
   * declareArgument( "file", "Open a file", 1, "<filename>" );
   * \endcode
   * \param name The name of the argument.
   * \param description The description of the argument.
   * \param numValues The number of values (numeric or not) that follow the argument.
   * \param valuesNamesList The names list of the values that will be printed
   * with the argument, when help is requested.
   * \param active If true the argument will be showed when printGroup is
   * called. Useful when an argument need to be activated/deactivated.
   * \see printGroup
   * \see setGroup
   * \date 28/11/2005
   */
   void declareArgument( const string& name, const string& description, int numValues = 0,
                         const string& valuesNamesList = "");

   /**
   * Erases a declaration from the list of declarations.
   * \param name The name of the declaration to be erased.
   */
   void eraseDeclaration( const string& name);

   /**
   * Erases a declaration with a specific number of values from the list of declarations.
   * \param name The name of the declaration to be erased.
   * \param numValues The number of values (numeric or not) that follow the argument in
   * the previous declaration.
   */
   void eraseDeclaration( const string& name, const int numValues);

   /**
   * Display the arguments (with description and values) associated with
   * the groupName passed.
   * Example:
   * \code
   * Args a;
   * a.setGroup("group1");
   * a.declareArgument("arg1", "The first argument");
   * a.declareArgument("arg2", "The second argument");
   * a.setGroup("group2");
   * a.declareArgument("arg3", "Another argument");
   * \endcode
   * Then, printGroup("group1"), will print:
   \verbatim
group1:
   -arg1:
      The first argument
   -arg2:
      The second argument\endverbatim
   * \param groupName The name of the group.
   * \param out The output stream. By default is \a cout.
   * \param indSpaces The number of spaces for the indentation.
   * \date 28/11/2005
   */
   void printGroup( const string& groupName, ostream& out = cout, int indSpaces = 3 ) const;

   /**
   * Read the arguments passed with the commandline.
   * \param argc The number of arguments (standard C in main()).
   * \param argv The arguments (standard C in main()).
   * \return An enum that report on the result of the reading.
   * \see ArgsOutType
   * \date 16/11/2005
   */
   ArgsOutType	 readArguments( int argc, const char* argv[] );

   /**
   * Ask if an argument has been provided via command line.
   * \param argument The argument to be checked.
   * \return True if the argument has been written by the user, false otherwise.
   * \remark It does not check if the argument has been declared, but if it
   * has been called by the user at the command line.
   * \date 16/11/2005
   */
   bool		    hasArgument(const string& argument) const
                            { return _resArgs.find(argument) != _resArgs.end(); }

   /**
   * Returns the vector of the values that belongs to the given argument.
   * For instance with the command line:
   * \verbatim myProgram -arg val1 val2 10 11 \endverbatim
   * we have:
   * \code
   * getValuesVector("arg"); // -> ["val1", "val2", "10", "11"]
   * \endcode
   * \param argument The name of the argument.
   * \return A vector with the values belonging to the argument.
   * \date 16/11/2005
   */
   const vector<string>&  getValuesVector(const string& argument) const;

   /**
   * Get the number of values of an argument that has been written in the command line.
   * \param argument The name of the argument.
   * \remark This does \b not return the number of values for a declared argument!
   * \date 14/2/2006
   */
   int getNumValues(const string& argument) const 
      { return static_cast<int>(_resArgs[argument].size()); }

   /**
   * Get the nth value of the given argument, where n = index.
   * The value is stored in the variable valueToFill which can be any type,
   * thanks to the use of templates and string stream.
   * For instance with the command line:
   * \verbatim myProgram -arg val1 val2 10 11 \endverbatim
   * we have:
   * \code
   * int val;
   * getValue("arg", 2, val); // -> val = 10
   * \endcode
   * \param argument The name of the argument.
   * \param index The index of the value belonging to the argument.
   * \param valueToFill The value read.
   * \remark The index start counting from zero.
   * \date 16/11/2005
   */
   template <typename T>
   void getValue(const string& argument, int index, T& valueToFill) const
   { 
      checkValueRange(argument, index);
      stringstream ss(_resArgs[argument][index]);    
      ss >> valueToFill;
   }

   /**
   * Specialization of getValue to simply return the string directly.
   * This is very important in case the string contains white spaces
   * on the edges. With the stringstream they would be erased.
   * \param argument The name of the argument.
   * \param index The index of the value belonging to the argument.
   * \param valueToFill The value read.
   * \remark The index start counting from zero.
   * \date 16/2/2006
   */
   void getValue(const string& argument, int index, string& valueToFill) const
   {
      checkValueRange(argument, index);
      valueToFill = _resArgs[argument][index];
   }

   /**
   * Get the nth value of the given argument, where n = index.
   * The value is stored in the variable valueToFill which can be any type,
   * thanks to the use of templates and string stream.
   * For instance with the command line:
   * \verbatim myProgram -arg val1 val2 10 11 \endverbatim
   * we have:
   * \code
   * int val = getValue<int>("arg", 2); // -> val = 10
   * \endcode
   * \param argument The name of the argument.
   * \param index The index of the value belonging to the argument.
   * \return The requested value.
   * \remark The index start counting from zero.
   * \remark The type of the returned value must be specified with the template.
   * \date 16/11/2005
   */
   template <typename T>
   T getValue(const string& argument, int index = 0) const
   {
      T val;
      getValue(argument, index, val);
      return val;
   }

private:

   /**
   * Checks if the given index exists for the given argument.
   * It terminates the program with an error in the case this is not true.
   * \param argument The name of the argument.
   * \param index The index of the value belonging to the argument.
   * \return True if the index in in the range.
   * \date 16/2/2006
   */
   bool checkValueRange(const string& argument, int index) const;

   /**
   * Check if a given string begins with the argument discriminator.
   * \param str The string to be checked.
   * \return true if the string begins with the discriminator, false otherwise.
   * \see setArgumentDiscriminator
   * \date 14/2/2006
   */
   bool hasArgumentDiscriminator(const string& str) const;

   /**
   * Return a string of the argument without the discriminator.
   * \param str The string to be cropped.
   * \return A string with the argument only.
   * \see setArgumentDiscriminator
   * \see hasArgumentDiscriminator
   * \date 14/2/2006
   */
   string getArgumentString(const string& str) const;

   /**
   * Return a string which has been wrapped to fit into _maxColumns columns.
   * For instance, if _maxColumns is set to 20:
   * \code
   * string s = "A simple test of wrapping a line into some more. Here is how it goes";
   * string w1 = getWrappedString(s, 5);
   * string w2 = getWrappedString(s, 5, false);
   * cout << "w1: " << endl << w1 << endl;
   * cout << "w2: " << endl << w1 << endl;
   * \endcode
   * will print:
   * \verbatim
w1:
     A simple test of wrapping 
     a line into some more.
     Here is how it goes.
w2:
A simple test of wrapping 
     a line into some more.
     Here is how it goes.\endverbatim
   * \param str The string to be wrapped.
   * \param leftSpace The number of spaces before the line gets printed.
   * \param spacesInFirstLine If true, the first line will have spaces as all the
   * other lines, otherwise it will begin without spaces.
   * \remark The _maxColumns limit is not strict. It is checked only at the end of each
   * word.
   * \see printGroup
   * \date 28/11/2005
   */
   string getWrappedString(const string& str, int leftSpace = 0, 
                           bool spacesInFirstLine = true) const;

   /**
   * Holds the informations of a single argument.
   * \date 28/11/2005
   */
   struct Argument
   {
      /**
      * An empty constructor. Needed for the operations with vectors.
      * \date 28/11/2005
      */
      Argument() {}

      /**
      * The actual constructor.
      * \param name The name of the argument.
      * \param description The description of the argument.
      * \param numValues The number of values that follow the argument.
      * \param valuesNamesList The names list of the values that will be printed.
      * with the argument, when help is requested.
      * \see declareArgument(const string&, const string&, int, const string&)
      * \date 28/11/2005
      */
      Argument(const string& name, const string& description = "", 
               int numValues = 0, const string& valuesNamesList = "")
         : name(name), description(description), 
           numValues(numValues), valuesNamesList(valuesNamesList) {}

      string name; //!< The name of the argument. 
      string description; //!< The description of the argument.
      int numValues; //!< The number of values that follow the argument.
      string valuesNamesList; //!< The names list of the values that will be printed with the argument, when help is requested. 
   };

   /**
   * The current group. 
   * \see setGroup
   * \see declareArgument
   */
   string  _currentGroup;

   /**
   * The maximum number of column (in a loosely sense) allowed to be printed.
   * \see getWrappedString
   * \see printGroup
   */
   int     _maxColumns;

   /**
   * The argument discriminator.
   * \see setArgumentDiscriminator
   * \see hasArgumentDiscriminator
   * \date 14/2/2006
   */
   string  _argDiscriminator;

   mutable map< string, vector<Argument*> >   _groupedList; //!< The list of arguments per group (string=name of the group)
   
   /**
   * Arguments declared. The Argument pointer is the same of the one in _groupedList, but in a
   * multimap it is easier to look for the right argument when parsing the command line.
   */
   multimap<string, Argument*>  _declArgs;
   typedef multimap<string, Argument*>::iterator mm_iterator; //!< Iterators over ranges of _declArgs. 

   mutable map< string, vector<string> > _resArgs; //!< Arguments found.

};

}

#endif // __ARGS_H

