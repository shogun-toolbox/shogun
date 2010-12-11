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
* \file Defaults.h
* Defines macros and default values.
* This file just holds some macros and static strings that define
* some default values.
* \date 11/11/2005
*/

#ifndef __DEFAULTS_H
#define __DEFAULTS_H

static const char SHYP_NAME[] = "shyp"; //!< The default strong hypothesis file name
static const char SHYP_EXTENSION[] = "xml"; //!< The default strong hypothesis file name extension

static const char COMMENT[] = "Research code"; //!< Comment to put in the executable 

static const char defaultLearner[] = "SingleStumpLearner"; //!< The default weak learner

static const char CURRENT_VERSION[] = "0.85dev";

/**
* Defines the sorting method.
* Columns needs to be sorted when using the decision stumps algorithm;
* Generally conservativeness is not an issue with this type of algorithm
* but because I want to get \b exactly the same results on win32 an unix, 
* if I set non conservative sort, the outcome is different on the two.
* \remarks The valid values are:
* - 1 = slow but stable
* - 0 = fast but non stable
* \date 16/11/2005
*/
#define STABLE_SORT 0

/**
* Debug level. If there is a problem try activating this.
* \date 16/11/2005
*/ 
#define MB_DEBUG 0

#define S_(X) #X //!< Used with STRINGIZE can stringize a macro name
#define STRINGIZE(X) S_(X) //!< Stringize a macro name! :)

#endif // __DEFAULTS_H
