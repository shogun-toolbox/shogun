/* ***** BEGIN LICENSE BLOCK *****
 * Version: MPL 1.1
 *
 * The contents of this file are subject to the Mozilla Public License Version
 * 1.1 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 * http://www.mozilla.org/MPL/
 *
 * Software distributed under the License is distributed on an "AS IS" basis,
 * WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
 * for the specific language governing rights and limitations under the
 * License.
 *
 * The Original Code is the MSufSort suffix sorting algorithm (Version 2.2).
 *
 * The Initial Developer of the Original Code is
 * Michael A. Maniscalco
 * Portions created by the Initial Developer are Copyright (C) 2006
 * the Initial Developer. All Rights Reserved.
 *
 * Contributor(s):
 *
 *   Michael A. Maniscalco
 *
 * ***** END LICENSE BLOCK ***** */

#ifndef BWT_H
#define BWT_H

//=============================================================================================
// BWT demo using the MSufSort algorithm.
//
// Author: M.A. Maniscalco
// Date: 7/30/04
// email: michael@www.michael-maniscalco.com
//
// This code is free for non commercial use only.
//
//=============================================================================================



#include "MSufSort.h"




class BWT
{
public:
	BWT();

	virtual ~BWT();

	unsigned int Forward(SYMBOL_TYPE * data, unsigned int length);

	void Reverse(SYMBOL_TYPE * data, unsigned int length, unsigned int index);

	unsigned int MSufSortTime(){return m_suffixSorter->GetElapsedSortTime();}

	bool VerifySort(){return m_suffixSorter->VerifySort();}
private:

	MSufSort *			m_suffixSorter;
};

#endif
