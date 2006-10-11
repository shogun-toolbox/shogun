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

#include "InductionSort.h"

InductionSortObject::InductionSortObject(unsigned int inductionPosition, unsigned int inductionValue, 
										 unsigned int suffixIndex)
{
	// sort value is 64 bits long.
	// bits are ...
	// 63 - 60: induction position (0 - 15)
	// 59 - 29: induction value at induction position (0 - (2^30 -1))
	// 28 - 0:  suffix index for the suffix sorted by induction (0 - (2^30) - 1)
	m_sortValue[0] = inductionPosition << 28;
	m_sortValue[0] |= ((inductionValue & 0x3fffffff) >> 2);
	m_sortValue[1] = (inductionValue << 30);
	m_sortValue[1] |= suffixIndex;
}
