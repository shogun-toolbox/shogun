#ifdef USE_SVMMPI
/* -*-C++-*- */

/*
 * bcache.h : defintion of block cache for matrix class
 * Copyright (C) 2000-2001 The Australian National University
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License aLONG with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#ifndef __BCACHE_H_
#define __BCACHE_H_

#include <iostream>

template <class T>
struct __bc_blist {
  T *block;
  struct __bc_blist<T> *next;
};

template <class T>
struct __bc_size {
  unsigned size;
  struct __bc_blist<T> *blocklist;
  struct __bc_size<T> *prev;
  struct __bc_size<T> *next;
};

template <class T>
class CBlockCache {
public:
  CBlockCache(void);
  ~CBlockCache(void);

  void AddCacheSize(const unsigned numelems);

  T *GetBlock(const unsigned numelems);
  void ReturnBlock(T *block, const unsigned numelems);
  T *ResizeBlock(T *block, const unsigned oldnumelems,
		 const unsigned newnumelems);

  unsigned LONG GetNumElementsOutstanding(void) const { return (m_ElementsOutstanding); }
  unsigned LONG GetNumBytesOutstanding(void) const { return (m_ElementsOutstanding * sizeof(T)); }
  unsigned GetKBOutstanding(void) const { return (GetNumBytesOutstanding() / 1024); }
  unsigned LONG GetMaxBytesOutstanding(void) const { return (m_MaxElementsOutstanding * sizeof(T)); }
  unsigned GetMaxKBOutstanding(void) const { return (GetMaxBytesOutstanding() / 1024); }

  void SummarizeCached(ostream &os) const;

  void Flush(void);
private:
  inline unsigned CHAR Hash(const unsigned numelems) const {
    return ((numelems & 0xff) ^ ((numelems >> 8) & 0xff) ^
	    ((numelems >> 16) & 0xff) ^ ((numelems >> 24) && 0xff));
  }

  void IncrOutstanding(const unsigned numelems) {
    ++m_NumOutstanding;
    m_ElementsOutstanding += numelems;
    if (m_ElementsOutstanding > m_MaxElementsOutstanding)
      m_MaxElementsOutstanding = m_ElementsOutstanding;
  }
  void DecrOutstanding(const unsigned numelems) {
    --m_NumOutstanding;
    m_ElementsOutstanding -= numelems;
  }

  // Basic memory allocate primitives -- just blocks, not counting
  T *AllocateNewBlock(const unsigned numelems);
  void DeallocateBlock(T *ptr, const unsigned numelems);
  T *ResizeExistingBlock(T *ptr, const unsigned oldnumelems,
			 const unsigned newnumelems);

  struct __bc_size<T> *m_Sizes[256];
  struct __bc_blist<T> *m_UnusedBlocks;

  /* Various counters */
  unsigned LONG m_TotalElementRequests;
  unsigned m_NumRequests;

  /* This is number of blocks that the app is currently using */
  unsigned LONG m_ElementsOutstanding;
  unsigned LONG m_MaxElementsOutstanding;
  unsigned m_NumOutstanding;

  /* These are blocks that app has returned and are being cached */
  unsigned LONG m_ElementsCached;
  unsigned m_NumCached;
};

#endif /* ! __BCACHE_H_ */
#endif
