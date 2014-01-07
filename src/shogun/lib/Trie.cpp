/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg, Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <lib/common.h>
#include <io/SGIO.h>
#include <lib/Trie.h>
#include <mathematics/Math.h>

namespace shogun
{
template <>
void CTrie<POIMTrie>::POIMs_extract_W_helper(
	const int32_t nodeIdx, const int32_t depth, const int32_t offset,
	const int32_t y0, float64_t* const* const W, const int32_t K )
{
	ASSERT(nodeIdx!=NO_CHILD)
	ASSERT(depth<K)
	float64_t* const W_kiy = & W[ depth ][ offset + y0 ];
	POIMTrie* const node = &TreeMem[ nodeIdx ];
	int32_t sym;

	if( depth < degree-1 )
	{
		const int32_t offset1 = offset * NUM_SYMS;
		for( sym = 0; sym < NUM_SYMS; ++sym )
		{
			ASSERT(W_kiy[sym]==0)
			const int32_t childIdx = node->children[ sym ];
			if( childIdx != NO_CHILD )
			{
				W_kiy[ sym ] = TreeMem[ childIdx ].weight;

				if (depth < K-1)
				{
					const int32_t y1 = ( y0 + sym ) * NUM_SYMS;
					POIMs_extract_W_helper( childIdx, depth+1, offset1, y1, W, K );
				}
			}
		}
	}
	else
	{
		ASSERT(depth==degree-1)
		for( sym = 0; sym < NUM_SYMS; ++sym )
		{
			ASSERT(W_kiy[sym]==0)
			W_kiy[ sym ] = node->child_weights[ sym ];
		}
	}
}

template <>
void CTrie<POIMTrie>::POIMs_extract_W(
	float64_t* const* const W, const int32_t K)
{
  ASSERT(degree>=1)
  ASSERT(K>=1)
  const int32_t N = length;
  int32_t i;
  for( i = 0; i < N; ++i ) {
    //SG_PRINT("W_helper( %d )\n", i )
    POIMs_extract_W_helper( trees[i], 0, i*NUM_SYMS, 0*NUM_SYMS, W, K );
  }
}

template <>
void CTrie<POIMTrie>::POIMs_calc_SLR_helper1(
	const float64_t* const distrib, const int32_t i, const int32_t nodeIdx,
	int32_t left_tries_idx[4], const int32_t depth, int32_t const lastSym,
	float64_t* S, float64_t* L, float64_t* R )
{
  ASSERT(depth==degree-1)
  ASSERT(nodeIdx!=NO_CHILD)

  const float64_t* const distribLeft  = & distrib[ (i-1)     * NUM_SYMS ];
  POIMTrie* const node = &TreeMem[ nodeIdx ];
  int32_t symRight;
  int32_t symLeft;

  // --- init
  node->S = 0;
  node->L = 0;
  node->R = 0;

  if (i+depth<length)
  {
	  const float64_t* const distribRight = & distrib[ (i+depth) * NUM_SYMS ];

	  // --- go thru direct children
	  for( symRight = 0; symRight < NUM_SYMS; ++symRight ) {
		  const float64_t w1 = node->child_weights[ symRight ];
		  const float64_t pRight = distribRight[ symRight ];
		  const float64_t incr1 = pRight * w1;
		  node->S += incr1;
		  node->R += incr1;
	  }
  }

  // --- collect precalced values from left neighbor tree
  for( symLeft = 0; symLeft < NUM_SYMS; ++symLeft )
  {
	  if (left_tries_idx[symLeft] != NO_CHILD)
	  {
		  POIMTrie* nodeLeft = &TreeMem[left_tries_idx[symLeft]];

		  ASSERT(nodeLeft)
		  const float64_t w2 = nodeLeft->child_weights[ lastSym ];
		  const float64_t pLeft = distribLeft[ symLeft ];
		  const float64_t incr2 = pLeft * w2;
		  node->S += incr2;
		  node->L += incr2;
	  }
  }

  // --- add w and return results
  const float64_t w0 = node->weight;
  node->S += w0;
  node->L += w0;
  node->R += w0;
  *S = node->S;
  *L = node->L;
  *R = node->R;
}


template <>
void CTrie<POIMTrie>::POIMs_calc_SLR_helper2(
	const float64_t* const distrib, const int32_t i, const int32_t nodeIdx,
	int32_t left_tries_idx[4], const int32_t depth, float64_t* S, float64_t* L,
	float64_t* R )
{
  ASSERT(0<=depth && depth<=degree-2)
  ASSERT(nodeIdx!=NO_CHILD)

  const float64_t* const distribLeft  = & distrib[ (i-1)     * NUM_SYMS ];
  POIMTrie* const node = &TreeMem[ nodeIdx ];
  float64_t dummy;
  float64_t auxS;
  float64_t auxR;
  int32_t symRight;
  int32_t symLeft;

  // --- init
  node->S = 0;
  node->L = 0;
  node->R = 0;

  // --- recurse thru direct children
  for( symRight = 0; symRight < NUM_SYMS; ++symRight )
  {
	  const int32_t childIdx = node->children[ symRight ];
	  if( childIdx != NO_CHILD )
	  {
		  if( depth < degree-2 )
		  {
			  int32_t new_left_tries_idx[4];

			  for( symLeft = 0; symLeft < NUM_SYMS; ++symLeft )
			  {
				  new_left_tries_idx[symLeft]=NO_CHILD;

				  if (left_tries_idx[symLeft] != NO_CHILD)
				  {
					  POIMTrie* nodeLeft = &TreeMem[left_tries_idx[symLeft]];
					  ASSERT(nodeLeft)
					  new_left_tries_idx[ symLeft ]=nodeLeft->children[ symRight ];
				  }
			  }
			  POIMs_calc_SLR_helper2( distrib, i, childIdx, new_left_tries_idx, depth+1, &auxS, &dummy, &auxR );
		  }
		  else
			  POIMs_calc_SLR_helper1( distrib, i, childIdx, left_tries_idx, depth+1, symRight, &auxS, &dummy, &auxR );

		  if (i+depth<length)
		  {
			  const float64_t* const distribRight = & distrib[ (i+depth) * NUM_SYMS ];
			  const float64_t pRight = distribRight[ symRight ];
			  node->S += pRight * auxS;
			  node->R += pRight * auxR;
		  }
	  }
  }

  // --- collect precalced values from left neighbor tree
  for( symLeft = 0; symLeft < NUM_SYMS; ++symLeft )
  {
	  if (left_tries_idx[symLeft] != NO_CHILD)
	  {
		  const POIMTrie* nodeLeft = &TreeMem[left_tries_idx[symLeft]];
		  ASSERT(nodeLeft)
		  const float64_t pLeft = distribLeft[ symLeft ];

		  node->S += pLeft * nodeLeft->S;
		  node->L += pLeft * nodeLeft->L;

		  if (i+depth<length)
		  {
			  const float64_t* const distribRight = & distrib[ (i+depth) * NUM_SYMS ];
			  // - second order correction for S
			  auxS = 0;
			  if( depth < degree-2 )
			  {
				  for( symRight = 0; symRight < NUM_SYMS; ++symRight )
				  {
					  const int32_t childIdxLeft = nodeLeft->children[ symRight ];
					  if( childIdxLeft != NO_CHILD )
					  {
						  const POIMTrie* const childLeft = &TreeMem[ childIdxLeft ];
						  auxS += distribRight[symRight] * childLeft->S;
					  }
				  }
			  }
			  else
			  {
				  for( symRight = 0; symRight < NUM_SYMS; ++symRight ) {
					  auxS += distribRight[symRight] * nodeLeft->child_weights[ symRight ];
				  }
			  }
			  node->S -= pLeft* auxS;
		  }
	  }
  }

  // --- add w and return results
  const float64_t w0 = node->weight;
  //SG_PRINT("  d=%d, node=%d, dS=%.3f, w=%.3f\n", depth, nodeIdx, node->S, w0 )
  node->S += w0;
  node->L += w0;
  node->R += w0;
  *S = node->S;
  *L = node->L;
  *R = node->R;
}



template <>
void CTrie<POIMTrie>::POIMs_precalc_SLR( const float64_t* const distrib )
{
	if( degree == 1 ) {
		return;
	}

	ASSERT(degree>=2)
	const int32_t N = length;
	float64_t dummy;
	int32_t symLeft;
	int32_t leftSubtrees[4];
	for( symLeft = 0; symLeft < NUM_SYMS; ++symLeft )
		leftSubtrees[ symLeft ] = NO_CHILD;

	for(int32_t i = 0; i < N; ++i )
	{
		POIMs_calc_SLR_helper2( distrib, i, trees[i], leftSubtrees, 0, &dummy, &dummy, &dummy );

		const POIMTrie* const node = &TreeMem[ trees[i] ];
		ASSERT(trees[i]!=NO_CHILD)

		for(symLeft = 0; symLeft < NUM_SYMS; ++symLeft )
			leftSubtrees[ symLeft ] = node->children[ symLeft ];
	}
}

template <>
void CTrie<POIMTrie>::POIMs_get_SLR(
	const int32_t parentIdx, const int32_t sym, const int32_t depth,
	float64_t* S, float64_t* L, float64_t* R )
{
  ASSERT(parentIdx!=NO_CHILD)
  const POIMTrie* const parent = &TreeMem[ parentIdx ];
  if( depth < degree ) {
    const int32_t nodeIdx = parent->children[ sym ];
    const POIMTrie* const node = &TreeMem[ nodeIdx ];
    *S = node->S;
    *L = node->L;
    *R = node->R;
  } else {
    ASSERT(depth==degree)
    const float64_t w = parent->child_weights[ sym ];
    *S = w;
    *L = w;
    *R = w;
  }
}

template <>
void CTrie<POIMTrie>::POIMs_add_SLR_helper2(
	float64_t* const* const poims, const int32_t K, const int32_t k,
	const int32_t i, const int32_t y, const float64_t valW,
	const float64_t valS, const float64_t valL, const float64_t valR,
	const int32_t debug)
{
	//SG_PRINT("i=%d, d=%d, y=%d:  w=%.3f \n", i, k, y, valW )
	const int32_t nk = nofsKmers[ k ];
	ASSERT(1<=k && k<=K)
	ASSERT(0<=y && y<nk)
	int32_t z;
	int32_t j;

	// --- add superstring score; subtract "w", as it was counted twice
	if( debug==0 || debug==2 )
	{
		poims[ k-1 ][ i*nk + y ] += valS - valW;
	}

	// --- left partial overlaps
	if( debug==0 || debug==3 )
	{
		int32_t r;
		for( r = 1; k+r <= K; ++r )
		{
			const int32_t nr = nofsKmers[ r ];
			const int32_t nz = nofsKmers[ k+r ];
			float64_t* const poim = & poims[ k+r-1 ][ i*nz ];
			z = y * nr;
			for( j = 0; j < nr; ++j )
			{
				if( !( 0 <= z && z < nz ) ) {
					SG_PRINT("k=%d, nk=%d,  r=%d, nr=%d,  nz=%d \n", k, nk, r, nr, nz )
					SG_PRINT("  j=%d, y=%d, z=%d \n", j, y, z )
				}
				ASSERT(0<=z && z<nz)
				poim[ z ] += valL - valW;
				++z;
			}
		}
	}
	// --- right partial overlaps
	if( debug==0 || debug==4 )
	{
		int32_t l;
		for( l = 1; k+l <= K && l <= i; ++l )
		{
			const int32_t nl = nofsKmers[ l ];
			const int32_t nz = nofsKmers[ k+l ];
			float64_t* const poim = & poims[ k+l-1 ][ (i-l)*nz ];
			z = y;
			for( j = 0; j < nl; ++j ) {
				ASSERT(0<=z && z<nz)
				poim[ z ] += valR - valW;
				z += nk;
			}
		}
	}
}

template <>
void CTrie<POIMTrie>::POIMs_add_SLR_helper1(
	const int32_t nodeIdx, const int32_t depth, const int32_t i,
	const int32_t y0, float64_t* const* const poims, const int32_t K,
	const int32_t debug)
{
	ASSERT(nodeIdx!=NO_CHILD)
	ASSERT(depth<K)
	POIMTrie* const node = &TreeMem[ nodeIdx ];
	int32_t sym;
	if( depth < degree-1 )
	{
		if( depth < K-1 )
		{
			for( sym = 0; sym < NUM_SYMS; ++sym )
			{
				const int32_t childIdx = node->children[ sym ];
				if( childIdx != NO_CHILD )
				{
					POIMTrie* const child = &TreeMem[ childIdx ];
					const int32_t y = y0 + sym;
					const int32_t y1 = y * NUM_SYMS;
					const float64_t w = child->weight;
					POIMs_add_SLR_helper2( poims, K, depth+1, i, y, w, child->S, child->L, child->R, debug );
					POIMs_add_SLR_helper1( childIdx, depth+1, i, y1, poims, K, debug );
				}
			}
		}
		else
		{
			ASSERT(depth==K-1)
			for( sym = 0; sym < NUM_SYMS; ++sym )
			{
				const int32_t childIdx = node->children[ sym ];
				if( childIdx != NO_CHILD ) {
					POIMTrie* const child = &TreeMem[ childIdx ];
					const int32_t y = y0 + sym;
					const float64_t w = child->weight;
					POIMs_add_SLR_helper2( poims, K, depth+1, i, y, w, child->S, child->L, child->R, debug );
				}
			}
		}
	}
	else
	{
		ASSERT(depth==degree-1)
		for( sym = 0; sym < NUM_SYMS; ++sym )
		{
			const float64_t w = node->child_weights[ sym ];
			const int32_t y = y0 + sym;
			POIMs_add_SLR_helper2( poims, K, depth+1, i, y, w, w, w, w, debug );
		}
	}
}



template <>
void CTrie<POIMTrie>::POIMs_add_SLR(
	float64_t* const* const poims, const int32_t K, const int32_t debug)
{
  ASSERT(degree>=1)
  ASSERT(K>=1)
  {
    const int32_t m = ( degree > K ) ? degree : K;
    nofsKmers = SG_MALLOC(int32_t,  m+1 );
    int32_t n;
    int32_t k;
    n = 1;
    for( k = 0; k < m+1; ++k ) {
      nofsKmers[ k ] = n;
      n *= NUM_SYMS;
    }
  }
  const int32_t N = length;
  int32_t i;
  for( i = 0; i < N; ++i ) {
    POIMs_add_SLR_helper1( trees[i], 0, i, 0*NUM_SYMS, poims, K, debug );
  }
  SG_FREE(nofsKmers);
}
}
