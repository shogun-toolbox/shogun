#include "lib/common.h"
#include "lib/io.h"
#include "lib/Trie.h"
#include "lib/Mathematics.h"

template <>
void CTrie<POIMTrie>::POIMs_extract_W_helper( const INT nodeIdx, const int depth, const INT offset, const INT y0,
				    DREAL* const* const W, const INT K )
{
	ASSERT(nodeIdx!=NO_CHILD);
	ASSERT(depth<K);
	DREAL* const W_kiy = & W[ depth ][ offset + y0 ];
	POIMTrie* const node = &TreeMem[ nodeIdx ];
	INT sym;

	if( depth < degree-1 )
	{
		const INT offset1 = offset * NUM_SYMS;
		for( sym = 0; sym < NUM_SYMS; ++sym )
		{
			ASSERT(W_kiy[sym]==0);
			const INT childIdx = node->children[ sym ];
			if( childIdx != NO_CHILD )
			{
				W_kiy[ sym ] = TreeMem[ childIdx ].weight;

				if (depth < K-1)
				{
					const INT y1 = ( y0 + sym ) * NUM_SYMS;
					POIMs_extract_W_helper( childIdx, depth+1, offset1, y1, W, K );
				}
			}
		}
	}
	else
	{
		ASSERT(depth==degree-1);
		for( sym = 0; sym < NUM_SYMS; ++sym )
		{
			ASSERT(W_kiy[sym]==0);
			W_kiy[ sym ] = node->child_weights[ sym ];
		}
	}
}

template <>
void CTrie<POIMTrie>::POIMs_extract_W( DREAL* const* const W, const INT K )
{
  ASSERT(degree>=1);
  ASSERT(K>=1);
  const INT N = length;
  INT i;
  for( i = 0; i < N; ++i ) {
    //printf( "W_helper( %d )\n", i );
    POIMs_extract_W_helper( trees[i], 0, i*NUM_SYMS, 0*NUM_SYMS, W, K );
  }
}

template <>
void CTrie<POIMTrie>::POIMs_calc_SLR_helper1( const DREAL* const distrib, const INT i,
				    const INT nodeIdx, INT left_tries_idx[4], const int depth, INT const lastSym,
				    DREAL* S, DREAL* L, DREAL* R )
{
  ASSERT(depth==degree-1);
  ASSERT(nodeIdx!=NO_CHILD);

  const DREAL* const distribLeft  = & distrib[ (i-1)     * NUM_SYMS ];
  POIMTrie* const node = &TreeMem[ nodeIdx ];
  INT symRight;
  INT symLeft;

  // --- init
  node->S = 0;
  node->L = 0;
  node->R = 0;

  if (i+depth<length)
  {
	  const DREAL* const distribRight = & distrib[ (i+depth) * NUM_SYMS ];

	  // --- go thru direct children
	  for( symRight = 0; symRight < NUM_SYMS; ++symRight ) {
		  const DREAL w1 = node->child_weights[ symRight ];
		  const DREAL pRight = distribRight[ symRight ];
		  const DREAL incr1 = pRight * w1;
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

		  ASSERT(nodeLeft);
		  const DREAL w2 = nodeLeft->child_weights[ lastSym ];
		  const DREAL pLeft = distribLeft[ symLeft ];
		  const DREAL incr2 = pLeft * w2;
		  node->S += incr2;
		  node->L += incr2;
	  }
  }

  // --- add w and return results
  const DREAL w0 = node->weight;
  node->S += w0;
  node->L += w0;
  node->R += w0;
  *S = node->S;
  *L = node->L;
  *R = node->R;
}


template <>
void CTrie<POIMTrie>::POIMs_calc_SLR_helper2( const DREAL* const distrib, const INT i,
				    const INT nodeIdx, INT left_tries_idx[4], const int depth,
				    DREAL* S, DREAL* L, DREAL* R )
{
  ASSERT(0<=depth && depth<=degree-2);
  ASSERT(nodeIdx!=NO_CHILD);

  const DREAL* const distribLeft  = & distrib[ (i-1)     * NUM_SYMS ];
  POIMTrie* const node = &TreeMem[ nodeIdx ];
  DREAL dummy;
  DREAL auxS;
  DREAL auxR;
  INT symRight;
  INT symLeft;

  // --- init
  node->S = 0;
  node->L = 0;
  node->R = 0;

  // --- recurse thru direct children
  for( symRight = 0; symRight < NUM_SYMS; ++symRight )
  {
	  const INT childIdx = node->children[ symRight ];
	  if( childIdx != NO_CHILD )
	  {
		  if( depth < degree-2 ) 
		  {
			  INT new_left_tries_idx[4];

			  for( symLeft = 0; symLeft < NUM_SYMS; ++symLeft )
			  {
				  new_left_tries_idx[symLeft]=NO_CHILD;

				  if (left_tries_idx[symLeft] != NO_CHILD)
				  {
					  POIMTrie* nodeLeft = &TreeMem[left_tries_idx[symLeft]];
					  ASSERT(nodeLeft);
					  new_left_tries_idx[ symLeft ]=nodeLeft->children[ symRight ];
				  }
			  }
			  POIMs_calc_SLR_helper2( distrib, i, childIdx, new_left_tries_idx, depth+1, &auxS, &dummy, &auxR );
		  }
		  else 
			  POIMs_calc_SLR_helper1( distrib, i, childIdx, left_tries_idx, depth+1, symRight, &auxS, &dummy, &auxR );

		  if (i+depth<length)
		  {
			  const DREAL* const distribRight = & distrib[ (i+depth) * NUM_SYMS ];
			  const DREAL pRight = distribRight[ symRight ];
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
		  ASSERT(nodeLeft);
		  const DREAL pLeft = distribLeft[ symLeft ];

		  node->S += pLeft * nodeLeft->S;
		  node->L += pLeft * nodeLeft->L;

		  if (i+depth<length)
		  {
			  const DREAL* const distribRight = & distrib[ (i+depth) * NUM_SYMS ];
			  // - second order correction for S
			  auxS = 0;
			  if( depth < degree-2 )
			  {
				  for( symRight = 0; symRight < NUM_SYMS; ++symRight )
				  {
					  const INT childIdxLeft = nodeLeft->children[ symRight ];
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
  const DREAL w0 = node->weight;
  //printf( "  d=%d, node=%d, dS=%.3f, w=%.3f\n", depth, nodeIdx, node->S, w0 );
  node->S += w0;
  node->L += w0;
  node->R += w0;
  *S = node->S;
  *L = node->L;
  *R = node->R;
}



template <>
void CTrie<POIMTrie>::POIMs_precalc_SLR( const DREAL* const distrib )
{
	if( degree == 1 ) {
		return;
	}

	ASSERT(degree>=2);
	const INT N = length;
	DREAL dummy;
	INT symLeft;
	INT leftSubtrees[4];
	for( symLeft = 0; symLeft < NUM_SYMS; ++symLeft )
		leftSubtrees[ symLeft ] = NO_CHILD;

	for(INT i = 0; i < N; ++i )
	{
		POIMs_calc_SLR_helper2( distrib, i, trees[i], leftSubtrees, 0, &dummy, &dummy, &dummy );

		const POIMTrie* const node = &TreeMem[ trees[i] ];
		ASSERT(trees[i]!=NO_CHILD);

		for(symLeft = 0; symLeft < NUM_SYMS; ++symLeft )
			leftSubtrees[ symLeft ] = node->children[ symLeft ];
	}
}

template <>
void CTrie<POIMTrie>::POIMs_get_SLR( const INT parentIdx, const INT sym, const int depth,
			   DREAL* S, DREAL* L, DREAL* R )
{
  ASSERT(parentIdx!=NO_CHILD);
  const POIMTrie* const parent = &TreeMem[ parentIdx ];
  if( depth < degree ) {
    const INT nodeIdx = parent->children[ sym ];
    const POIMTrie* const node = &TreeMem[ nodeIdx ];
    *S = node->S;
    *L = node->L;
    *R = node->R;
  } else {
    ASSERT(depth==degree);
    const DREAL w = parent->child_weights[ sym ];
    *S = w;
    *L = w;
    *R = w;
  }
}

template <>
void CTrie<POIMTrie>::POIMs_add_SLR_helper2( DREAL* const* const poims, const int K, const int k, const INT i, const INT y,
				   const DREAL valW, const DREAL valS, const DREAL valL, const DREAL valR, const INT debug )
{
	//printf( "i=%d, d=%d, y=%d:  w=%.3f \n", i, k, y, valW );
	const INT nk = nofsKmers[ k ];
	ASSERT(1<=k && k<=K);
	ASSERT(0<=y && y<nk);
	INT z;
	INT j;

	// --- add superstring score; subtract "w", as it was counted twice
	if( debug==0 || debug==2 )
	{
		poims[ k-1 ][ i*nk + y ] += valS - valW;
	}

	// --- left partial overlaps
	if( debug==0 || debug==3 )
	{
		INT r;
		for( r = 1; k+r <= K; ++r )
		{
			const INT nr = nofsKmers[ r ];
			const INT nz = nofsKmers[ k+r ];
			DREAL* const poim = & poims[ k+r-1 ][ i*nz ];
			z = y * nr;
			for( j = 0; j < nr; ++j )
			{
				if( !( 0 <= z && z < nz ) ) {
					printf( "k=%d, nk=%d,  r=%d, nr=%d,  nz=%d \n", k, nk, r, nr, nz );
					printf( "  j=%d, y=%d, z=%d \n", j, y, z );
				}
				ASSERT(0<=z && z<nz);
				poim[ z ] += valL - valW;
				++z;
			}
		}
	}
	// --- right partial overlaps
	if( debug==0 || debug==4 )
	{
		INT l;
		for( l = 1; k+l <= K && l <= i; ++l )
		{
			const INT nl = nofsKmers[ l ];
			const INT nz = nofsKmers[ k+l ];
			DREAL* const poim = & poims[ k+l-1 ][ (i-l)*nz ];
			z = y;
			for( j = 0; j < nl; ++j ) {
				ASSERT(0<=z && z<nz);
				poim[ z ] += valR - valW;
				z += nk;
			}
		}
	}
}

template <>
void CTrie<POIMTrie>::POIMs_add_SLR_helper1( const INT nodeIdx, const int depth, const INT i, const INT y0,
				   DREAL* const* const poims, const INT K, const INT debug )
{
	ASSERT(nodeIdx!=NO_CHILD);
	ASSERT(depth<K);
	POIMTrie* const node = &TreeMem[ nodeIdx ];
	INT sym;
	if( depth < degree-1 )
	{
		if( depth < K-1 )
		{
			for( sym = 0; sym < NUM_SYMS; ++sym )
			{
				const INT childIdx = node->children[ sym ];
				if( childIdx != NO_CHILD )
				{
					POIMTrie* const child = &TreeMem[ childIdx ];
					const INT y = y0 + sym;
					const INT y1 = y * NUM_SYMS;
					const DREAL w = child->weight;
					POIMs_add_SLR_helper2( poims, K, depth+1, i, y, w, child->S, child->L, child->R, debug );
					POIMs_add_SLR_helper1( childIdx, depth+1, i, y1, poims, K, debug );
				}
			}
		}
		else
		{
			ASSERT(depth==K-1);
			for( sym = 0; sym < NUM_SYMS; ++sym )
			{
				const INT childIdx = node->children[ sym ];
				if( childIdx != NO_CHILD ) {
					POIMTrie* const child = &TreeMem[ childIdx ];
					const INT y = y0 + sym;
					const DREAL w = child->weight;
					POIMs_add_SLR_helper2( poims, K, depth+1, i, y, w, child->S, child->L, child->R, debug );
				}
			}
		}
	}
	else
	{
		ASSERT(depth==degree-1);
		for( sym = 0; sym < NUM_SYMS; ++sym )
		{
			const DREAL w = node->child_weights[ sym ];
			const INT y = y0 + sym;
			POIMs_add_SLR_helper2( poims, K, depth+1, i, y, w, w, w, w, debug );
		}
	}
}



template <>
void CTrie<POIMTrie>::POIMs_add_SLR( DREAL* const* const poims, const INT K, const INT debug )
{
  ASSERT(degree>=1);
  ASSERT(K>=1);
  {
    const int m = ( degree > K ) ? degree : K;
    nofsKmers = new INT[ m+1 ];
    INT n;
    INT k;
    n = 1;
    for( k = 0; k < m+1; ++k ) {
      nofsKmers[ k ] = n;
      n *= NUM_SYMS;
    }
  }
  const INT N = length;
  INT i;
  for( i = 0; i < N; ++i ) {
    POIMs_add_SLR_helper1( trees[i], 0, i, 0*NUM_SYMS, poims, K, debug );
  }
  delete[] nofsKmers;
}
