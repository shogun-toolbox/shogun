#include "lib/common.h"
#include "lib/io.h"
#include "kernel/Trie.h"
#include "lib/Mathmatics.h"

CTrie::CTrie(INT d): degree(d)
{
#ifdef USE_TREEMEM
	TreeMemPtrMax=1024*1024/sizeof(struct Trie) ;
	TreeMemPtr=0 ;
	TreeMem = (struct Trie*)malloc(TreeMemPtrMax*sizeof(struct Trie)) ;
#endif
	
	length = 0;
	trees=NULL;
} ;


CTrie::~CTrie()
{
	destroy() ;
	
#ifdef USE_TREEMEM
	free(TreeMem) ;
#endif
}

void CTrie::destroy()
{
	if (trees!=NULL)
	{
		delete_tree() ;
		for (INT i=0; i<length; i++)
		{
			delete trees[i];
			trees[i]=NULL;
		}
		delete[] trees;
		trees=NULL;
	}
}

void CTrie::create(INT len)
{
#ifdef OSF1
	trees=new (struct Trie**)[len] ;		
#else
	trees=new struct Trie*[len] ;		
#endif
	for (INT i=0; i<len; i++)
	{
		trees[i]=new struct Trie ;
		trees[i]->weight=0 ;
		for (INT j=0; j<4; j++)
		    trees[i]->children[j] = NO_CHILD ;
	} 
	length = len ;
}


void CTrie::delete_tree(struct Trie * p_tree)
{
	if (p_tree==NULL)
	{
		if (trees==NULL)
			return;

		for (INT i=0; i<length; i++)
		{
#ifndef USE_TREEMEM
			delete_tree(trees[i]);
#endif

			for (INT k=0; k<4; k++)
				trees[i]->children[k]=NO_CHILD;
		}

#ifdef USE_TREEMEM
		TreeMemPtr=0;
#endif
		return;
	}

	for (INT i=0; i<4; i++)
	{
		if (p_tree->children[i]!=NO_CHILD)
		{
#ifndef USE_TREEMEM
			delete_tree(p_tree->children[i]);
			delete p_tree->children[i];
#endif
			p_tree->children[i]=NO_CHILD;
		} 
		p_tree->weight=0;
	}
} 


void CTrie::add_to_trie(int i, INT * vec, float alpha, DREAL *weights)
{
	struct Trie *tree = trees[i] ;
	
	INT max_depth = 0 ;
	for (INT j=0; (j<degree) && (i+j<length); j++)
		if (CMath::abs(weights[j]*alpha)>1e-8)
			max_depth = j+1 ;
	
	for (INT j=0; (j<max_depth) && (i+j<length); j++)
	{
		if ((j<degree-1) && (tree->children[vec[i+j]]!=NO_CHILD))
		{
#ifdef USE_TREEMEM
			tree=&TreeMem[tree->children[vec[i+j]]] ;
#else
			tree=tree->children[vec[i+j]] ;
#endif
			tree->weight += alpha*weights[j];
		}
		else if (j==degree-1)
		{
			tree->child_weights[vec[i+j]] += alpha*weights[j];
			break ;
		}
		else
		{
#ifdef USE_TREEMEM
			tree->children[vec[i+j]]=TreeMemPtr++;
			INT tmp=tree->children[vec[i+j]] ;
			check_treemem() ;
			tree=&TreeMem[tmp] ;
#else
			tree->children[vec[i+j]] = new struct Trie ;
			ASSERT(tree->children[vec[i+j]]!=NULL) ;
			tree=tree->children[vec[i+j]] ;
#endif
			if (j==degree-2)
			{
				for (INT k=0; k<4; k++)
					tree->child_weights[k]=0;
			}
			else
			{
				for (INT k=0; k<4; k++)
					tree->children[k]=NO_CHILD;
			}
			tree->weight = alpha*weights[j] ;
		}
	}
}


DREAL CTrie::compute_abs_weights_tree(struct Trie* p_tree) 
{
	DREAL ret=0 ;
	
	if (p_tree==NULL)
		return 0 ;
	
	for (INT k=0; k<4; k++)
		ret+=(p_tree->child_weights[k]) ;
	
	return ret ;
	
	ret+=(p_tree->weight) ;
	
#ifdef USE_TREEMEM
	for (INT i=0; i<4; i++)
		if (p_tree->children[i]!=NO_CHILD)
			ret += compute_abs_weights_tree(&TreeMem[p_tree->children[i]])  ;
#else
	for (INT i=0; i<4; i++)
		if (p_tree->children[i]!=NO_CHILD)
			ret += compute_abs_weights_tree(p_tree->children[i])  ;
#endif
	
	return ret ;
}


DREAL *CTrie::compute_abs_weights(int &len) 
{
	DREAL * sum=new DREAL[length*4] ;
	for (INT i=0; i<length*4; i++)
		sum[i]=0 ;
	len=length ;
	
	for (INT i=0; i<length; i++)
	{
		struct Trie *tree = trees[i] ;
		ASSERT(tree!=NULL) ;
		for (INT k=0; k<4; k++)
		{
#ifdef USE_TREEMEM
			sum[i*4+k]=compute_abs_weights_tree(&TreeMem[tree->children[k]]) ;
#else
			sum[i*4+k]=compute_abs_weights_tree(tree->children[k]) ;
#endif
		}
	}

	return sum ;
}

void CTrie::compute_scoring_helper(struct Trie* tree, INT i, INT j, DREAL weight, INT d, INT max_degree, INT num_feat, INT num_sym, INT sym_offset, INT offs, DREAL* result)
{
	if (tree==NULL)
		tree=trees[i] ;
	
	ASSERT(tree!=NULL) ;
	
	if (i+j<num_feat)
	{
		if (j<degree-1)
		{
			for (INT k=0; k<num_sym; k++)
			{
				if (tree->children[k]!=NO_CHILD)
				{
#ifdef USE_TREEMEM
					struct Trie* child=&TreeMem[tree->children[k]];
#else
					struct Trie* child=tree->children[k];
#endif
					//continue recursion if not yet at max_degree, else add to result
					if (d<max_degree-1)
						compute_scoring_helper(child, i, j+1, weight+child->weight, d+1, max_degree, num_feat, num_sym, sym_offset, num_sym*offs+k, result);
					else
						result[sym_offset*(i+j-max_degree+1)+num_sym*offs+k] += weight;

					//do recursion starting from this position
					if (d==0)
						compute_scoring_helper(child, i, j+1, 0.0, 0, max_degree, num_feat, num_sym, sym_offset, offs, result);
				}
			}
		}
		else if (j==degree-1)
		{
			for (INT k=0; k<num_sym; k++)
			{
				//continue recursion if not yet at max_degree, else add to result
				if (d<max_degree-1 && i<num_feat-1)
					compute_scoring_helper(trees[i+1], i+1, 0, weight+tree->child_weights[k], d+1, max_degree, num_feat, num_sym, sym_offset, num_sym*offs+k, result);
				else
					result[sym_offset*(i+j-max_degree+1)+num_sym*offs+k] += weight+tree->child_weights[k];
			}
		}
	}
}

void CTrie::add_example_to_tree_mismatch_recursion(struct Trie *tree,  INT i, DREAL alpha,
												   INT *vec, INT len_rem, 
												   INT degree_rec, INT mismatch_rec, 
												   INT max_mismatch, DREAL * weights) 
{
	if (tree==NULL)
		tree=trees[i] ;
	ASSERT(tree!=NULL) ;
	
	if ((len_rem<=0) || (mismatch_rec>max_mismatch) || (degree_rec>degree))
		return ;
	ASSERT(tree!=NULL) ;
	const INT other[4][3] = {	{1,2,3},{0,2,3},{0,1,3},{0,1,2} } ;
	
	struct Trie *subtree = NULL ;
	
	if (degree_rec==degree-1)
	{
		tree->child_weights[vec[0]] += alpha*weights[degree_rec+degree*mismatch_rec];
		if (mismatch_rec+1<=max_mismatch)
			for (INT o=0; o<3; o++)
				tree->child_weights[other[vec[0]][o]] += alpha*weights[degree_rec+degree*(mismatch_rec+1)];
		return ;
	}
	else
	{
		if (tree->children[vec[0]]!=NO_CHILD)
		{
#ifdef USE_TREEMEM
		  subtree=&TreeMem[tree->children[vec[0]]] ;
#else
		  subtree=tree->children[vec[0]] ;
#endif
		  subtree->weight += alpha*weights[degree_rec+degree*mismatch_rec];
		}
		else 
		{
#ifdef USE_TREEMEM
			tree->children[vec[0]]=TreeMemPtr++ ;
			INT tmp=tree->children[vec[0]] ;
			check_treemem() ;
			subtree=&TreeMem[tmp] ;
#else
			tree->children[vec[0]]=new struct Trie ;
			ASSERT(tree->children[vec[0]]!=NULL) ;
			subtree=tree->children[vec[0]] ;
#endif			
			if (degree_rec==degree-2)
			{
				for (INT k=0; k<4; k++)
					tree->child_weights[k]=0;
			}
			else
			{
				for (INT k=0; k<4; k++)
					tree->children[k]=NO_CHILD;
			}
			subtree->weight = alpha*weights[degree_rec+degree*mismatch_rec] ;
		}
		add_example_to_tree_mismatch_recursion(subtree,  i, alpha,
											   &vec[1], len_rem-1, 
											   degree_rec+1, mismatch_rec, max_mismatch, weights) ;
		
		if (mismatch_rec+1<=max_mismatch)
		{
			for (INT o=0; o<3; o++)
			{
				INT ot = other[vec[0]][o] ;
				if (tree->children[ot]!=NO_CHILD)
				{
#ifdef USE_TREEMEM
					subtree=&TreeMem[tree->children[ot]] ;
#else
					subtree=tree->children[ot] ;
#endif
					subtree->weight += alpha*weights[degree_rec+degree*(mismatch_rec+1)];
				}
				else 
				{
#ifdef USE_TREEMEM
					tree->children[ot]=TreeMemPtr++ ;
					INT tmp=tree->children[ot] ;
					check_treemem() ;
					subtree=&TreeMem[tmp] ;
#else
					tree->children[ot]=new struct Trie ;
					ASSERT(tree->children[ot]!=NULL) ;
					subtree=tree->children[ot] ;
#endif
					if (degree_rec==degree-2)
					{
						for (INT k=0; k<4; k++)
							tree->child_weights[k]=0;
					}
					else
					{
						for (INT k=0; k<4; k++)
							tree->children[k]=NO_CHILD;
					}
					subtree->weight = alpha*weights[degree_rec+degree*(mismatch_rec+1)] ;
				}
				
				add_example_to_tree_mismatch_recursion(subtree,  i, alpha,
													   &vec[1], len_rem-1, 
													   degree_rec+1, mismatch_rec+1, max_mismatch, weights) ;
			}
		}
	}
}
