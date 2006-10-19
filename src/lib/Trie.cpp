#include "lib/common.h"
#include "lib/io.h"
#include "lib/Trie.h"
#include "lib/Mathematics.h"

CTrie::CTrie(INT d): degree(d), position_weights(NULL)
{
	TreeMemPtrMax=1024*1024/sizeof(struct Trie) ;
	TreeMemPtr=0 ;
	TreeMem = (struct Trie*)malloc(TreeMemPtrMax*sizeof(struct Trie)) ;
	
	length = 0;
	trees=NULL;
	tree_initialized=false ;
} ;


CTrie::~CTrie()
{
	destroy() ;
	
	free(TreeMem) ;
}

void CTrie::destroy()
{
	if (trees!=NULL)
	{
		delete_trees() ;
		for (INT i=0; i<length; i++)
			trees[i] = NO_CHILD;
		TreeMemPtr=0 ;
		delete[] trees;
		trees=NULL;
	}
}

void CTrie::create(INT len)
{
	trees=new INT[len] ;		
	TreeMemPtr=0 ;
	for (INT i=0; i<len; i++)
		trees[i]=get_node() ;
	length = len ;
}


void CTrie::delete_trees()
{
	if (trees==NULL)
		return;
	
	for (INT i=0; i<length; i++)
	{
		for (INT k=0; k<4; k++)
			TreeMem[trees[i]].children[k]=NO_CHILD;
		TreeMem[trees[i]].has_seq=false ;
	}
	TreeMemPtr=length;
} 


void CTrie::add_to_trie(int i, INT * vec, float alpha, DREAL *weights)
{
	INT tree = trees[i] ;
	
	INT max_depth = 0 ;
	for (INT j=0; (j<degree) && (i+j<length); j++)
		if (CMath::abs(weights[j]*alpha)>1e-8)
			max_depth = j+1 ;
	
	for (INT j=0; (j<max_depth) && (i+j<length); j++)
    {
		ASSERT((vec[i+j]>=0) && (vec[i+j]<4)) ;
		if ((j<degree-1) && (TreeMem[tree].children[vec[i+j]]!=NO_CHILD))
		{
			if (TreeMem[tree].children[vec[i+j]]<0)
			{
				// special treatment of the next nodes
				ASSERT(j >= degree-16) ;
				// get the right element
				INT node = - TreeMem[tree].children[vec[i+j]] ;
				ASSERT(TreeMem[node].has_seq) ;
				
                // check whether the same string is stored
				int mismatch_pos = -1 ;
				for (int k=0; (j+k<max_depth) && (i+j+k<length); k++)
				{
					ASSERT((vec[i+j+k]>=0) && (vec[i+j+k]<4)) ;
					ASSERT(TreeMem[node].seq[k]<4) ;
					ASSERT(k<16) ;
					if (TreeMem[node].seq[k]!=vec[i+j+k])
					{
						mismatch_pos=k ;
						break ;
					}
				}
				
				if (mismatch_pos==-1)
					// if so, then just increase the weight by alpha and stop
					TreeMem[node].weight+=alpha ;
				else
					// otherwise
					// 1. replace current node with new node
					// 2. create new nodes until mismatching positon
					// 2. add a branch with old string (old node) and the new string (new node)
				{
					fprintf(stderr, "splitting at depth=%i\n", j+mismatch_pos) ;
					//ASSERT(j+mismatch_pos<degree-2) ;
					
					// replace old node
					INT tmp = get_node() ;
					TreeMem[tree].children[vec[i+j]]=tmp ;
					INT last_node = tmp ;
					TreeMem[last_node].weight=(TreeMem[node].weight+alpha)*weights[j] ;
					ASSERT(vec[i+j]==TreeMem[node].seq[0]) ;
					
					// create new nodes until mismatch
					for (int k=1; k<mismatch_pos; k++)
					{
						ASSERT((vec[i+j+k]>=0) && (vec[i+j+k]<4)) ;
						ASSERT(vec[i+j+k]==TreeMem[node].seq[k]) ;
						
						INT tmp=get_node() ;
						TreeMem[last_node].children[vec[i+j+k]]=tmp ;
						TreeMem[last_node].weight = (TreeMem[node].weight+alpha)*weights[j+k] ;
						last_node=tmp ;
					}
					ASSERT(vec[i+j+mismatch_pos]!=TreeMem[node].seq[mismatch_pos]) ;
					
					// the branch for the existing string
					TreeMem[last_node].children[TreeMem[node].seq[mismatch_pos]] = -node ;
					for (INT q=0; (j+q+mismatch_pos<max_depth) && (i+j+q+mismatch_pos<length); q++)
					{
						ASSERT(q<16);
						TreeMem[node].seq[q] = TreeMem[node].seq[q+mismatch_pos] ;
					}
					TreeMem[node].has_seq=true ;
					
					// the new branch
					ASSERT((vec[i+j+mismatch_pos]>=0) && (vec[i+j+mismatch_pos]<4)) ;
					{
						INT tmp = get_node() ;
						TreeMem[last_node].children[vec[i+j+mismatch_pos]] = -tmp ;
						last_node=tmp ;
					}
					TreeMem[last_node].weight = alpha ;
					TreeMem[last_node].has_seq = true ;
					for (INT q=0; (j+q+mismatch_pos<max_depth) && (i+j+q+mismatch_pos<length); q++)
						TreeMem[last_node].seq[q] = vec[i+j+mismatch_pos+q] ;
				}
				break ;
			} 
			else
			{
				//fprintf(stderr, "%i/%i\n", TreeMem[tree].children[vec[i+j]], TreeMemPtrMax) ;
				tree=TreeMem[tree].children[vec[i+j]] ;
				ASSERT((tree>=0) && (tree<TreeMemPtrMax)) ;
				ASSERT(!TreeMem[tree].has_seq) ;
				TreeMem[tree].weight += alpha*weights[j];
			}
		}
		else if (j==degree-1)
		{
			// special treatment of the last node
			ASSERT(!TreeMem[tree].has_seq) ;
			TreeMem[tree].child_weights[vec[i+j]] += alpha*weights[j];
			break ;
		}
		else
		{
			bool use_seq = (j>=degree-13) && (j<degree-2) ;//(j>degree-16) ;
			ASSERT(!TreeMem[tree].has_seq) ;

			INT tmp = get_node() ;
			if (use_seq)
				TreeMem[tree].children[vec[i+j]] = -tmp ;
			else
				TreeMem[tree].children[vec[i+j]] = tmp ;
			tree=tmp ;
			
			ASSERT((tree>=0) && (tree<TreeMemPtrMax)) ;
			TreeMem[tree].has_seq=use_seq ;

			if (use_seq)
			{
				TreeMem[tree].weight = alpha ;
				for (INT q=0; (j+q<max_depth) && (i+j+q<length); q++)
				{
					ASSERT(q<16) ;
					TreeMem[tree].seq[q]=vec[i+j+q] ;
				}
				break ;
			}
			else
			{
				TreeMem[tree].weight = alpha*weights[j] ;
				if (j==degree-2)
				{
					for (INT k=0; k<4; k++)
						TreeMem[tree].child_weights[k]=0;
				}
				else
				{
					for (INT k=0; k<4; k++)
						TreeMem[tree].children[k]=NO_CHILD;
				}
			}
		}
    }
}


DREAL CTrie::compute_abs_weights_tree(INT tree) 
{
	DREAL ret=0 ;
	
	if (tree==NO_CHILD)
		return 0 ;
	ASSERT(tree>=0) ;
	
	for (INT k=0; k<4; k++)
		ret+=(TreeMem[tree].child_weights[k]) ;
	
	return ret ;
	
	ret+=(TreeMem[tree].weight) ;
	
	for (INT i=0; i<4; i++)
		if (TreeMem[tree].children[i]!=NO_CHILD)
			ret += compute_abs_weights_tree(TreeMem[tree].children[i])  ;
	
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
		ASSERT(trees[i]!=NO_CHILD) ;
		for (INT k=0; k<4; k++)
		{
			sum[i*4+k]=compute_abs_weights_tree(TreeMem[trees[i]].children[k]) ;
		}
	}

	return sum ;
}

void CTrie::compute_scoring_helper(INT tree, INT i, INT j, DREAL weight, INT d, INT max_degree, INT num_feat, INT num_sym, INT sym_offset, INT offs, DREAL* result)
{
  if (tree==NO_CHILD)
    tree=trees[i] ;
  
  ASSERT(tree!=NO_CHILD) ;
  
  if (i+j<num_feat)
  {
      if (j<degree-1)
	  {
		  for (INT k=0; k<num_sym; k++)
		  {
			  if (TreeMem[tree].children[k]!=NO_CHILD)
			  {
				  INT child=TreeMem[tree].children[k];
				  //continue recursion if not yet at max_degree, else add to result
				  if (d<max_degree-1)
					  compute_scoring_helper(child, i, j+1, weight+TreeMem[child].weight, d+1, max_degree, num_feat, num_sym, sym_offset, num_sym*offs+k, result);
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
				  compute_scoring_helper(trees[i+1], i+1, 0, weight+TreeMem[tree].child_weights[k], d+1, max_degree, num_feat, num_sym, sym_offset, num_sym*offs+k, result);
			  else
				  result[sym_offset*(i+j-max_degree+1)+num_sym*offs+k] += weight+TreeMem[tree].child_weights[k];
		  }
	  }
  }
}

void CTrie::add_example_to_tree_mismatch_recursion(INT tree,  INT i, DREAL alpha,
						   INT *vec, INT len_rem, 
						   INT degree_rec, INT mismatch_rec, 
						   INT max_mismatch, DREAL * weights) 
{
	if (tree==NO_CHILD)
		tree=trees[i] ;
	ASSERT(tree!=NO_CHILD) ;
	
	if ((len_rem<=0) || (mismatch_rec>max_mismatch) || (degree_rec>degree))
		return ;
	const INT other[4][3] = {	{1,2,3},{0,2,3},{0,1,3},{0,1,2} } ;
	
	INT subtree = NO_CHILD ;
	
	if (degree_rec==degree-1)
    {
		TreeMem[tree].child_weights[vec[0]] += alpha*weights[degree_rec+degree*mismatch_rec];
		if (mismatch_rec+1<=max_mismatch)
			for (INT o=0; o<3; o++)
				TreeMem[tree].child_weights[other[vec[0]][o]] += alpha*weights[degree_rec+degree*(mismatch_rec+1)];
		return ;
    }
	else
    {
		if (TreeMem[tree].children[vec[0]]!=NO_CHILD)
		{
			subtree=TreeMem[tree].children[vec[0]] ;
			TreeMem[subtree].weight += alpha*weights[degree_rec+degree*mismatch_rec];
		}
		else 
		{
			TreeMem[tree].children[vec[0]]=TreeMemPtr++ ;
			INT tmp=TreeMem[tree].children[vec[0]] ;
			check_treemem() ;
			subtree=tmp ;
			if (degree_rec==degree-2)
			{
				for (INT k=0; k<4; k++)
					TreeMem[tree].child_weights[k]=0;
			}
			else
			{
				for (INT k=0; k<4; k++)
					TreeMem[tree].children[k]=NO_CHILD;
			}
			TreeMem[subtree].weight = alpha*weights[degree_rec+degree*mismatch_rec] ;
		}
		add_example_to_tree_mismatch_recursion(subtree,  i, alpha,
											   &vec[1], len_rem-1, 
											   degree_rec+1, mismatch_rec, max_mismatch, weights) ;
		
		if (mismatch_rec+1<=max_mismatch)
		{
			for (INT o=0; o<3; o++)
			{
				INT ot = other[vec[0]][o] ;
				if (TreeMem[tree].children[ot]!=NO_CHILD)
				{
					subtree=TreeMem[tree].children[ot] ;
					TreeMem[subtree].weight += alpha*weights[degree_rec+degree*(mismatch_rec+1)];
				}
				else 
				{
					TreeMem[tree].children[ot]=TreeMemPtr++ ;
					INT tmp=TreeMem[tree].children[ot] ;
					check_treemem() ;
					subtree=tmp ;
					if (degree_rec==degree-2)
					{
						for (INT k=0; k<4; k++)
							TreeMem[tree].child_weights[k]=0;
					}
					else
					{
						for (INT k=0; k<4; k++)
							TreeMem[tree].children[k]=NO_CHILD;
					}
					TreeMem[subtree].weight = alpha*weights[degree_rec+degree*(mismatch_rec+1)] ;
				}
				
				add_example_to_tree_mismatch_recursion(subtree,  i, alpha,
													   &vec[1], len_rem-1, 
													   degree_rec+1, mismatch_rec+1, max_mismatch, weights) ;
			}
		}
    }
}
