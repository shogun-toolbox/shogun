/*   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *   Copyright (C) 2009 - 2012 Jun Liu and Jieping Ye 
 */

#ifndef  ORDERTREE_SLEP
#define  ORDERTREE_SLEP

#define IGNORE_IN_CLASSLIST

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>


/*
 * In this file, we propose an O(n^2) algorithm for solving the problem:
 *
 * min   1/2 \|x - u\|^2
 * s.t.  x_i \ge x_j \ge 0, (i,j) \in I,
 *
 * where I is the edge set of the tree
 *
 *
 */

/*
 * Last updated on January, 21, 2011
 *
 * 1) the function merge is a non-recursive process for merging one tree with the other
 *
 * 2) we follow the writeup to revise the function computeMaximalMean
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS
IGNORE_IN_CLASSLIST struct NodeNum
{
	int node_num;
	struct NodeNum *next;
};

IGNORE_IN_CLASSLIST struct ChildrenNum
{
	int children_num;
	int *children;
};

IGNORE_IN_CLASSLIST struct Node
{
	int flag; /*if the maximal root-tree of the subtree rooted at this node has been computed, flag=1, otherwise 0*/
	double m; /*During the computation, it stores the maximal mean from this node to (grandson) child node
			   *The number of nodes on this path is stored in num
			   *
			   *It is intialized with the value of u(node_num)
			   */
	int num;  /*the number of nodes, whose avarage gives the maximal mean---x*/
	struct Node *brother; /*the pointer to the brother node(s)*/
	struct Node *child; /*the pointer to the child node(s)*/
	struct NodeNum *firstNode; /*the first node in the "maximal mean" group*/
	struct NodeNum *lastNode; /*the last node in the "maximal mean" group*/
};
#endif

/*
 * We build a tree with the input from a file
 *
 * The file has n rows represented in the following format
 *
 |  parent   | number of children | children
 18               3             10  13  17
 10               3             5   8   9
 13               2             11  12
 17               3             13  14  15
 5                2             1  4
 8                2             6  7
 9                0
 11               0
 12               0
 14               0
 15               0
 16               0
 1                0
 4                2              2  3
 6                0
 7                0
 2                0
 3                0
 *
 *
 * Each row provides the information of one parent node and its children
 *
 * If a parent node is not included in any row, it is regarded that it is the leaf node.
 * For example, it is valid that the rows with zero children can be deleted.
 *
 * Node number is numbered from 1 to n, where n denotes the number of nodes.
 *
 * In the program, we deduct the number by 1, as C starts from 0, instead of 1.
 *
 */

void readFromFile(char * FileName, struct ChildrenNum ** TreeInfo, int n){
	FILE *fp;
	struct ChildrenNum * treeInfo;
	int i, j, num, nodeId;


	fp=fopen(FileName, "r");

	if(!fp){
		printf("\n\n Fatal Error!!!");
		printf("\n\n Failure in reading the file:%s!", FileName);
		printf("\n\n The program does not check the correctness of the tree provided in the file: %s!", FileName);
		return;
	}

	treeInfo=(struct ChildrenNum *)malloc(sizeof(struct ChildrenNum)*n);

	if(!treeInfo){
		printf("\n Allocation of treeInfo failure!");
		return;
	}

	for(i=0;i<n;i++){
		treeInfo[i].children_num=0;
		treeInfo[i].children=NULL;
	}


	while (!feof(fp)) {

		i=-1;num=-1;
		if ( fscanf(fp, "%d %d", &i, &num)!=2){

			/*if this is due to extra spaces/breaks etc., we terminate reading the file */
			if(feof(fp))
				break;

			printf("\n For each row, it should has at least two numbers: nodeNum and number of children");
			return;
		}

		if (i>n || i<1){
			printf("\n The node number should be between [1, %d]!",n);
			return;
		}

		i=i-1;
		/*i=i-1, as C starts from 0 instead of 1*/
		if (num>0){            
			treeInfo[i].children_num=num;            

			treeInfo[i].children=(int *)malloc(sizeof(int)*num);

			if(!treeInfo[i].children){
				printf("\n Allocation of treeInfo failure!");
				return;
			}

			for(j=0;j<num;j++){
				if(!fscanf(fp, "%d", &nodeId) ){
					printf("\n This row should have %d children nodes!", num);
					return;
				}
				else{
					if (nodeId>n || nodeId<1){
						printf("\n The node number should be between [1, %d]!", n);
						return;
					}

					treeInfo[i].children[j]=nodeId-1;
					/*add -1, as C starts from 0 instead of 1*/
				}

			}
		}
	}

	fclose(fp);

	/*
	   printf("\n In readFromFile!");
	   for(i=0;i<n;i++){
	   printf("\n %d: %d:",i, treeInfo[i].children_num);

	   for(j=0;j<treeInfo[i].children_num;j++)
	   printf(" %d", treeInfo[i].children[j]);
	   }
	   printf("\n Out of readFromFile!");
	   */


	*TreeInfo=treeInfo;/*set value for TreeInfo*/
}


/*
 *
 * We build the tree in a recursive manner
 *
 */
void buildTree(struct Node* root, struct ChildrenNum * treeInfo, double *u){


	struct Node * newNode;
	struct NodeNum * currentNode;
	int currentRoot=root->firstNode->node_num;
	int numberOfChildren=treeInfo[currentRoot].children_num;
	int i;

	/* insert the children nodes of the current root
	*/
	for(i=0;i<numberOfChildren;i++){


		newNode=(struct Node *)malloc(sizeof(struct Node));
		currentNode=(struct NodeNum *)malloc(sizeof(struct NodeNum));

		if(!newNode){
			printf("\n Allocation in buildTree failure!");
			return;
		}

		if(!currentNode){
			printf("\n Allocation in buildTree failure!");
			return;
		}


		newNode->flag=0;
		newNode->m=u[treeInfo[currentRoot].children[i]];
		newNode->num=1;
		newNode->child=NULL;

		currentNode->node_num=treeInfo[currentRoot].children[i];
		currentNode->next=NULL;
		newNode->firstNode=newNode->lastNode=currentNode;

		/*
		 * insert newnode to be the children nodes of root
		 *
		 */
		newNode->brother=root->child;
		root->child=newNode;

		/*
		 * treat newNode as the root, and add its children
		 *
		 */

		buildTree(newNode, treeInfo, u);
	}
}

/*
 * initilize the root, which means that the tree is built by this function.
 * as the root is the starting point of a tree
 * 
 * we use the input file for building the tree
 */

void initializeRoot(struct Node ** Root, char * FileName, double *u, int rootNum, int n){

	struct NodeNum * currentNode;
	struct Node *root;
	struct ChildrenNum * treeInfo;
	int i;

	/*read the from the file to construct treeInfo*/
	readFromFile(FileName, &treeInfo, n);

	if(rootNum>n || rootNum <1){
		printf("\n The node number of the root should be between [1, %d]!", n);
		return;
	}

	rootNum=rootNum-1;
	/*add -1, as C starts from 0 instead of 1*/

	root=(struct Node *)malloc(sizeof(struct Node));
	currentNode=(struct NodeNum *)malloc(sizeof(struct NodeNum));

	if(!root){
		printf("\n Allocation in computeGroups failure!");
		return;
	}

	if(!currentNode){
		printf("\n Allocation in computeGroups failure!");
		return;
	}


	root->flag=0;
	root->m=u[rootNum];
	root->num=1;
	root->brother=root->child=NULL;

	currentNode->node_num=rootNum;
	currentNode->next=NULL;
	root->firstNode=root->lastNode=currentNode;

	/*build the tree using buildTree*/
	buildTree(root, treeInfo, u);

	/*free treeInfo*/
	for(i=0;i<n;i++){
		if (treeInfo[i].children_num)
			free(treeInfo[i].children);
	}
	free(treeInfo);

	*Root=root;
}



/*
 * initilize the root for the full binary tree
 *
 * We do not need to give the input file, as binary tree is very special
 */

void initializeRootBinary(struct Node ** Root, double *u, int n){

	struct NodeNum * currentNode;
	struct Node *root;
	struct ChildrenNum * treeInfo;
	int rootNum=1;
	int i, half=n/2;

	/*
	 *
	 * readFromFile(FileName, &treeInfo, n);
	 *
	 * Replace the above function.
	 *
	 * we build treeInfo here directly using the special structure
	 *
	 */

	treeInfo=(struct ChildrenNum *)malloc(sizeof(struct ChildrenNum)*n);    
	if(!treeInfo){
		printf("\n Allocation of treeInfo failure!");
		return;
	}

	for(i=0;i<half;i++){
		treeInfo[i].children_num=2;
		treeInfo[i].children=(int *)malloc(sizeof(int)*2);
		treeInfo[i].children[0]=2*i+1;
		treeInfo[i].children[1]=2*i+2;
	}

	for(i=half;i<n;i++){
		treeInfo[i].children_num=0;
		treeInfo[i].children=NULL;
	}


	rootNum=rootNum-1;
	/*add -1, as C starts from 0 instead of 1*/

	root=(struct Node *)malloc(sizeof(struct Node));
	currentNode=(struct NodeNum *)malloc(sizeof(struct NodeNum));

	if(!root){
		printf("\n Allocation in computeGroups failure!");
		return;
	}

	if(!currentNode){
		printf("\n Allocation in computeGroups failure!");
		return;
	}


	root->flag=0;
	root->m=u[rootNum];
	root->num=1;
	root->brother=root->child=NULL;

	currentNode->node_num=rootNum;
	currentNode->next=NULL;
	root->firstNode=root->lastNode=currentNode;

	/*build the tree using buildTree*/
	buildTree(root, treeInfo, u);

	/*free treeInfo*/
	for(i=0;i<half;i++){
		free(treeInfo[i].children);
	}
	free(treeInfo);

	*Root=root;
}


/*
 * initilize the root for the full binary tree
 *
 * We do not need to give the input file, as tree of depth 1 is very special
 */

void initializeRootDepth1(struct Node ** Root, double *u, int n){

	struct NodeNum * currentNode;
	struct Node *root;
	struct ChildrenNum * treeInfo;
	int rootNum=1;
	int i;

	/*
	 * readFromFile(FileName, &treeInfo, n);
	 *
	 * we build treeInfo here, using the special structure of the tree with depth 1
	 *
	 */

	treeInfo=(struct ChildrenNum *)malloc(sizeof(struct ChildrenNum)*n);    
	if(!treeInfo){
		printf("\n Allocation of treeInfo failure!");
		return;
	}

	for(i=0;i<n;i++){
		treeInfo[i].children_num=0;
		treeInfo[i].children=NULL;
	}

	/*process the root*/
	if (n>1){
		treeInfo[0].children_num=n-1;
		treeInfo[0].children=(int *)malloc(sizeof(int)*(n-1));
		for(i=1;i<n;i++)
			treeInfo[0].children[i-1]=i;
	}

	rootNum=rootNum-1;
	/*add -1, as C starts from 0 instead of 1*/

	root=(struct Node *)malloc(sizeof(struct Node));
	currentNode=(struct NodeNum *)malloc(sizeof(struct NodeNum));

	if(!root){
		printf("\n Allocation in computeGroups failure!");
		return;
	}

	if(!currentNode){
		printf("\n Allocation in computeGroups failure!");
		return;
	}


	root->flag=0;
	root->m=u[rootNum];
	root->num=1;
	root->brother=root->child=NULL;

	currentNode->node_num=rootNum;
	currentNode->next=NULL;
	root->firstNode=root->lastNode=currentNode;

	/*build the tree using buildTree*/
	buildTree(root, treeInfo, u);

	/*free treeInfo*/
	if(n>1)
		free(treeInfo[0].children);
	free(treeInfo);

	*Root=root;
}



/*
 * merge root with maxNode
 */
void merge(struct Node * root, struct Node * maxNode ){
	struct Node * childrenNode, *maxNodeChild;

	root->m= (root->m* root->num + maxNode->m *maxNode->num)/(root->num + maxNode->num);
	root->num+=maxNode->num;
	root->lastNode->next=maxNode->firstNode;
	root->lastNode=maxNode->lastNode;

	/*
	 * update the brother list of maxNode (when removing maxNode)
	 *
	 */
	if (root->child==maxNode){
		root->child=maxNode->brother;
	}
	else{
		childrenNode=root->child;

		while(childrenNode->brother!=maxNode){
			childrenNode=childrenNode->brother;
		}
		/*childrenNode's brother is maxNode*/
		childrenNode->brother=maxNode->brother;
	}


	/*
	 * change the children of maxNode to the children of root
	 */
	maxNodeChild=maxNode->child;
	if (maxNodeChild){
		/*if maxNode has at least a child*/

		while(maxNodeChild->brother)
			maxNodeChild=maxNodeChild->brother;
		/*maxNodeChild points to the last child of maxNode*/

		maxNodeChild->brother=root->child;
		root->child=maxNode->child;
	}

	/*
	 * remove maxNode from the children list of root
	 */
	free(maxNode);

}



/*
 * compute the maximal mean for each node
 *
 */

void computeMaximalMean(struct Node * root){
	struct Node * childrenNode, *maxNode;
	double mean;

	/*if root already points to a leaf node, we do nothing*/
	if(!root->child){

		/*the value of a maximal root-tree is non-negative*/
		if (root->m <0)
			root->m =0;

		root->flag=1;
		return;
	}

	/*the following loop corresponds to line 5-20 of the algorithm*/
	while(1){

		childrenNode=root->child;
		if(!childrenNode){

			if (root->m <0)
				root->m =0;

			root->flag=1;
			return;
		}

		/*we note that, childrenNode->m >=0*/

		mean=0;

		/*visit all its children nodes, to get the maximal "mean" and corresponding maxNode*/
		while(childrenNode){

			/*if the maximal root-tree at childrenNode is not computed, we compute it*/
			if (!childrenNode->flag)
				computeMaximalMean(childrenNode);

			if (childrenNode->m >= mean){
				mean=childrenNode->m;
				maxNode=childrenNode;
			}

			childrenNode=childrenNode->brother;            
		}

		if ( (root->m <= 0) && (mean==0) ){
			/* merge root with all its children, in this case, 
			 * its children is a super-node 
			 * (thus does not has any other children, due to merge)*/

			childrenNode=root->child;
			while(childrenNode){
				merge(root, childrenNode);
				childrenNode=root->child;
			}

			root->m =0;            
			root->flag=1;
			return;
		}

		if (root->m > mean){

			root->flag=1;
			return;
		}

		merge(root,maxNode);
	}

}



/*
 * compute the maximal mean for each node, without the non-negative constraint
 * 
 * Composed on November 23, 2011.
 *
 */

void computeMaximalMean_without_nonnegative(struct Node * root){
	struct Node * childrenNode, *maxNode;
	double mean;
	int mean_flag;

	/*if root already points to a leaf node, we do nothing*/
	if(!root->child){

		/*the value of a maximal root-tree is not necessarily non-negative, when the non-negative constraint is not imposed*/

		/*
		   The following is removed
		   if (root->m <0)
		   root->m =0;
		   */


		root->flag=1;
		return;
	}

	/*the following loop corresponds to line 5-20 of the algorithm */
	while(1){

		childrenNode=root->child;
		if(!childrenNode){

			/*the value of a maximal root-tree is not necessarily non-negative, when the non-negative constraint is not imposed*/

			/*
			   The following is removed

			   if (root->m <0)
			   root->m =0;
			   */

			root->flag=1;
			return;
		}

		/*we note that, childrenNode->m >=0 does not necesarily hold.
		  Therefore, for mean, we need to initialize with a small value. We initialize it with the value of its first child node
		  */

		mean_flag=0; /*0 denotes that "mean" has not been really specified*/

		/*visit all its children nodes, to get the maximal "mean" and corresponding maxNode*/
		while(childrenNode){

			/*if the maximal root-tree at childrenNode is not computed, we compute it*/
			if (!childrenNode->flag)
				computeMaximalMean_without_nonnegative(childrenNode);

			/*if mean has not been specified, let us specify it, and set mean_flag to 1*/
			if (!mean_flag){
				mean=childrenNode->m;
				mean_flag=1;
			}

			if (childrenNode->m >= mean){
				mean=childrenNode->m;
				maxNode=childrenNode;
			}

			childrenNode=childrenNode->brother;            
		}

		if (root->m > mean){

			root->flag=1;
			return;
		}

		merge(root,maxNode);
	}

}


/*
 * computeSolution
 *
 */


void computeSolution(double *x, struct Node *root){
	struct Node * child;
	struct NodeNum *currentNode;
	double mean;

	if (root){        
		/*
		 * process the root
		 * 
		 * set the value for x
		 */

		mean=root->m;

		currentNode=root->firstNode;
		while(currentNode){
			x[currentNode->node_num]=mean;
			currentNode=currentNode->next;
		}            

		/*process the children of root*/
		child=root->child;
		while(child){
			computeSolution(x, child);

			child=child->brother;
		}
	}
}

/*
 * traverse the tree
 * used for debugging the correctness of the code
 */

void traversalTree(struct Node *root){
	struct Node * child;
	struct NodeNum *currentNode;

	if (root){
		printf("\n\n root->m =%2.5f, num:%d \n Nodes:",root->m,root->num);

		currentNode=root->firstNode;
		while(currentNode){
			printf(" %d ", currentNode->node_num);
			currentNode=currentNode->next;
		}     

		printf("\n root: %d, child:", root->m);

		/*print out the children of root*/
		child=root->child;
		while(child){
			printf(" %d", child->m);
			child=child->brother;
		}

		/*print out the children of children*/
		child=root->child;
		while(child){
			traversalTree(child);

			child=child->brother;
		}
	}
}





/*
 * free the dynamic space generated by alloc
 */

void deleteTree(struct Node *root){
	struct Node *child, *temp;
	struct NodeNum *currentNode;

	if (root){

		child=root->child;

		while(child){

			temp=child->brother;
			/*point to its brother*/

			deleteTree(child);
			/*free its chlidren*/

			child=temp;
		}

		/*
		 * free root
		 *
		 * 1. free NodeNum pointed by firstNode and lastNode
		 * 2. free Node
		 */
		currentNode=root->firstNode;
		while(currentNode){
			root->firstNode=currentNode->next;
			free(currentNode);

			currentNode=root->firstNode;
		}
		root->lastNode=NULL;
		free(root);
	}
}

/*
 * This is the main function for the general tree
 *
 */

void orderTree(double *x, char * FileName, double *u, int rootNum, int n){
	struct Node * root;

	/*
	 * build the tree using initializeRoot
	 */
	initializeRoot(&root, FileName, u, rootNum, n);  

	/*
	   printf("\n\n Before computation");
	   traversalTree(root);
	   */


	/*
	 * compute the maximal average for each node
	 */

	computeMaximalMean(root);


	/*compute the solution from the tree*/

	computeSolution(x, root);


	/*
	   printf("\n\n After computation");
	   traversalTree(root);
	   */


	/* delete the tree
	*/
	deleteTree(root);
}


/*
 * This is the main function for the general tree, without the non-negative constraint
 *
 */

void orderTree_without_nonnegative(double *x, char * FileName, double *u, int rootNum, int n){
	struct Node * root;

	/*
	 * build the tree using initializeRoot
	 */
	initializeRoot(&root, FileName, u, rootNum, n);  

	/*
	   printf("\n\n Before computation");
	   traversalTree(root);
	   */


	/*
	 * compute the maximal average for each node
	 */

	computeMaximalMean_without_nonnegative(root);


	/*compute the solution from the tree*/

	computeSolution(x, root);


	/*
	   printf("\n\n After computation");
	   traversalTree(root);
	   */


	/* delete the tree
	*/
	deleteTree(root);
}



/*
 * This is the main function for the full binary tree
 *
 */

void orderTreeBinary(double *x, double *u, int n){
	struct Node * root;

	/*
	 * build the tree using initializeRootBinary for the binary tree
	 *
	 * please make sure that n=2^{depth +1} -1
	 *
	 */

	initializeRootBinary(&root, u, n);

	/*
	   printf("\n\n Before computation");
	   traversalTree(root);
	   */


	/*
	 * compute the maximal average for each node
	 */

	computeMaximalMean(root);


	/*compute the solution from the tree*/

	computeSolution(x, root);


	/*
	   printf("\n\n After computation");
	   traversalTree(root);
	   */


	/* delete the tree
	*/
	deleteTree(root);
}


/*
 * This is the main function for the tree with depth 1
 *
 */

void orderTreeDepth1(double *x, double *u, int n){
	struct Node * root;

	/*
	 * build the tree using initializeRootDepth1 for the tree with depth 1
	 *
	 */

	initializeRootDepth1(&root, u, n);

	/*
	   printf("\n\n Before computation");
	   traversalTree(root);
	   */


	/*
	 * compute the maximal average for each node
	 */

	computeMaximalMean(root);


	/*compute the solution from the tree*/

	computeSolution(x, root);


	/*
	   printf("\n\n After computation");
	   traversalTree(root);
	   */


	/* delete the tree
	*/
	deleteTree(root);
}
#endif   /* ----- #ifndef ORDERTREE_SLEP  ----- */
