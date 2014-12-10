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

#ifndef  ALTRA_SLEP
#define  ALTRA_SLEP

#include <shogun/mathematics/Math.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/*
 * Important Notice: September 20, 2010
 *
 * In this head file, we assume that the features in the tree strucutre
 * are well ordered. That is to say, the indices of the left nodes is always less
 * than the right nodes. Ideally, this can be achieved by reordering the features.
 *
 * The advantage of this ordered features is that, we donot need to use an explicit
 * variable for recording the indices.
 *
 * To deal with the more general case when the features might not be well ordered,
 * we provide the functions in the head file "general_altra.h". Compared with the files in this head file,
 * we need an additional parameter G, which contains the indices of the nodes.
 *
 *
 */

/*
 * -------------------------------------------------------------------
 *                       Functions and parameter
 * -------------------------------------------------------------------
 *
 * altra solves the following problem
 *
 * 1/2 \|x-v\|^2 + \sum \lambda_i \|x_{G_i}\|,
 *
 * where x and v are of dimension n,
 *       \lambda_i >=0, and G_i's follow the tree structure
 *
 * It is implemented in Matlab as follows:
 *
 * x=altra(v, n, ind, nodes);
 *
 * ind is a 3 x nodes matrix.
 *       Each column corresponds to a node.
 *
 *       The first element of each column is the starting index,
 *       the second element of each column is the ending index
 *       the third element of each column corrreponds to \lambbda_i.
 *
 * -------------------------------------------------------------------
 *                       Notices:
 * -------------------------------------------------------------------
 *
 * 1. The nodes in the parameter "ind" should be given in the 
 *    either
 *           the postordering of depth-first traversal
 *    or 
 *           the reverse breadth-first traversal.
 *
 * 2. When each elements of x are penalized via the same L1 
 *    (equivalent to the L2 norm) parameter, one can simplify the input
 *    by specifying 
 *           the "first" column of ind as (-1, -1, lambda)
 *
 *    In this case, we treat it as a single "super" node. Thus in the value
 *    nodes, we only count it once.
 *
 * 3. The values in "ind" are in [1,n].
 *
 * 4. The third element of each column should be positive. The program does
 *    not check the validity of the parameter. 
 *
 *    It is still valid to use the zero regularization parameter.
 *    In this case, the program does not change the values of 
 *    correponding indices.
 *    
 *
 * -------------------------------------------------------------------
 *                       History:
 * -------------------------------------------------------------------
 *
 * Composed by Jun Liu on April 20, 2010
 *
 * For any question or suggestion, please email j.liu@asu.edu.
 *
 */
void altra(double *x, double *v, int n, double *ind, int nodes, double mult=1.0);

/*
 * altra_mt is a generalization of altra to the 
 * 
 * multi-task learning scenario (or equivalently the multi-class case)
 *
 * altra_mt(X, V, n, k, ind, nodes);
 *
 * It applies altra for each row (1xk) of X and V
 *
 */
void altra_mt(double *X, double *V, int n, int k, double *ind, int nodes, double mult=1.0);

/*
 * compute
 *  lambda2_max=computeLambda2Max(x,n,ind,nodes);
 *
 * compute the 2 norm of each group, which is divided by the ind(3,:),
 * then the maximum value is returned
 */
/*
 *This function does not consider the case ind={[-1, -1, 100]',...}
 *
 *This functions is not used currently.
 */
void computeLambda2Max(double *lambda2_max, double *x, int n, double *ind, int nodes);

/*
 * -------------------------------------------------------------------
 *                       Function and parameter
 * -------------------------------------------------------------------
 *
 * treeNorm compute
 *
 *        \sum \lambda_i \|x_{G_i}\|,
 *
 * where x is of dimension n,
 *       \lambda_i >=0, and G_i's follow the tree structure
 *
 * The file is implemented in the following in Matlab:
 *
 * tree_norm=treeNorm(x, n, ind,nodes);
 */
double treeNorm(double *x, int ldx, int n, double *ind, int nodes);

/*
 * -------------------------------------------------------------------
 *                       Function and parameter
 * -------------------------------------------------------------------
 *
 * findLambdaMax compute
 * 
 * the lambda_{max} that achieves a zero solution for
 *
 *     min  1/2 \|x-v\|^2 +  \lambda_{\max} * \sum  w_i \|x_{G_i}\|,
 *
 * where x is of dimension n,
 *       w_i >=0, and G_i's follow the tree structure
 *
 * The file is implemented in the following in Matlab:
 *
 * lambdaMax=findLambdaMax(v, n, ind,nodes);
 */
double findLambdaMax(double *v, int n, double *ind, int nodes);

/*
 * findLambdaMax_mt is a generalization of findLambdaMax to the 
 * 
 * multi-task learning scenario (or equivalently the multi-class case)
 *
 * lambdaMax=findLambdaMax_mt(X, V, n, k, ind, nodes);
 *
 * It applies findLambdaMax for each row (1xk) of X and V
 *
 */
double findLambdaMax_mt(double *V, int n, int k, double *ind, int nodes);
#endif   /* ----- #ifndef ALTRA_SLEP  ----- */

