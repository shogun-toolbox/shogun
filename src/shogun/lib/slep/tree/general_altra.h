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

#ifndef  GENERAL_ALTRA_SLEP
#define  GENERAL_ALTRA_SLEP

#include <shogun/lib/config.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/*
 * Important Notice: September 20, 2010
 *
 * In this head file, we deal with the case that the features might not be well ordered.
 * 
 * If the features in the tree strucutre are well ordered, i.e., the indices of the left nodes is always less
 * than the right nodes, please refer to "altra.h".
 *
 * The advantage of "altra.h" is that, we donot need to use an explicit
 * variable for recording the indices.
 *
 *
 */

/*
 * -------------------------------------------------------------------
 *                       Functions and parameter
 * -------------------------------------------------------------------
 *
 * general_altra solves the following problem
 *
 * 1/2 \|x-v\|^2 + \sum \lambda_i \|x_{G_i}\|,
 *
 * where x and v are of dimension n,
 *       \lambda_i >=0, and G_i's follow the tree structure
 *
 * It is implemented in Matlab as follows:
 *
 * x=general_altra(v, n, G, ind, nodes);
 *
 * G contains the indices of the groups.
 *   It is a row vector. Its length equals to \sum_i \|G_i\|.
 *   If all the entries are penalized with L1 norm,
 *      its length is \sum_i \|G_i\| - n.
 *
 * ind is a 3 x nodes matrix.
 *       Each column corresponds to a node.
 *
 *       The first element of each column is the starting index,
 *       the second element of each column is the ending index
 *       the third element of each column corrreponds to \lambbda_i.
 *
 *
 *
 * The following example shows how G and ind works:
 *
 * G={ {1, 2}, {4, 5}, {3, 6}, {7, 8},
 *     {1, 2, 3, 6}, {4, 5, 7, 8}, 
 *     {1, 2, 3, 4, 5, 6, 7, 8} }.
 *
 * ind={ [1, 2, 100]', [3, 4, 100]', [5, 6, 100]', [7, 8, 100]',
 *       [9, 12, 100]', [13, 16, 100]', [17, 24, 100]' }
 * 
 * where "100" denotes the weight for the nodes.
 *
 *
 *
 * -------------------------------------------------------------------
 *                       Notices:
 * -------------------------------------------------------------------
 *
 * 1. The features in the tree might not be well ordered. Otherwise, you are
 *    suggested to use "altra.h".
 *
 * 2. When each elements of x are penalized via the same L1 
 *    (equivalent to the L2 norm) parameter, one can simplify the input
 *    by specifying 
 *           the "first" column of ind as (-1, -1, lambda)
 *
 *    In this case, we treat it as a single "super" node. Thus in the value
 *    nodes, we only count it once.
 *
 * 3. The values in "ind" are in [1,length(G)].
 *
 * 4. The third element of each column should be positive. The program does
 *    not check the validity of the parameter. 
 *
 * 5. The values in G should be within [1, n].
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
void general_altra(double *x, double *v, int n, double *G, double *ind, int nodes, double mult=1.0);

/*
 * altra_mt is a generalization of altra to the 
 * 
 * multi-task learning scenario (or equivalently the multi-class case)
 *
 * altra_mt(X, V, n, k, G, ind, nodes);
 *
 * It applies altra for each row (1xk) of X and V
 *
 */
void general_altra_mt(double *X, double *V, int n, int k, double *G, double *ind, int nodes, double mult=1.0);

/*
 * compute
 *  lambda2_max=general_computeLambda2Max(x,n,G, ind,nodes);
 *
 * compute the 2 norm of each group, which is divided by the ind(3,:),
 * then the maximum value is returned
 */
/*
 *This function does not consider the case ind={[-1, -1, 100]',...}
 *
 *This functions is not used currently.
 */
void general_computeLambda2Max(double *lambda2_max, double *x, int n, double *G, double *ind, int nodes);

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
 * tree_norm=general_treeNorm(x, n, G, ind,nodes);
 */
double general_treeNorm(double *x, int ldx, int n, double *G, double *ind, int nodes);

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
 * lambdaMax=general_findLambdaMax(v, n, G, ind,nodes);
 */
double general_findLambdaMax(double *v, int n, double *G, double *ind, int nodes);

/*
 * findLambdaMax_mt is a generalization of findLambdaMax to the 
 * 
 * multi-task learning scenario (or equivalently the multi-class case)
 *
 * lambdaMax=general_findLambdaMax_mt(X, V, n, k, G, ind, nodes);
 *
 * It applies findLambdaMax for each row (1xk) of X and V
 *
 */
double general_findLambdaMax_mt(double *V, int n, int k, double *G, double *ind, int nodes);
#endif   /* ----- #ifndef GENERAL_ALTRA_SLEP  ----- */

