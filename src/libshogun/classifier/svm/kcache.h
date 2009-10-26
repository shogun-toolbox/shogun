/***********************************************************************
 * 
 *  LUSH Lisp Universal Shell
 *    Copyright (C) 2002 Leon Bottou, Yann Le Cun, AT&T Corp, NECI.
 *  Includes parts of TL3:
 *    Copyright (C) 1987-1999 Leon Bottou and Neuristique.
 *  Includes selected parts of SN3.2:
 *    Copyright (C) 1991-2001 AT&T Corp.
 * 
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 * 
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 * 
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA
 * 
 ***********************************************************************/

/***********************************************************************
 * $Id: kcache.h,v 1.8 2007/01/25 22:42:09 leonb Exp $
 **********************************************************************/

#ifndef KCACHE_H
#define KCACHE_H

namespace shogun
{
/* ------------------------------------- */
/* GENERIC KERNEL TYPE */


/* --- larank_kernel_t
   This is the type for user defined symmetric kernel functions.
   It returns the Gram matrix element at position <i>,<j>. 
   Argument <closure> represents arbitrary additional information.
*/
#ifndef larank_KERNEL_T_DEFINED
#define larank_KERNEL_T_DEFINED
  typedef double (*larank_kernel_t) (int i, int j, void *closure);
#endif



/* ------------------------------------- */
/* CACHE FOR KERNEL VALUES */


/* --- larank_kcache_t
   This is the opaque data structure for a kernel cache.
*/
  typedef struct larank_kcache_s larank_kcache_t;

/* --- larank_kcache_create
   Returns a cache object for kernel <kernelfun>.
   The cache handles a Gram matrix of size <n>x<n> at most.
   Argument <closure> is passed to the kernel function <kernelfun>.
 */
  larank_kcache_t *larank_kcache_create (larank_kernel_t kernelfunc,
					 void *closure);

/* --- larank_kcache_destroy
   Deallocates a kernel cache object.
*/
  void larank_kcache_destroy (larank_kcache_t * self);

/* --- larank_kcache_set_maximum_size
   Sets the maximum memory size used by the cache.
   Argument <entries> indicates the maximum cache memory in bytes
   The default size is 256Mb.
*/
  void larank_kcache_set_maximum_size (larank_kcache_t * self, long entries);

/* --- larank_kcache_get_maximum_size
   Returns the maximum cache memory.
 */
  long larank_kcache_get_maximum_size (larank_kcache_t * self);

/* --- larank_kcache_get_current_size
   Returns the currently used cache memory.
   This can slighly exceed the value specified by 
   <larank_kcache_set_maximum_size>.
 */
  long larank_kcache_get_current_size (larank_kcache_t * self);

/* --- larank_kcache_query
   Returns the possibly cached value of the Gram matrix element (<i>,<j>).
   This function will not modify the cache geometry.
 */
  double larank_kcache_query (larank_kcache_t * self, int i, int j);

/* --- larank_kcache_query_row
   Returns the <len> first elements of row <i> of the Gram matrix.
   The cache user can modify the order of the row elements
   using the larank_kcache_swap() functions.  Functions larank_kcache_i2r() 
   and larank_kcache_r2i() convert from example index to row position 
   and vice-versa.
*/

  float *larank_kcache_query_row (larank_kcache_t * self, int i, int len);

/* --- larank_kcache_status_row
   Returns the number of cached entries for row i.
*/

  int larank_kcache_status_row (larank_kcache_t * self, int i);

/* --- larank_kcache_discard_row
   Indicates that we wont need row i in the near future.
*/

  void larank_kcache_discard_row (larank_kcache_t * self, int i);


/* --- larank_kcache_i2r
   --- larank_kcache_r2i
   Return an array of integer of length at least <n> containing
   the conversion table from example index to row position and vice-versa. 
*/

  int *larank_kcache_i2r (larank_kcache_t * self, int n);
  int *larank_kcache_r2i (larank_kcache_t * self, int n);


/* --- larank_kcache_swap_rr
   --- larank_kcache_swap_ii
   --- larank_kcache_swap_ri
   Swaps examples in the row ordering table.
   Examples can be specified by indicating their row position (<r1>, <r2>)
   or by indicating the example number (<i1>, <i2>).
*/

  void larank_kcache_swap_rr (larank_kcache_t * self, int r1, int r2);
  void larank_kcache_swap_ii (larank_kcache_t * self, int i1, int i2);
  void larank_kcache_swap_ri (larank_kcache_t * self, int r1, int i2);


/* --- larank_kcache_set_buddy
   This function is called to indicate that the caches <self> and <buddy>
   implement the same kernel function. When a buddy experiences a cache
   miss, it can try querying its buddies instead of calling the 
   kernel function.  Buddy relationship is transitive. */

  void larank_kcache_set_buddy (larank_kcache_t * self,
				larank_kcache_t * buddy);
}
#endif
