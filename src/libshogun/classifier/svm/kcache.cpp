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
 * $Id: kcache.c,v 1.9 2007/01/25 22:42:09 leonb Exp $
 **********************************************************************/

#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "lib/io.h"
#include "lib/Mathematics.h"
#include "classifier/svm/kcache.h"

namespace shogun
{
struct larank_kcache_s
{
  larank_kernel_t func;
  larank_kcache_t *prevbuddy;
  larank_kcache_t *nextbuddy;
  void *closure;
  long maxsize;
  long cursize;
  int l;
  int *i2r;
  int *r2i;
  int maxrowlen;
  /* Rows */
  int *rsize;
  float *rdiag;
  float **rdata;
  int *rnext;
  int *rprev;
  int *qnext;
  int *qprev;
};

static void *
xmalloc (int n)
{
  void *p = malloc (n);
  if (!p)
    SG_SERROR ("Function malloc() has returned zero\n");
  return p;
}

static void *
xrealloc (void *ptr, int n)
{
  if (!ptr)
    ptr = malloc (n);
  else
    ptr = realloc (ptr, n);
  if (!ptr)
    SG_SERROR ("Function realloc has returned zero\n");
  return ptr;
}

static void
xminsize (larank_kcache_t * self, int n)
{
  int ol = self->l;
  if (n > ol)
    {
      int i;
      int nl = CMath::max (256, ol);
      while (nl < n)
	nl = nl + nl;
      self->i2r = (int *) xrealloc (self->i2r, nl * sizeof (int));
      self->r2i = (int *) xrealloc (self->r2i, nl * sizeof (int));
      self->rsize = (int *) xrealloc (self->rsize, nl * sizeof (int));
      self->qnext = (int *) xrealloc (self->qnext, (1 + nl) * sizeof (int));
      self->qprev = (int *) xrealloc (self->qprev, (1 + nl) * sizeof (int));
      self->rdiag = (float *) xrealloc (self->rdiag, nl * sizeof (float));
      self->rdata = (float **) xrealloc (self->rdata, nl * sizeof (float *));
      self->rnext = self->qnext + 1;
      self->rprev = self->qprev + 1;
      for (i = ol; i < nl; i++)
	{
	  self->i2r[i] = i;
	  self->r2i[i] = i;
	  self->rsize[i] = -1;
	  self->rnext[i] = i;
	  self->rprev[i] = i;
	  self->rdata[i] = 0;
	}
      self->l = nl;
    }
}

larank_kcache_t *
larank_kcache_create (larank_kernel_t kernelfunc, void *closure)
{
  larank_kcache_t *self;
  self = (larank_kcache_t *) xmalloc (sizeof (larank_kcache_t));
  memset (self, 0, sizeof (larank_kcache_t));
  self->l = 0;
  self->maxrowlen = 0;
  self->func = kernelfunc;
  self->prevbuddy = self;
  self->nextbuddy = self;
  self->closure = closure;
  self->cursize = sizeof (larank_kcache_t);
  self->maxsize = 256 * 1024 * 1024;
  self->qprev = (int *) xmalloc (sizeof (int));
  self->qnext = (int *) xmalloc (sizeof (int));
  self->rnext = self->qnext + 1;
  self->rprev = self->qprev + 1;
  self->rprev[-1] = -1;
  self->rnext[-1] = -1;
  return self;
}

void
larank_kcache_destroy (larank_kcache_t * self)
{
  if (self)
    {
      int i;
      larank_kcache_t *nb = self->nextbuddy;
      larank_kcache_t *pb = self->prevbuddy;
      pb->nextbuddy = nb;
      nb->prevbuddy = pb;
      /* delete */
      if (self->i2r)
	free (self->i2r);
      if (self->r2i)
	free (self->r2i);
      if (self->rdata)
	for (i = 0; i < self->l; i++)
	  if (self->rdata[i])
	    free (self->rdata[i]);
      if (self->rdata)
	free (self->rdata);
      if (self->rsize)
	free (self->rsize);
      if (self->rdiag)
	free (self->rdiag);
      if (self->qnext)
	free (self->qnext);
      if (self->qprev)
	free (self->qprev);
      memset (self, 0, sizeof (larank_kcache_t));
      free (self);
    }
}

int *
larank_kcache_i2r (larank_kcache_t * self, int n)
{
  xminsize (self, n);
  return self->i2r;
}

int *
larank_kcache_r2i (larank_kcache_t * self, int n)
{
  xminsize (self, n);
  return self->r2i;
}

static void
xextend (larank_kcache_t * self, int k, int nlen)
{
  int olen = self->rsize[k];
  if (nlen > olen)
    {
      float *ndata = (float *) xmalloc (nlen * sizeof (float));
      if (olen > 0)
	{
	  float *odata = self->rdata[k];
	  memcpy (ndata, odata, olen * sizeof (float));
	  free (odata);
	}
      self->rdata[k] = ndata;
      self->rsize[k] = nlen;
      self->cursize += (long) (nlen - olen) * sizeof (float);
      self->maxrowlen = CMath::max (self->maxrowlen, nlen);
    }
}

static void
xtruncate (larank_kcache_t * self, int k, int nlen)
{
  int olen = self->rsize[k];
  if (nlen < olen)
    {
      float *ndata;
      float *odata = self->rdata[k];
      if (nlen > 0)
	{
	  ndata = (float *) xmalloc (nlen * sizeof (float));
	  memcpy (ndata, odata, nlen * sizeof (float));
	}
      else
	{
	  ndata = 0;
	  self->rnext[self->rprev[k]] = self->rnext[k];
	  self->rprev[self->rnext[k]] = self->rprev[k];
	  self->rnext[k] = self->rprev[k] = k;
	}
      free (odata);
      self->rdata[k] = ndata;
      self->rsize[k] = nlen;
      self->cursize += (long) (nlen - olen) * sizeof (float);
    }
}

static void
xswap (larank_kcache_t * self, int i1, int i2, int r1, int r2)
{
  /* swap row data */
  if (r1 < self->maxrowlen || r2 < self->maxrowlen)
    {
      int mrl = 0;
      int k = self->rnext[-1];
      while (k >= 0)
	{
	  int nk = self->rnext[k];
	  int n = self->rsize[k];
	  int rr = self->i2r[k];
	  float *d = self->rdata[k];
	  if (r1 < n)
	    {
	      if (r2 < n)
		{
		  float t1 = d[r1];
		  float t2 = d[r2];
		  d[r1] = t2;
		  d[r2] = t1;
		}
	      else if (rr == r2)
		{
		  d[r1] = self->rdiag[k];
		}
	      else
		{
		  int arsize = self->rsize[i2];
		  if (rr < arsize && rr != r1)
		    d[r1] = self->rdata[i2][rr];
		  else
		    xtruncate (self, k, r1);
		}
	    }
	  else if (r2 < n)
	    {
	      if (rr == r1)
		{
		  d[r2] = self->rdiag[k];
		}
	      else
		{
		  int arsize = self->rsize[i1];
		  if (rr < arsize && rr != r2)
		    d[r2] = self->rdata[i1][rr];
		  else
		    xtruncate (self, k, r2);
		}
	    }
	  mrl = CMath::max (mrl, self->rsize[k]);
	  k = nk;
	}
      self->maxrowlen = mrl;
    }
  /* swap r2i and i2r */
  self->r2i[r1] = i2;
  self->r2i[r2] = i1;
  self->i2r[i1] = r2;
  self->i2r[i2] = r1;
}

void
larank_kcache_swap_rr (larank_kcache_t * self, int r1, int r2)
{
  xminsize (self, 1 + CMath::max (r1, r2));
  xswap (self, self->r2i[r1], self->r2i[r2], r1, r2);
}

void
larank_kcache_swap_ii (larank_kcache_t * self, int i1, int i2)
{
  xminsize (self, 1 + CMath::max (i1, i2));
  xswap (self, i1, i2, self->i2r[i1], self->i2r[i2]);
}

void
larank_kcache_swap_ri (larank_kcache_t * self, int r1, int i2)
{
  xminsize (self, 1 + CMath::max (r1, i2));
  xswap (self, self->r2i[r1], i2, r1, self->i2r[i2]);
}

static double
xquery (larank_kcache_t * self, int i, int j)
{
  /* search buddies */
  larank_kcache_t *cache = self->nextbuddy;
  do
    {
      int l = cache->l;
      if (i < l && j < l)
	{
	  int s = cache->rsize[i];
	  int p = cache->i2r[j];
	  if (p < s)
	    return cache->rdata[i][p];
	  if (i == j && s >= 0)
	    return cache->rdiag[i];
	  p = cache->i2r[i];
	  s = cache->rsize[j];
	  if (p < s)
	    return cache->rdata[j][p];
	}
      cache = cache->nextbuddy;
    }
  while (cache != self);
  /* compute */
  return (*self->func) (i, j, self->closure);
}


double
larank_kcache_query (larank_kcache_t * self, int i, int j)
{
  ASSERT (self);
  ASSERT (i >= 0);
  ASSERT (j >= 0);
  return xquery (self, i, j);
}


static void
xpurge (larank_kcache_t * self)
{
  if (self->cursize > self->maxsize)
    {
      int k = self->rprev[-1];
      while (self->cursize > self->maxsize && k != self->rnext[-1])
	{
	  int pk = self->rprev[k];
	  xtruncate (self, k, 0);
	  k = pk;
	}
    }
}

float *
larank_kcache_query_row (larank_kcache_t * self, int i, int len)
{
  ASSERT (i >= 0);
  if (i < self->l && len <= self->rsize[i])
    {
      self->rnext[self->rprev[i]] = self->rnext[i];
      self->rprev[self->rnext[i]] = self->rprev[i];
    }
  else
    {
      int olen, p;
      float *d;
      if (i >= self->l || len >= self->l)
	xminsize (self, CMath::max (1 + i, len));
      olen = self->rsize[i];
      if (olen < len)
	{
	  if (olen < 0)
	    {
	      self->rdiag[i] = (*self->func) (i, i, self->closure);
	      olen = self->rsize[i] = 0;
	    }
	  xextend (self, i, len);
	  d = self->rdata[i];
	  self->rsize[i] = olen;
	  for (p = olen; p < len; p++)
	    d[p] = larank_kcache_query (self, self->r2i[p], i);
	  self->rsize[i] = len;
	}
      self->rnext[self->rprev[i]] = self->rnext[i];
      self->rprev[self->rnext[i]] = self->rprev[i];
      xpurge (self);
    }
  self->rprev[i] = -1;
  self->rnext[i] = self->rnext[-1];
  self->rnext[self->rprev[i]] = i;
  self->rprev[self->rnext[i]] = i;
  return self->rdata[i];
}

int
larank_kcache_status_row (larank_kcache_t * self, int i)
{
  ASSERT (self);
  ASSERT (i >= 0);
  if (i < self->l)
    return CMath::max (0, self->rsize[i]);
  return 0;
}

void
larank_kcache_discard_row (larank_kcache_t * self, int i)
{
  ASSERT (self);
  ASSERT (i >= 0);
  if (i < self->l && self->rsize[i] > 0)
    {
      self->rnext[self->rprev[i]] = self->rnext[i];
      self->rprev[self->rnext[i]] = self->rprev[i];
      self->rprev[i] = self->rprev[-1];
      self->rnext[i] = -1;
      self->rnext[self->rprev[i]] = i;
      self->rprev[self->rnext[i]] = i;
    }
}

void
larank_kcache_set_maximum_size (larank_kcache_t * self, long entries)
{
  ASSERT (self);
  ASSERT (entries > 0);
  self->maxsize = entries;
  xpurge (self);
}

long
larank_kcache_get_maximum_size (larank_kcache_t * self)
{
  ASSERT (self);
  return self->maxsize;
}

long
larank_kcache_get_current_size (larank_kcache_t * self)
{
  ASSERT (self);
  return self->cursize;
}

void
larank_kcache_set_buddy (larank_kcache_t * self, larank_kcache_t * buddy)
{
  larank_kcache_t *p = self;
  larank_kcache_t *selflast = self->prevbuddy;
  larank_kcache_t *buddylast = buddy->prevbuddy;
  /* check functions are identical */
  ASSERT (self->func == buddy->func);
  /* make sure we are not already buddies */
  do
    {
      if (p == buddy)
	return;
      p = p->nextbuddy;
    }
  while (p != self);
  /* link */
  selflast->nextbuddy = buddy;
  buddy->prevbuddy = selflast;
  buddylast->nextbuddy = self;
  self->prevbuddy = buddylast;
}
}
