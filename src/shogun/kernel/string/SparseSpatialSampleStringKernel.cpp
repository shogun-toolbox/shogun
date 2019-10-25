/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Math.h>
#include <shogun/kernel/string/SparseSpatialSampleStringKernel.h>
#include <shogun/features/StringFeatures.h>

using namespace shogun;

SparseSpatialSampleStringKernel::SparseSpatialSampleStringKernel()
: StringKernel<char>(0), t(2), d(5)
{
}

SparseSpatialSampleStringKernel::SparseSpatialSampleStringKernel(const std::shared_ptr<StringFeatures<char>>& l,
		const std::shared_ptr<StringFeatures<char>>& r) : StringKernel<char>(0), t(2), d(5)
{
	init(l, r);
}

bool SparseSpatialSampleStringKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	StringKernel<char>::init(l, r);
	return init_normalizer();
}

void SparseSpatialSampleStringKernel::cleanup()
{
	Kernel::cleanup();
}

SparseSpatialSampleStringKernel::~SparseSpatialSampleStringKernel()
{
}

SSKFeatures *SparseSpatialSampleStringKernel::extractTriple(int **S, int *len, int nStr, int d1, int d2)
{
	int i, j;
	int n, nfeat;
	int *group;
	int *features;
	int *s;
	int c;
	SSKFeatures *F;

	nfeat = 0;
	for (i = 0; i < nStr; ++i)
		nfeat += len[i] - d1 -d2;
	group = SG_MALLOC(int, nfeat);
	features = SG_MALLOC(int, nfeat*3);
	c = 0;
	for (i = 0; i < nStr; ++i)
	{
		n = len[i] - d1 - d2;
		s = S[i];
		for (j = 0; j < n; ++j)
		{
			features[c] = s[j];
			features[c+nfeat] = s[j+d1];
			features[c+2*nfeat] = s[j+d1+d2];
			group[c] = i;
			c++;
		}
	}
	ASSERT(nfeat==c)
	F = SG_MALLOC(SSKFeatures, 1);
	(*F).features = features;
	(*F).group = group;
	(*F).n = nfeat;
	return F;
}


SSKFeatures *SparseSpatialSampleStringKernel::extractDouble(int **S, int *len, int nStr, int d1)
{
	int i, j;
	int n, nfeat;
	int *group;
	int *features;
	int *s;
	int c;
	SSKFeatures *F;

	nfeat = 0;
	for (i = 0; i < nStr; ++i)
		nfeat += len[i] - d1;
	group = SG_MALLOC(int, nfeat);
	features = SG_MALLOC(int, nfeat*2);
	c = 0;
	for (i = 0; i < nStr; ++i)
	{
		n = len[i] - d1;
		s = S[i];
		for (j = 0; j < n; ++j)
		{
			features[c] = s[j];
			features[c+nfeat] = s[j+d1];
			group[c] = i;
			c++;
		}
	}
	if (nfeat!=c)
		printf("Something is wrong...\n");
	F = SG_MALLOC(SSKFeatures, 1);
	(*F).features = features;
	(*F).group = group;
	(*F).n = nfeat;
	return F;
}


void SparseSpatialSampleStringKernel::compute_double(int32_t idx_a, int32_t idx_b)
{
	int d1;
	SSKFeatures *features;
	int *sortIdx;
	int *features_srt;
	int *group_srt;
	int maxIdx;
	int **S=NULL;
	int *len=NULL;
	int nStr=0, nfeat;
	int i;
	int na=0;
	int *K=NULL;

	for (d1 = 1; d1 <= d; ++d1)
	{
		if ( isVerbose ) printf("Extracting features..."), fflush( stdout );
		features = extractDouble(S,len,nStr,d1);
		nfeat = (*features).n;
		printf("d=%d: %d features\n", d1, nfeat);
		maxIdx = 0;
		for (i = 0; i < nfeat*2; ++i)
			maxIdx = maxIdx > (*features).features[i] ? maxIdx : (*features).features[i];
		if (na < maxIdx+1)
		{
			printf("WARNING: Sequence elements are outside the specified range [0,%d]\n",na);
			printf("\tUsing [0,%d] instead\n", maxIdx);
			na = maxIdx+1;
		}
		if (isVerbose)
		{
			printf("done.\n");
			printf("Sorting...");
			fflush( stdout );
		}
		sortIdx = cntsrtna((*features).features,2,(*features).n,na);
		if (isVerbose)  printf("done.\n");
		features_srt = SG_MALLOC(int, nfeat*2);
		group_srt = SG_MALLOC(int, nfeat);
		for (i = 0; i < nfeat; ++i)
		{
			features_srt[i]=(*features).features[sortIdx[i]];
			features_srt[i+nfeat]=(*features).features[sortIdx[i]+nfeat];
			group_srt[i] = (*features).group[sortIdx[i]];
		}
		SG_FREE(sortIdx);
		SG_FREE((*features).features);
		SG_FREE((*features).group);
		SG_FREE(features);
		if (isVerbose)
		{
			printf("Counting...");
			fflush( stdout );
		}
		countAndUpdate(K,features_srt,group_srt,2,nfeat,nStr);
		SG_FREE(features_srt);
		SG_FREE(group_srt);
		if (isVerbose)
		{
			printf("done.");
		}
	}
}

void SparseSpatialSampleStringKernel::compute_triple(int32_t idx_a, int32_t idx_b)
{
	int d1, d2;
	SSKFeatures *features;
	int *sortIdx;
	int *features_srt;
	int *group_srt;
	int maxIdx;
	int **S=NULL;
	int *len=NULL;
	int nStr=0, nfeat;
	int i;
	int na=0;
	int *K=NULL;

	for (d1 = 1; d1 <= d; ++d1)
	{
		for (d2 = 1; d2 <= d; ++d2)
		{
			if (isVerbose)
			{
				printf("Extracting features...");
				fflush( stdout );
			}
			features = extractTriple(S,len,nStr,d1,d2);
			nfeat = (*features).n;
			printf("(%d,%d): %d features\n", d1, d2, nfeat);
			maxIdx = 0;
			for (i = 0; i < nfeat*3; ++i)
				maxIdx = maxIdx > (*features).features[i] ? maxIdx : (*features).features[i];
			if (na < maxIdx+1)
			{
				printf("WARNING: Sequence elements are outside the specified range [0,%d]\n",na);
				printf("\tUsing [0,%d] instead\n", maxIdx);
				na = maxIdx+1;
			}
			if (isVerbose)
			{
				printf("done.\n");
				printf("Sorting...");
				fflush( stdout );
			}
			sortIdx = cntsrtna((*features).features,3,(*features).n,na);
			if (isVerbose)
			{
				printf("done.\n");
			}
			features_srt = SG_MALLOC(int, nfeat*3);
			group_srt = SG_MALLOC(int, nfeat);
			for (i = 0; i < nfeat; ++i)
			{
				features_srt[i]=(*features).features[sortIdx[i]];
				features_srt[i+nfeat]=(*features).features[sortIdx[i]+nfeat];
				features_srt[i+2*nfeat]=(*features).features[sortIdx[i]+2*nfeat];
				group_srt[i] = (*features).group[sortIdx[i]];
			}
			SG_FREE(sortIdx);
			SG_FREE((*features).features);
			SG_FREE((*features).group);
			SG_FREE(features);
			if (isVerbose)
			{
				printf("Counting...");
				fflush( stdout );
			}
			countAndUpdate(K,features_srt,group_srt,3,nfeat,nStr);
			SG_FREE(features_srt);
			SG_FREE(group_srt);
			if (isVerbose)
			{
				printf("done.\n");
			}
		}
	}
}

void SparseSpatialSampleStringKernel::countAndUpdate(int *outK, int *sx, int *g, int k, int r, int nStr)
{
	char same;
	int i, j;
	int cu;
	long int ucnt;
	long int startInd, endInd, j1;
	int *curfeat, *ucnts, *updind;

	curfeat = SG_MALLOC(int, k);
	ucnts = SG_MALLOC(int, nStr);
	updind = SG_MALLOC(int, nStr);
	i = 0;
	ucnt = 0;
	while (i<r)
	{
		for (j = 0; j < k; ++j)
			curfeat[j]=sx[i+j*r];
		same=1;
		for (j = 0;j < k; ++j)
			if (curfeat[j]!=sx[i+j*r])
			{
				same=0;
				break;
			}
		startInd=i;
		while (same && i<r)
		{
			i++;
			if (i >= r) break;
			same = 1;
			for (j = 0; j < k; ++j)
				if (curfeat[j]!=sx[i+j*r])
				{
					same=0;
					break;
				}
		}
		endInd= (i<r) ? (i-1):(r-1);
		ucnt++;
		if ((long int)endInd-startInd+1>2*nStr)
		{
			for (j = 0; j < nStr; ++j) ucnts[j]=0;
			for (j = startInd;j <= endInd; ++j)  ucnts[g[j]]++;
			cu=0;
			for (j=0;j<nStr;j++)
			{
				if (ucnts[j]>0)
				{
					updind[cu] = j;
					cu++;
				}
			}
			for (j=0;j<cu;j++)
				for (j1=0;j1<cu;j1++)
					outK[updind[j]+updind[j1]*nStr]+=ucnts[updind[j]]*ucnts[updind[j1]];
		}
		else
		{
			for (j = startInd;j <= endInd; ++j)
				for (j1 = startInd;j1 <= endInd; ++j1)
					outK[ g[j]+nStr*g[j1] ]++;
		}
	}
	SG_FREE(updind);
	SG_FREE(ucnts);
	SG_FREE(curfeat);
}

int *SparseSpatialSampleStringKernel::cntsrtna(int *sx, int k, int r, int na)
{
	int *sxc, *bc, *sxl, *cc, *regroup;
	int i, j;

	sxc = SG_MALLOC(int, na);
	bc  = SG_MALLOC(int, na);
	sxl = SG_MALLOC(int, r);
	cc  = SG_MALLOC(int, r);
	regroup = SG_MALLOC(int, r);

	for (i = 0; i < r; ++i)
		regroup[i]=i;
	for (j = k-1; j >= 0; --j)
	{
		for(i = 0; i < na; ++i)
			sxc[i]=0;
		for (i = 0; i < r; ++i)
		{
			cc[i]=sx[regroup[i]+j*r];
			sxc[ cc[i] ]++;
		}
		bc[0]=0;
		for (i = 1;i < na; ++i)
			bc[i]=bc[i-1]+sxc[i-1];
		for (i = 0; i < r; ++i)
			sxl[bc[ cc[i] ]++] = regroup[i];
		for (i = 0; i < r; ++i)
			regroup[i] = sxl[i];
	}
	SG_FREE(sxl); SG_FREE(bc); SG_FREE(sxc); SG_FREE(cc);

	return regroup;
}

float64_t SparseSpatialSampleStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	if (t==2)
		compute_double(idx_a, idx_b);
	if (t==3)
		compute_triple(idx_a, idx_b);

	error("t out of range - shouldn't happen");
	return -1;
}
