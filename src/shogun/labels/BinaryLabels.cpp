#include <shogun/labels/DenseLabels.h>
#include <shogun/labels/BinaryLabels.h>

using namespace shogun;

CBinaryLabels::CBinaryLabels() : CDenseLabels()
{
}

CBinaryLabels::CBinaryLabels(int32_t num_labels) : CDenseLabels(num_labels)
{
}

#ifndef SWIGJAVA
CBinaryLabels::CBinaryLabels(SGVector<int32_t> src) : CDenseLabels()
{
	SGVector<float64_t> values(src.vlen);
	for (int32_t i=0; i<values.vlen; i++)
		values[i] = src[i];
	set_int_labels(src);
	set_values(values);
}

CBinaryLabels::CBinaryLabels(SGVector<int64_t> src) : CDenseLabels()
{
	SGVector<float64_t> values(src.vlen);
	for (int32_t i=0; i<values.vlen; i++)
		values[i] = src[i];
	set_int_labels(src);
	set_values(values);
}
#endif

CBinaryLabels::CBinaryLabels(SGVector<float64_t> src, float64_t threshold) : CDenseLabels()
{
	SGVector<float64_t> labels(src.vlen);
	for (int32_t i=0; i<labels.vlen; i++)
		labels[i] = src[i]+threshold>=0 ? +1.0 : -1.0;
	set_labels(labels);
	set_values(src);
}

CBinaryLabels::CBinaryLabels(CFile* loader) : CDenseLabels(loader)
{
}

CBinaryLabels* CBinaryLabels::obtain_from_generic(CLabels* base_labels)
{
	if ( base_labels->get_label_type() == LT_BINARY )
		return (CBinaryLabels*) base_labels;
	else
		SG_SERROR("base_labels must be of dynamic type CBinaryLabels")

	return NULL;
}


void CBinaryLabels::ensure_valid(const char* context)
{
    CDenseLabels::ensure_valid(context);
    bool found_plus_one=false;
    bool found_minus_one=false;

    int32_t subset_size=get_num_labels();
    for (int32_t i=0; i<subset_size; i++)
    {
        int32_t real_i=m_subset_stack->subset_idx_conversion(i);
        if (m_labels[real_i]==+1.0)
            found_plus_one=true;
        else if (m_labels[real_i]==-1.0)
            found_minus_one=true;
        else
        {
            SG_ERROR("%s%sNot a two class labeling label[%d]=%f (only +1/-1 "
                    "allowed)\n", context?context:"", context?": ":"", i, m_labels[real_i]);
        }
    }
    
    if (!found_plus_one)
    {
        SG_ERROR("%s%sNot a two class labeling - no positively labeled examples found\n",
                context?context:"", context?": ":"");
    }

    if (!found_minus_one)
    {
        SG_ERROR("%s%sNot a two class labeling - no negatively labeled examples found\n",
                context?context:"", context?": ":"");
    }
}

ELabelType CBinaryLabels::get_label_type()
{
	return LT_BINARY;
}

void CBinaryLabels::scores_to_probabilities()
{
	SG_DEBUG("entering CBinaryLabels::scores_to_probabilities()\n")

	REQUIRE(m_current_values.vector, "%s::scores_to_probabilities() requires "
			"values vector!\n", get_name());

	/* count prior0 and prior1 if needed */
	int32_t prior0=0;
	int32_t prior1=0;
	SG_DEBUG("counting number of positive and negative labels\n")
	{
		for (index_t i=0; i<m_current_values.vlen; ++i)
		{
			if (m_current_values[i]>0)
				prior1++;
			else
				prior0++;
		}
	}
	SG_DEBUG("%d pos; %d neg\n", prior1, prior0)

	/* parameter setting */
	/* maximum number of iterations */
	index_t maxiter=100;

	/* minimum step taken in line search */
	float64_t minstep=1E-10;

	/* for numerically strict pd of hessian */
	float64_t sigma=1E-12;
	float64_t eps=1E-5;

	/* construct target support */
	float64_t hiTarget=(prior1+1.0)/(prior1+2.0);
	float64_t loTarget=1/(prior0+2.0);
	index_t length=prior1+prior0;

	SGVector<float64_t> t(length);
	for (index_t i=0; i<length; ++i)
	{
		if (m_current_values[i]>0)
			t[i]=hiTarget;
		else
			t[i]=loTarget;
	}

	/* initial Point and Initial Fun Value */
	/* result parameters of sigmoid */
	float64_t a=0;
	float64_t b=CMath::log((prior0+1.0)/(prior1+1.0));
	float64_t fval=0.0;

	for (index_t i=0; i<length; ++i)
	{
		float64_t fApB=m_current_values[i]*a+b;
		if (fApB>=0)
			fval+=t[i]*fApB+CMath::log(1+CMath::exp(-fApB));
		else
			fval+=(t[i]-1)*fApB+CMath::log(1+CMath::exp(fApB));
	}

	index_t it;
	float64_t g1;
	float64_t g2;
	for (it=0; it<maxiter; ++it)
	{
		SG_DEBUG("Iteration %d, a=%f, b=%f, fval=%f\n", it, a, b, fval)

		/* Update Gradient and Hessian (use H' = H + sigma I) */
		float64_t h11=sigma; //Numerically ensures strict PD
		float64_t h22=h11;
		float64_t h21=0;
		g1=0;
		g2=0;

		for (index_t i=0; i<length; ++i)
		{
			float64_t fApB=m_current_values[i]*a+b;
			float64_t p;
			float64_t q;
			if (fApB>=0)
			{
				p=CMath::exp(-fApB)/(1.0+CMath::exp(-fApB));
				q=1.0/(1.0+CMath::exp(-fApB));
			}
			else
			{
				p=1.0/(1.0+CMath::exp(fApB));
				q=CMath::exp(fApB)/(1.0+CMath::exp(fApB));
			}

			float64_t d2=p*q;
			h11+=m_current_values[i]*m_current_values[i]*d2;
			h22+=d2;
			h21+=m_current_values[i]*d2;
			float64_t d1=t[i]-p;
			g1+=m_current_values[i]*d1;
			g2+=d1;
		}

		/* Stopping Criteria */
		if (CMath::abs(g1)<eps && CMath::abs(g2)<eps)
			break;

		/* Finding Newton direction: -inv(H') * g */
		float64_t det=h11*h22-h21*h21;
		float64_t dA=-(h22*g1-h21*g2)/det;
		float64_t dB=-(-h21*g1+h11*g2)/det;
		float64_t gd=g1*dA+g2*dB;

		/* Line Search */
		float64_t stepsize=1;

		while (stepsize>=minstep)
		{
			float64_t newA=a+stepsize*dA;
			float64_t newB=b+stepsize*dB;

			/* New function value */
			float64_t newf=0.0;
			for (index_t i=0; i<length; ++i)
			{
				float64_t fApB=m_current_values[i]*newA+newB;
				if (fApB>=0)
					newf+=t[i]*fApB+CMath::log(1+CMath::exp(-fApB));
				else
					newf+=(t[i]-1)*fApB+CMath::log(1+CMath::exp(fApB));
			}

			/* Check sufficient decrease */
			if (newf<fval+0.0001*stepsize*gd)
			{
				a=newA;
				b=newB;
				fval=newf;
				break;
			}
			else
				stepsize=stepsize/2.0;
		}

		if (stepsize<minstep)
		{
			SG_WARNING("%s::scores_to_probabilities(): line search fails, A=%f, "
					"B=%f, g1=%f, g2=%f, dA=%f, dB=%f, gd=%f\n",
					get_name(), a, b, g1, g2, dA, dB, gd);
		}
	}

	if (it>=maxiter-1)
	{
		SG_WARNING("%s::scores_to_probabilities(): reaching maximal iterations,"
				" g1=%f, g2=%f\n", get_name(), g1, g2);
	}

	SG_DEBUG("fitted sigmoid: a=%f, b=%f\n", a, b)

	/* now the sigmoid is fitted, convert all values to probabilities */
	for (index_t i=0; i<m_current_values.vlen; ++i)
	{
		float64_t fApB=m_current_values[i]*a+b;
		m_current_values[i]=fApB>=0 ? CMath::exp(-fApB)/(1.0+exp(-fApB)) :
				1.0/(1+CMath::exp(fApB));
	}

	SG_DEBUG("leaving CBinaryLabels::scores_to_probabilities()\n")
}
