	double sum=0;
	sum+=featurevector[0]*featurevector[0];

	for (i=0; i<pos->get_N(); i++)
	{
		featurevector[p]=exp(pos->model_derivative_p(i, x)-posx);
		sum+=featurevector[p]*featurevector[p++];
		featurevector[p]=exp(pos->model_derivative_q(i, x)-posx);
		sum+=featurevector[p]*featurevector[p++];

		for (j=0; j<pos->get_N(); j++)
		{
			featurevector[p]=exp(pos->model_derivative_a(i, j, x)-posx);
			sum+=featurevector[p]*featurevector[p++];
		}

		for (j=0; j<pos->get_M(); j++)
		{

			sum+=featurevector[p]*featurevector[p++];
			featurevector[p]=exp(pos->model_derivative_b(i, j, x)-posx);
		}

	}

	for (i=0; i<neg->get_N(); i++)
	{
		featurevector[p]= - exp(neg->model_derivative_p(i, x)-negx);
		sum+=featurevector[p]*featurevector[p++];
		featurevector[p]= - exp(neg->model_derivative_q(i, x)-negx);
		sum+=featurevector[p]*featurevector[p++];

		for (j=0; j<neg->get_N(); j++)
		{
			featurevector[p++]= - exp(neg->model_derivative_a(i, j, x)-negx);
			sum+=featurevector[p]*featurevector[p++];
		}

		for (j=0; j<neg->get_M(); j++)
		{
			featurevector[p++]= - exp(neg->model_derivative_b(i, j, x)-negx);
			sum+=featurevector[p]*featurevector[p++];
		}
	}

	sum=sqrt(sum);
	for (p=0; p<1+pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M()); p++)
		featurevector[p]/=sum;
