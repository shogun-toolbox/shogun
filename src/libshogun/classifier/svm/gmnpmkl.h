/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Alexander Binder
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef GMNPMKL_H_
#define GMNPMKL_H_

#include <vector>

#include "base/SGObject.h"
#include "lib/ShogunException.h"
#include "classifier/svm/GMNPSVM.h"
#include "kernel/Kernel.h" 

//for the testing method
#include "kernel/CustomKernel.h" 
#include "kernel/CombinedKernel.h" 
#include "features/DummyFeatures.h"


#ifdef USE_GLPK
#include <glpk.h>

/** CGMNPMKL is a class for a L1-norm (because there is with glpk a free solver
 * available) MKL for the multiclass svm CGMNPSVM kit is to be used as all
 * other SVM routines with the set_kernel, set_C, set_labels, set_epsilon
 * its own parameters are thresh (L2 norm of subkernel weights for termination) and
 * maxiters (how many silp iterations at most in order to force termination)
 *
 * TODO: check what options to pass from CGMNPMKL to CGMNPSVM
 * set C_mkl?
 * TODO: clear types (float64_t, size_t, int)
 */
class lpwrapper
{
public:
	int32_t lpwrappertype; // 0 -glpk

	lpwrapper();
	
	virtual ~lpwrapper();
	
	virtual void setup(const int32_t numkernels); 

	/** takes a set of alpha^t H alpha and -sum alpha
	 * and adds constraint32_t to the working set
	 * theta <= \beta^ (1) + -sumalpha  
	 */
	virtual void addconstraint(const ::std::vector<float64_t> & normw2,
			const float64_t sumofpositivealphas); 

	virtual void computeweights(std::vector<float64_t> & weights2); 
	
};

class glpkwrapper: public lpwrapper
{
public:
	
	glpkwrapper();
	virtual ~glpkwrapper();
	

	
protected:
	
	///prohibit copying, copy constructor by declaring protected?
	glpkwrapper operator=(glpkwrapper & gl);
	glpkwrapper(glpkwrapper & gl);
	glp_prob* linearproblem;
};

#else

class lpwrapper : public CSGObject
{
public:
	int32_t lpwrappertype; // 0 -glpk

	lpwrapper();
	
	virtual ~lpwrapper();
	
	virtual void setup(const int32_t numkernels); 

	// takes a set of alpha^t H alpha and -sum alpha and adds constraint32_t to the working set theta <= \beta^ (1) + -sumalpha  
	virtual void addconstraint(const ::std::vector<float64_t> & normw2,
			const float64_t sumofpositivealphas); 

	virtual void computeweights(std::vector<float64_t> & weights2); 
	
};

class glpkwrapper: public lpwrapper
{
public:
	
	glpkwrapper();
	virtual ~glpkwrapper();
	/** @return object name */
	inline virtual const char* get_name() const { return "GLPKWrapper"; }
	

	
protected:
	
	//prohibit copying, copy constructor by declaring protected?
	glpkwrapper operator=(glpkwrapper & gl);
	glpkwrapper(glpkwrapper & gl);
};

#endif //USE_GLPK

class glpkwrapper4CGMNPMKL: public glpkwrapper
{
public:
	int32_t numkernels;

	glpkwrapper4CGMNPMKL();
	virtual ~glpkwrapper4CGMNPMKL();
	
	
	virtual void setup(const int32_t numkernels2); 

	// takes a set of alpha^t H alpha and -sum alpha and adds constraint32_t to the working set theta <= \beta^ (1) + -sumalpha  
	virtual void addconstraint(const ::std::vector<float64_t> & normw2,
			const float64_t sumofpositivealphas); 

	virtual void computeweights(std::vector<float64_t> & weights2); 

	/** @return object name */
	inline virtual const char* get_name() const { return "GLPKWrapper4GMNPMKL"; }
	
};	

class CGMNPMKL : public CMultiClassSVM
{
public:
	CGMNPMKL();
	CGMNPMKL(float64_t C, CKernel* k, CLabels* lab);
	
	virtual ~CGMNPMKL();

	virtual bool train();

	/** get classifier type
	 *
	 * @return classifier type GMNPMKL
	 */
	virtual inline EClassifierType get_classifier_type() { return CT_GMNPMKL; }
	
	//returns the subkernelweights or NULL if none such have been computed, caller has to delete the returned pointer
	float64_t* getsubkernelweights(int32_t & numweights);
	
	float64_t thresh; // at what l2 norm of sub kernel weights change to quit
	int32_t maxiters; // how many iters of silp at max or <0 to ignore this
	int32_t lpwrappertype; // what kind of LP solver: 0 - glpk (default)


	
protected:
	
	//inits LP
	void lpsetup(const int32_t numkernels);
	//sets labels for the svm, creates it
	void initsvm();
	
	//uses implicitly Ckernel* kernel
	//inits this class
	void init();
	
	
	virtual bool evaluatefinishcriterion(const int32_t numberofsilpiterations);
	
	CGMNPSVM * svm; //the svm in the silp training
	
	lpwrapper* lpw; // the lp solver wrapper
	
	void addingweightsstep( const std::vector<float64_t> & curweights);
	//the following two is the actual know how in this class :)
	// uses the svm
	float64_t getsumofsignfreealphas();
	// uses the svm and Ckernel * kernel
	float64_t getsquarenormofprimalcoefficients(
			const int32_t ind);
	
	
	//numberofsilpiterations
	
	::std::vector< std::vector< float64_t> > weightshistory;
	
	
	int32_t numdat,numcl,numker;
	
};

#endif // GMNPMKL_H_
