/* -*-C++-*- */

/* $Id$ */

#ifndef __OPTIMIZER_H_
#define __OPTIMIZER_H_

class COptimizer
{
public:
  COptimizer(void) { }
  virtual ~COptimizer(void) { }

  virtual double bound(void) const = 0;
  virtual double margin(void) const = 0;

  virtual int verbose(void) const = 0;
  virtual unsigned maxiter(void) const = 0;
  virtual unsigned sigfigs(void) const = 0;
};

class CIntPointPR : public COptimizer
{
public:
  CIntPointPR(void) : COptimizer(), m_Bound(10.0), m_Margin(0.05),
		      m_Verbose(2), m_MaxIter(50), m_SigFigs(7) { }
  virtual ~CIntPointPR(void) { }

  virtual double bound(void) const { return (m_Bound); }
  virtual double margin(void) const { return (m_Margin); }

  virtual int verbose(void) const { return (m_Verbose); }
  virtual unsigned maxiter(void) const { return (m_MaxIter); }
  virtual unsigned sigfigs(void) const { return (m_SigFigs); }

  virtual void SetBound(const double b) { m_Bound = b; }
  virtual void SetSigFigs(const unsigned sfs) { m_SigFigs = sfs; }
  virtual void SetMaxIterations(const unsigned n) { m_MaxIter = n; }

protected:
  double m_Bound;
  double m_Margin;
  int m_Verbose;
  unsigned m_MaxIter;
  unsigned m_SigFigs;
};

#endif /* ! __OPTIMIZER_H_ */
