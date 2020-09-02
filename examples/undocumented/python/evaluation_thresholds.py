#!/usr/bin/env python
parameter_list = [[1000]]

def evaluation_thresholds (index):
	from shogun import BinaryLabels
	import shogun as sg
	import numpy as np

	np.random.seed(17)
	output=np.arange(-1,1,0.001)
	output=(0.3*output+0.7*(np.random.rand(len(output))-0.5))
	label=[-1.0]*(len(output)//2)
	label.extend([1.0]*(len(output)//2))
	label=np.array(label)

	pred=BinaryLabels(output)
	truth=BinaryLabels(label)

	evaluator=sg.create("ROCEvaluation")
	evaluator.evaluate(pred, truth)

	[fp,tp]=evaluator.get("ROC")

	thresh=evaluator.get("thresholds")
	b=thresh[index]

	#print("tpr", np.mean(output[label>0]>b), tp[index])
	#print("fpr", np.mean(output[label<0]>b), fp[index])

	return tp[index],fp[index],np.mean(output[label>0]>b),np.mean(output[label<0]>b)

if __name__=='__main__':
	print('Evaluation with Thresholds')
	evaluation_thresholds(*parameter_list[0])
