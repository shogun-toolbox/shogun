#!/usr/bin/python

import os
from parse import parse
from translate import translate

def translateExamples():
	for f in os.listdir("examples/"):
		ast = parse("examples/"+f)
		python = translate(ast, target="python")
		java = translate(ast, target="java")
		octave = translate(ast, target="octave")

		basename = f[:-len(".sg")]
		javaDir = "outputs/java/"
		pythonDir = "outputs/python/"
		octaveDir = "outputs/octave/"

		for d in [javaDir, pythonDir, octaveDir]:
			if not os.path.exists(d):
				os.makedirs(d)

		with open(javaDir + basename + ".java", "w") as nf:
			nf.write(java)

		with open(pythonDir + basename + ".py", "w") as nf:
			nf.write(python)

		with open(octaveDir + basename + ".m", "w") as nf:
			nf.write(octave)
		

if __name__ == "__main__":
	translateExamples()