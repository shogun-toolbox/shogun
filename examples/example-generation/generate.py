#!/usr/bin/python

import os
from parse import parse
from translate import translate
import json

"""
Directory to put all translations in.
E.g. if outputDir = "outputs" then translations are put in 
"outputs/<target_language>/<example_file>"
for each target language and example file
"""
outputDir = "outputs"
targetDictionariesDir = "targets"

def translateExamples():
    # Load all target dictionaries
    targets = []
    for target in os.listdir(targetDictionariesDir):
        with open(os.path.join(targetDictionariesDir, target)) as tFile:
            targets.append(json.load(tFile))

    for f in os.listdir("examples/"):
        # Parse the example file
        ast = parse(os.path.join("examples",f))
        basename = f[:-len(".sg")]

        # Translate ast to each target language
        for target in targets:
            translation = translate(ast, targetDict=target)
            directory = os.path.join(outputDir, target["OutputDirectoryName"])
            extension = target["FileExtension"]
            
            # Create directory if it does not exist
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Write translation
            with open(os.path.join(directory, basename + extension), "w") as nf:
                nf.write(translation)

if __name__ == "__main__":
    translateExamples()