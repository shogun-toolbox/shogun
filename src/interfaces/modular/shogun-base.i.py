"""
Copyright (c) 2016, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
 3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Written (W) 2016 Sanuj Sharma
"""

def entry(templateFile, class_list_file):
    templateLoader = jinja2.FileSystemLoader(searchpath="/")
    templateEnv = jinja2.Environment(loader=templateLoader)

    template = templateEnv.get_template(templateFile)

    classes = []
    with open(class_list_file) as f:
        for line in f:
            if not line[0] in {'\n', '\\'}:
                temp = [elt.strip() for elt in line.split(',')]
                classes.append(temp)

    auto_generated_msg = ("/*\n"
        " * THIS IS A GENERATED FILE!  DO NOT CHANGE THIS FILE!  CHANGE THE\n"
        " * CORRESPONDING TEMPLATE FILE, PLEASE!\n"
        " */;")

    templateVars = {'classes' : classes, 'auto_generated_msg' : auto_generated_msg}

    return template.render(templateVars)


# Usage: ./shogun-base.i.py <template file> <output file name> <class list file>
if __name__ == '__main__':
    import jinja2, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("template", help="template file that needs to be processed")
    parser.add_argument("output", help="name with path of output file")
    parser.add_argument("clist", help="name with path of class list file")
    args = parser.parse_args()
    outputText = entry(args.template, args.clist)
    f = open(args.output, 'w')
    f.writelines(outputText)
    f.close()
