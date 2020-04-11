from matplotlib import rc
from matplotlib.pyplot import rcParams

# sudo apt install dvipng texlive-latex-extra texlive-fonts-recommended
rc('text', usetex=True)
rc('text.latex', unicode=True)
rc('text.latex', preamble='\\usepackage{mathpazo}')
rc('font', family='Palatino')
rcParams.update({'font.size': 8})
rc('figure', figsize=(6, 4))
