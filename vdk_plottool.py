# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

# plt.style.use('vodka')

txtdata1 = """
0	3.76E-04
1	3.66E-04
2	3.95E-04
3	3.95E-04
4	5.91E-04
5	5.30E-04
6	7.05E-04
7	6.60E-04
8	5.67E-04
9	4.48E-04
10	6.72E-04
11	4.35E-04
12	8.67E-04

"""

txtdata2 = """
0	3.53E-04
1	4.85E-04
3	4.72E-04
5	5.74E-04
7	7.66E-04
12	9.15E-04

"""

txtdata3 = """
0	2.72E-04
1	5.66E-04
2	5.46E-04
3	6.03E-04
4	3.68E-04

"""

d_data1 = """
0	3.52E-04
1	5.62E-04
2	5.73E-04
3	4.23E-04
4	4.08E-04
5	5.33E-04
6	4.64E-04
7	6.63E-04
8	4.90E-04
9	4.42E-04
10	4.55E-04
11	6.67E-04
12	8.06E-04
"""

d_data2 = """
0	5.55E-04
1	6.58E-04
3	6.80E-04
5	7.47E-04
7	7.28E-04
12	9.34E-04
"""
d_data3 = """
0	4.95E-04
1	4.59E-04
2	3.99E-04
3	4.12E-04
4	5.68E-04
"""

a_data1 = """
0	1.02E-03
1	1.16E-03
2	6.24E-04
3	7.75E-04
4	6.05E-04
5	6.50E-04
6	9.24E-04
7	5.63E-04
8	6.78E-04
9	1.16E-03
10	5.37E-04
11	5.86E-04
12	1.04E-03
"""

a_data2 = """
0	1.16E-03
1	1.24E-03
3	1.37E-03
5	1.29E-03
7	1.43E-03
12	1.71E-03
"""
a_data3 = """
0	2.98E-04
1	3.59E-04
2	4.48E-04
3	1.35E-03
4	7.12E-04
"""

e_data1 = """
0	9.18E-04
1	1.42E-03
2	1.05E-03
3	7.99E-04
4	1.27E-03
5	1.14E-03
6	9.96E-04
7	1.05E-03
8	8.01E-04
9	5.97E-04
10	5.92E-04
11	8.64E-04
12	8.65E-04
"""

e_data2 = """
0	2.18E-03
1	1.96E-03
3	1.11E-03
5	1.38E-03
7	1.03E-03
12	9.81E-04
"""
e_data3 = """
0	8.68E-04
1	6.05E-04
2	3.86E-04
3	4.31E-04
4	6.39E-04
"""


def tofile(fname, data):
    f = open(fname, 'w')
    f.write(data)
    f.close()
    return np.loadtxt(fname, delimiter='\t')


#res1 = tofile('tmpfile1.csv', txtdata1)
#res2 = tofile('tmpfile2.csv', txtdata2)
#res3 = tofile('tmpfile3.csv', txtdata3)

#res1 = tofile('tmpfile1.csv', d_data1)
#res2 = tofile('tmpfile2.csv', d_data2)
#res3 = tofile('tmpfile3.csv', d_data3)

#res1 = tofile('tmpfile1.csv', a_data1)
#res2 = tofile('tmpfile2.csv', a_data2)
#res3 = tofile('tmpfile3.csv', a_data3)

res1 = tofile('tmpfile1.csv', e_data1)
res2 = tofile('tmpfile2.csv', e_data2)
res3 = tofile('tmpfile3.csv', e_data3)

p1 = np.polyfit(res1[:, 0], res1[:, 1], deg=1)
aprx1 = np.poly1d(p1)
p2 = np.polyfit(res2[:, 0], res2[:, 1], deg=1)
aprx2 = np.poly1d(p2)
p3 = np.polyfit(res3[:, 0], res3[:, 1], deg=1)
aprx3 = np.poly1d(p3)

fit1 = '$\\overline{{\\varepsilon}} = ({0:5.3g}m+{1:5.3g})10^{{-4}}$'.format(p1[0] * 10000, p1[1] * 10000)
fit2 = '$\\overline{{\\varepsilon}} = ({0:5.3g}m+{1:5.3g})10^{{-4}}$'.format(p2[0] * 10000, p2[1] * 10000)
fit3 = '$\\overline{{\\varepsilon}} = ({0:5.3g}m+{1:5.3g})10^{{-4}}$'.format(p3[0] * 10000, p3[1] * 10000)

plt.plot(res1[:, 0], res1[:, 1], 'C0-', label='Ibanez RB 630')
plt.plot(res1[:, 0], aprx1(res1[:, 0]), 'C0--', label='Ibanez RB 630 fitting ' + fit1, lw=1)
plt.plot(res2[:, 0], res2[:, 1], 'C1-', label='Cort C4H')
plt.plot(res2[:, 0], aprx2(res2[:, 0]), 'C1:', label='Cort C4H fitting ' + fit2, lw=1)
plt.plot(res3[:, 0], res3[:, 1], 'C2-', label='Yamaha')
plt.plot(res2[:, 0], aprx3(res2[:, 0]), 'C2-.', label='Yamaha fitting ' + fit3, lw=1)
plt.title('E-string')
plt.xlabel('m, fret number')
plt.ylabel('$\\overline{\\varepsilon}$')
plt.xlim(0, 12)
plt.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
plt.grid()
plt.tight_layout()
plt.legend()

plt.savefig('E_damp.png', dpi=300 )
plt.savefig('E_damp.eps')

plt.show()
