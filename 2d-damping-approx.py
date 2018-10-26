# curvefit with non linear least squares (curve_fit function)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg
from scipy.optimize import curve_fit
from matplotlib import cm

file_name = 'damp_values_yamaha_2str.csv'

res_arr = np.loadtxt(file_name, delimiter=';')

fret_num = res_arr[:, 0]
omega0 = res_arr[:, 2]
omega_rad = omega0 * 2. * np.pi
A = res_arr[:, 3]
phi = res_arr[:, 4]
beta = res_arr[:, 5]
delta = res_arr[:, 6]
Amax = res_arr[:, 7]
zeta = res_arr[:, 8]
m = res_arr[:, 9]
yoshida = res_arr[:, 10]
DFT1 = res_arr[:, 11]

avg_damp = np.mean([yoshida, m, beta, DFT1], axis=0)
avg_damp1 = np.median([yoshida, m, beta, DFT1], axis=0)

# x_data = np.concatenate((omega0, omega0, omega0, omega0))
# y_data = np.concatenate((fret_num, fret_num, fret_num, fret_num))
# z_data = np.concatenate((yoshida, m, beta, DFT1))

x_data = omega0
y_data = fret_num
z_data = avg_damp1

for i in np.unique(fret_num):
    cond = fret_num == i
    #    plt.plot(omega0[cond], avg_damp[cond], "k--", label="avg")
    plt.plot(omega0[cond], avg_damp1[cond], "C{0}".format(int(i)), label="avg_med")
#    plt.plot(omega0[cond], beta[cond], 'C0', label='Half-power bandwidth')
#    plt.plot(omega0[cond], m[cond], 'C1', label='Hilbert transform')
#    plt.plot(omega0[cond], yoshida[cond], 'C2', label='Yoshida')
#    plt.plot(omega0[cond], DFT1[cond], 'C3', label='IpDFT')
#    plt.show()
plt.show()
plt.close()

data = np.c_[x_data, y_data, z_data]

# data = np.zeros((3, 3))
# data[:, 0] = np.array([0., 1., 2.])
# data[:, 1] = np.array([3., 4., 5.])
# data[:, 2] = np.array([6., 7., 8.])

fig = plt.figure()
ax = fig.gca(projection='3d')

for i in np.unique(fret_num):
    cond = fret_num == i
    #    ax.plot(fret_num[cond], omega0[cond], beta[cond], 'C0', label='Half-power bandwidth')
    #    ax.plot(fret_num[cond], omega0[cond], m[cond], 'C1', label='Hilbert transform')
    #    ax.plot(fret_num[cond], omega0[cond], yoshida[cond], 'C2', label='Yoshida')
    #    ax.plot(fret_num[cond], omega0[cond], DFT1[cond], 'C3', label='IpDFT')
    ax.plot(fret_num[cond], omega0[cond], avg_damp1[cond], "C4", label="avg_med")

plt.show()
plt.close()

# some 3-dim points
# mean = np.array([0.0, 0.0, 0.0])
# cov = np.array([[1.0, -0.5, 0.8], [-0.5, 1.1, 0.0], [0.8, 0.0, 1.0]])
# data = np.random.multivariate_normal(mean, cov, 50)


# print data

# regular grid covering the domain of the data
X, Y = np.meshgrid(np.linspace(data[:, 0].min(), data[:, 0].max(), 50, endpoint=True),
                   np.linspace(data[:, 1].min(), data[:, 1].max(), 50, endpoint=True))
XX = X.flatten()
YY = Y.flatten()

order = 99  # 1: linear, 2: quadratic
if order == 1:
    # best-fit linear plane
    A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
    C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients

    # evaluate it on grid
    Z = C[0] * X + C[1] * Y + C[2]

    # or expressed using matrix/vector product
    # Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

if order == 101:
    A = np.zeros((x_data.size, 7), dtype=float)

    A[:, 0: 3] = x_data[:, np.newaxis] ** [0, 0.5, 1]
    A[:, 3: 6] = A[:, 0: 3] * y_data[:, np.newaxis] ** [1, 2, 3]
    A[:, 6] = y_data
    C, res, rnk, _ = scipy.linalg.lstsq(A, z_data)

    Z = C[0] + C[1] * X ** 0.5 + C[2] * X + C[3] * Y + C[4] * Y ** 2 + C[5] * Y ** 3

    print
    print C, res, rnk

if order == 100:
    A = np.zeros((x_data.size, 6), dtype=float)

    A[: 0, 3] = x_data[:, np.newaxis, np.newaxis] ** [0, 0.5, 1]
    A[:, 3:6] = A[:, 0:3]

    A[:, 3] *= y_data
    A[:, 4] *= y_data
    A[:, 5] *= y_data

    C, res, rnk, _ = scipy.linalg.lstsq(A, z_data)

    Z = C[0] + C[1] * X ** 0.5 + C[2] * X + C[3] * Y + C[4] * X ** 0.5 * Y + C[5] * X * Y

    print
    print C, res, rnk

if order == 999:
    # Define problem
    A = np.zeros((x_data.size, 8), dtype=float)

    A[:, 0] = x_data ** 0
    A[:, 1] = x_data ** 0.5
    A[:, 2] = x_data ** 1
    A[:, 3] = x_data ** 3

    A[:, 4:8] = A[:, 0:4]
    A[:, 4] *= y_data
    A[:, 5] *= y_data
    A[:, 6] *= y_data
    A[:, 7] *= y_data

    b = z_data

    # Use nnls to get initial guess
    x0, rnorm = scipy.optimize.nnls(A, b)


    # Define minimisation function
    def fn(x, A, b):
        return np.linalg.norm(A.dot(x) - b)


    # Define constraints and bounds
    constr_f = lambda x: np.abs(x[4] * x[1] - x[0] * x[5]) + np.abs(x[5] * x[2] - x[1] * x[6]) + np.abs(
        x[6] * x[3] - x[2] * x[7]) + np.abs(x[3] * x[4] - x[0] * x[7]) + np.abs(x[2] * x[4] - x[0] * x[6]) + np.abs(
        x[3] * x[5] - x[1] * x[7])

    cons = {'type': 'eq', 'fun': constr_f}
    # bounds = [[0., None], [0., None], [0., None], [0., None], [None, None], [None, None], [None, None], [None, None]]
    bounds = None
    const = None
    # Call minimisation subject to these values
    minout = scipy.optimize.minimize(fn, x0, args=(A, b), method='SLSQP', bounds=bounds, constraints=cons)
    C = minout.x

    print(C, constr_f(C), fn(C, A, b))

    print C[4] / C[0], C[5] / C[1], C[6] / C[2], C[7] / C[3]

    func = lambda x, y: C[0] + C[1] * x ** 0.5 + C[2] * x + C[3] * x ** 3 + C[4] * y + C[5] * x ** 0.5 * y + C[
        6] * x * y + C[7] * x ** 3 * y

    cc = C[4:8] / C[0:4]
    cc[~ np.isfinite(cc)] = 0
    ccc = np.sum(cc) / np.count_nonzero(cc)
    print cc, ccc

    func1 = lambda x, y: (C[0] + C[1] * x ** 0.5 + C[2] * x + C[3] * x ** 3) * (1. + ccc * y)
    Z = func1(X, Y)

if order == 99:
    A = np.zeros((x_data.size, 8), dtype=float)

    A[:, 0] = x_data ** 0
    A[:, 1] = x_data ** 0.5
    A[:, 2] = x_data ** 1
    A[:, 3] = x_data ** 3

    A[:, 4:8] = A[:, 0:4]
    A[:, 4] *= y_data
    A[:, 5] *= y_data
    A[:, 6] *= y_data
    A[:, 7] *= y_data

    from symfit import parameters, variables, Fit, Model, Ge
    from symfit.core import minimizers
    from symfit.core.objectives import LogLikelihood

    a0, a1, a2, a3 = parameters('a0, a1, a2, a3')
    a4, a5, a6, a7 = parameters('a4, a5, a6, a7')

    a0.min = 0.
    a1.min = 0.
    a2.min = 0.
    a3.min = 0.
    # a4.min = 0.
    # a5.min = 0.

    a0.value = 0.3390921000e-1
    a1.value = 0.5422763222e-2
    a2.value = 0.5555555556e-4
    a3.value = 4.166064907e-13

    x, y, z = variables('x, y, z')
    # model = {z: a0 + a1 * x ** 0.5 + a2 * x + a3 * x ** 3 + a4 * y + a5 * x ** 0.5 * y + a6 * x * y + a7 * x ** 3 * y}
    # model = {z: a0 + a1 * x ** 0.5 + a2 * x + a3 * x ** 3 + a4 * y + a5 * x ** 0.5 * y + a6 * x * y + a7 * x ** 3 * y}
    # model = {z: (a0 + a1 * x ** 0.5 + a2 * x + a3 * x ** 3.)*(1 + a4*y + a5*y**2)}
    model = {z: (a0 + a1 * x ** 0.5 + a2 * x + a3 * x ** 3.) * (1. + a4 * y)}

    fit = Fit(model, x=x_data, y=y_data, z=z_data)
    fit_result = fit.execute()

    # print fit_result

    import scipy.optimize

    C, res = scipy.optimize.nnls(A, z_data)
    # C, res, rnk, _ = scipy.linalg.lstsq(A, z_data)

    #    Z = C[0] + C[1] * X ** 0.5 + C[2] * X + C[3] * X ** 3 + C[4] * Y + C[5] * X ** 0.5 * Y + C[6] * X * Y + C[
    #        7] * X ** 3 * Y

    func = lambda x, y: C[0] + C[1] * x ** 0.5 + C[2] * x + C[3] * x ** 3 + C[4] * y + C[5] * x ** 0.5 * y + C[
        6] * x * y + C[7] * x ** 3 * y

    for i in range(C.size):
        print C[i], '\t',
    print

    cc = C[4:8] / C[0:4]
    cc[~ np.isfinite(cc)] = 0
    ccc = np.sum(cc)
    print cc, ccc

    func1 = lambda x, y: (C[0] + C[1] * x ** 0.5 + C[2] * x + C[3] * x ** 3) * (1. + ccc * y)

    Z = func(X, Y)

    print
    print C, res

if order == 98:

    A = np.zeros((x_data.size, 3 * 4), dtype=float)

    A[:, 0] = x_data ** 0
    A[:, 1] = x_data ** 0.5
    A[:, 2] = x_data ** 1
    A[:, 3] = x_data ** 3

    A[:, 4:8] = A[:, 0:4]
    A[:, 8:12] = A[:, 0:4]

    A[:, 4] *= y_data
    A[:, 5] *= y_data
    A[:, 6] *= y_data
    A[:, 7] *= y_data

    A[:, 8] *= y_data ** 3
    A[:, 9] *= y_data ** 3
    A[:, 10] *= y_data ** 3
    A[:, 11] *= y_data ** 3

    C, res, rnk, _ = scipy.linalg.lstsq(A, z_data)

    # C, res = scipy.optimize.nnls(A, z_data)

    Z = C[0] + C[1] * X ** 0.5 + C[2] * X + C[3] * X ** 3 + C[4] * Y + C[5] * X ** 0.5 * Y + C[6] * X * Y + C[
        7] * X ** 3 * Y + C[8] * Y ** 3 + C[9] * X ** 0.5 * Y ** 3 + C[10] * X * Y ** 3 + C[11] * X ** 3 * Y ** 3

    func = lambda X, Y: C[0] + C[1] * X ** 0.5 + C[2] * X + C[3] * X ** 3 + C[4] * Y + C[5] * X ** 0.5 * Y + C[
        6] * X * Y + C[
                            7] * X ** 3 * Y + C[8] * Y ** 3 + C[9] * X ** 0.5 * Y ** 3 + C[10] * X * Y ** 3 + C[
                            11] * X ** 3 * Y ** 3

    print
    print C, res, rnk



elif order == 2:
    # best-fit quadratic curve
    A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
    C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])

    # evaluate it on a grid
    Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2], C).reshape(X.shape)

    print C

fig = plt.figure()
ax = fig.gca(projection='3d')
# plot points and fitted surface
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.5)
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', s=50)
plt.xlabel(r'$\omega$, Hz')
plt.ylabel(r'Fret Number')
ax.set_zlabel(r'$\varepsilon$, 1/s')
cset = ax.contourf(X, Y, Z, zdir='y', offset=y_data.max())
cset = ax.contourf(X, Y, Z, zdir='x', offset=x_data.min())
ax.axis('equal')
ax.axis('tight')
plt.tight_layout()
plt.savefig(file_name + '.png', dpi=300)
plt.show()


freq = np.linspace(55., 500., 100)

for fn in [1, 3, 7, 12]:
    #    d2 = fit.model(x=freq, y=fn, **fit_result.params)[0]

    # plt.plot(freq, d2, 'k--')

    damp = func(freq, fn)
    damp1 = func1(freq, fn)
    cond = fret_num == fn

    # plt.plot(omega0[cond], beta[cond])
    # plt.plot(omega0[cond], m[cond])
    # plt.plot(omega0[cond], yoshida[cond])
    # plt.plot(omega0[cond], DFT1[cond])
    plt.plot(omega0[cond], avg_damp1[cond])

    plt.plot(freq, damp, '--')
    plt.plot(freq, damp1, '--')
    plt.show()

# plt.pcolormesh(X, Y, Z)
# plt.colorbar()
# plt.show()
