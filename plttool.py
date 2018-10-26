import numpy as np
import matplotlib.pyplot as plt

plt.style.use('vodka')

l = 1.83

for i in range(10):
    print np.sin(np.pi*i)

if False:
    l0 = np.linspace(1e-3, l-1e-4, 100, endpoint=True)

    for n in range(1, 6):
        an = -2 * l * (np.sin(np.pi * n) * l0 - np.sin(l0 * np.pi * n / l) * l) / (l0 * np.pi ** 2 * n ** 2 * (l - l0))
        plt.plot(l0/l, an, label="$k = " + str(n)+'$')

    plt.legend(loc='upper right', shadow=False)
    plt.xlabel(u'$l_0/l$')
    plt.ylabel(u'$a_k$')
    plt.savefig('fig1.png')
    plt.show()




