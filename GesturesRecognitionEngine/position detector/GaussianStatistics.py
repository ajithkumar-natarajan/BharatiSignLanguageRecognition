import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import signal
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class GaussianStatistics():
    """
    """
    def MultivariateGaussian(self, mu, sd, N = 10, A = 1):
        """
            mu: location of peak value
                array
                    ex: 2D = [5., 5.]
            sd: spread of gaussian
                covariance cross-covariance matrix
                    ex: 2D = [[20., -1.],
                               [-1., 20]]
            N: spread radius
                any real number -> scalar field
            A: Amplitude of gaussian
                any real number -> scalar field
        """

        X = np.linspace(0, N, N)
        Y = np.linspace(0, N, N)
        X, Y = np.meshgrid(X, Y)

        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        n = mu.shape[0]
        sd_det = np.linalg.det(sd)
        sd_inv = np.linalg.inv(sd)
        N = np.sqrt((2*np.pi)**n * sd_det)

        fac = np.einsum('...k,kl,...l->...', pos-mu, sd_inv, pos-mu)

        return A * np.exp(-fac / 2)

    def DOG(self, iRad, eRad, iA, eA):
        """
        """
        eSd = np.array([[eRad**2, -1.],
                        [-1., eRad**2]])
        iSd = np.array([[(iRad)**2, -1.],
                        [-1., (iRad)**2]])
        mu  = np.array([iRad, iRad])
        self.iRad = iRad
        self.eRad = eRad
        self.pos = self.MultivariateGaussian(mu, eSd, N = 2*iRad, A = eA)
        self.neg = self.MultivariateGaussian(mu, iSd, N = 2*iRad, A = iA)
        return self.pos, self.neg

    def Visualize(self, Gaussian, _type = '2d'):
        """
        """
        X = np.arange(0, Gaussian.shape[0])
        Y = np.arange(0, Gaussian.shape[1])
        X, Y = np.meshgrid(X, Y)

        if _type == '3d':
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            surf = ax.plot_surface(X, Y, Gaussian, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
            # Customize the z axis.
            ax.set_zlim(-1.01, 1.01)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.show()
        else:
            plt.imshow(Gaussian)
            plt.show()

    def OrientationBar(self, N = 10,
                    mu=np.array([5., 5.]),
                    Sigma=np.array([[20., -19.8], [-19.8, 20]]),
                    theta=0, display=False):
        """
        """
        if (theta > 90.): theta = theta - 180.
        theta = theta*np.pi/180.

        # update Sigma based om theta
        Sigma = np.array([[ Sigma[0,0]*(np.pi*0.5-theta)**2 +1.,
                            Sigma[0,1]*theta*(np.pi*0.5-theta) + 1.],
                          [ Sigma[1,0]*theta*(np.pi*0.5-theta) + 1.,
                            Sigma[1,1]*(theta)**2 +1.]])

        _bar = self.MultivariateGaussian(mu, Sigma, N)
        if display: self.Visualize(_bar)
        return _bar
