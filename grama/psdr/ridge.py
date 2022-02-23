"""Ridge function approximation from function values"""
# (c) 2017 Jeffrey M. Hokanson (jeffrey@hokanson.us)

import numpy as np

from .function import BaseFunction
from .subspace import SubspaceBasedDimensionReduction
from .pgf import PGF

class RidgeFunction(BaseFunction, SubspaceBasedDimensionReduction):
	@property
	def U(self):
		return self._U

	def shadow_plot(self, X = None, fX = None, dim = None, ax = 'auto', pgfname = None, **kwargs):
		if dim is None:
			dim = self.U.shape[1]
		else:
			assert dim == self.U.shape[1]

		ax = SubspaceBasedDimensionReduction.shadow_plot(self, X = X, fX = fX, dim = dim, ax = ax, pgfname = pgfname)

		# Draw the response surface
		if dim == 1:
			Y = np.dot(self.U.T, X.T).T
			lb = np.min(Y)
			ub = np.max(Y)
			
			xx = np.linspace(lb, ub, 500)
			Uxx = np.hstack([self.U*xxi for xxi in xx]).T
			yy = self.eval(Uxx)

			if ax is not None:
				ax.plot(xx, yy, 'r-')

			if pgfname is not None:	
				pgfname2 = pgfname[:pgfname.rfind('.')] + '_response' + pgfname[pgfname.rfind('.'):]
				pgf = PGF()
				pgf.add('x', xx)
				pgf.add('fx', yy )
				pgf.write(pgfname2)
					

		elif dim == 2:	
			Y = np.dot(self.U.T, X.T).T
			lb0 = np.min(Y[:,0])
			ub0 = np.max(Y[:,0])

			lb1 = np.min(Y[:,1])
			ub1 = np.max(Y[:,1])

			# Constuct mesh on the domain
			xx0 = np.linspace(lb0, ub0, 50)
			xx1 = np.linspace(lb1, ub1, 50)
			XX0, XX1 = np.meshgrid(xx0, xx1)
			UXX = np.vstack([XX0.flatten(), XX1.flatten()])
			XX = np.dot(self.U, UXX).T
			YY = self.eval(XX).reshape(XX0.shape)
			
			ax.contour(xx0, xx1, YY, 
				levels = np.linspace(np.min(fX), np.max(fX), 20), 
				vmin = np.min(fX), vmax = np.max(fX),
				linewidths = 0.5)
		

		else: 
			raise NotImplementedError("Cannot draw shadow plots in more than two dimensions")	

		return ax		


