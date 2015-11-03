class Complex:

	f = 0.
	g = 1.

	def __init__(self,real=1.,imag=2.):

		self.r = real
		self.i = imag

	def prin(self):

		print self.r

class deuxieme(Complex):

	def multiply(self):

		self.r = 2*self.r


