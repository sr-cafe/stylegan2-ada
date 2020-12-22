from PIL import Image
from .Latent import Latent

class GeneratedImage:
	def __init__(self, image, latent):
		self.image = image
		if isinstance(latent, Latent):
			self.latent = latent
		else:
			self.latent = Latent(latent)

	def as_image(self, size=None):
		image = Image.fromarray(self.image[0])

		if size is not None:
			image = image.resize(size, size)

		return image

	def as_z_vector(self):
		return self.latent.vector
