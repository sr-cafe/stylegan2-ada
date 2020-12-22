from PIL import Image
from .Latent import Latent
from .ImageUtils import ImageUtils

class GeneratedImage:
	@staticmethod
	def to_image_list(images, size=None):
		return list(map(lambda image: image.as_image(size), images))

	@staticmethod
	def to_grid(images, thumb_size=ImageUtils.img_size, columns=None):
		return ImageUtils.to_grid(
			GeneratedImage.to_image_list(images),
			thumb_size,
			columns
		)

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
