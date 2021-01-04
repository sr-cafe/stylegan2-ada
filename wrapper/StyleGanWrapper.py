import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import pickle

from .GeneratedImage import GeneratedImage
from .Latent import Latent
from .NetworkLoader import NetworkLoader

class StyleGanWrapper:
	@staticmethod
	def expand_seed(seed, z_space_dims=512):
		rnd = np.random.RandomState(seed)
		return (rnd.randn(1, z_space_dims), rnd)

	def __init__(self, network_loader, truncation_psi=0.5):
		self.network_loader = network_loader
		self.truncation_psi = truncation_psi

	def load_network(self, network_path, saved_network_name):
		tflib.init_tf()

		self._G, self._D, self.Gs = self.network_loader.load(network_path, saved_network_name)

		self.noise_vars = [var for name, var in self.Gs.components.synthesis.vars.items() if name.startswith('noise')]
		self.Gs_kwargs = {
			'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
			'randomize_noise': False
		}

		return self

	# def __generate_from_z(self, z_vector, label=None):
	# 	image = self.Gs.run(z_vector, label, **self.Gs_kwargs)
	# 	return GeneratedImage(image, z_vector, self.Gs_kwargs['truncation_psi'])

	# def __generate_from_w(self, w_vector, label=None):
	# 	image = self.Gs.components.synthesis.run(w_vector, **self.Gs_kwargs)
	# 	return GeneratedImage(image, w_vector, self.Gs_kwargs['truncation_psi'])

	def __generate(self, latent, label=None):

		if latent.truncation_psi is not None:
			truncation = latent.truncation_psi
		else:
			truncation = self.truncation_psi

		if truncation is not None:
			self.Gs_kwargs['truncation_psi'] = truncation

		if latent.type is 'z':
			image = self.Gs.run(latent.vector, label, **self.Gs_kwargs)
		else:
			image = self.Gs.components.synthesis.run(latent.vector, **self.Gs_kwargs)

		return GeneratedImage(image, latent)

	def _get_label(self, class_idx):
		label = np.zeros([1] + self.Gs.input_shapes[1][1:])
		if class_idx is not None:
			label[:, class_idx] = 1

		return label

	# def _set_truncation_psi(self, truncation_psi):
	# 	if truncation_psi is not None:
	# 		truncation = truncation_psi
	# 	else:
	# 		truncation = self.truncation_psi

	# 	if truncation is not None:
	# 		self.Gs_kwargs['truncation_psi'] = truncation

	def from_seed(self, seed, truncation_psi=None, class_idx=None):
		z, rnd = StyleGanWrapper.expand_seed(seed, *self.Gs.input_shape[1:]) # [minibatch, component]

		# self._set_truncation_psi(truncation_psi)

		latent = Latent(z, truncation_psi, 'z')

		tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in self.noise_vars}) # [height, width]
		return self.__generate(latent, self._get_label(class_idx))
		# return self.__generate_from_z(z, self._get_label(class_idx))

	def from_seeds(self, seeds, truncation_psi=None, class_idx=None):
		if not isinstance(seeds, list):
			seeds = [seeds]

		images = []

		for indx, seed in enumerate(seeds):
			images.append(self.from_seed(seed, truncation_psi, class_idx))

		return images

	def from_z_vector(self, z_vector, truncation_psi=None, class_idx=None):
		noise_rnd = np.random.RandomState(1) # fix noise

		latent = Latent(z_vector, truncation_psi, 'z')

		tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in self.noise_vars}) # [height, width]

		return self.__generate(latent, self._get_label(class_idx))

	def from_z_vectors(self, z_vectors, truncation_psi=None, class_idx=None):
		if not isinstance(z_vectors, list):
			z_vectors = [z_vectors]

		images = []

		for indx, z_vector in enumerate(z_vectors):
			images.append(self.from_z_vector(z_vector, truncation_psi, class_idx))

		return images

	def from_w_vector(self, w_vector, truncation_psi=None):
		latent = Latent(w_vector, truncation_psi, 'w')
		noise_rnd = np.random.RandomState(1) # fix noise
		tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in self.noise_vars})

		return self.__generate(latent)

	def from_latent(self, latent):
		noise_rnd = np.random.RandomState(1) # fix noise
		tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in self.noise_vars})

		return self.__generate(latent)

	def explore_neighbours(self, z_vector, radius, num_samples, truncation_psi=None):
		if isinstance(z_vector, GeneratedImage):
			z_vector = z_vector.z_vector

		images = [self.from_z_vector(z_vector, truncation_psi)]

		for indx in range(num_samples):
			random = np.random.uniform(-radius,radius,[1,512])
			z = np.clip(np.add(z_vector, random), -1, 1)
			images.append(self.from_z_vector(z, truncation_psi))

		return images

	def output_shape(self):
		return self.Gs.output_shape
