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

	def __init__(self, network_loader=None, truncation_psi=0.5):
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

	def set_network(self, network):
		self._G, self._D, self.Gs = network
		return self

	def save_network(self, filepath):
		with open(filepath, 'wb') as f:
			pickle.dump((self._G, self._D, self._Gs), f)
		return self

	def save(self, filepath):
		with open(filepath, 'wb') as f:
			pickle.dump(self, f)
		return self

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

	def from_seed(self, seed, truncation_psi=None, class_idx=None):
		z, rnd = StyleGanWrapper.expand_seed(seed, *self.Gs.input_shape[1:]) # [minibatch, component]

		latent = Latent(z, truncation_psi, seed, 'z')

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
		return self.from_seed(latent.seed, latent.truncation_psi)

	def output_shape(self):
		return self.Gs.output_shape
