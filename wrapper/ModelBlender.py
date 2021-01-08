import math
# import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib import tfutil

from .StyleGanWrapper import StyleGanWrapper

class ModelBlender:
	@staticmethod
	def extract_conv_names(model):
		# layers are G_synthesis/{res}x{res}/...
		# make a list of (name, resolution, level, position)
		# Currently assuming square

		model_names = list(model.trainables.keys())
		conv_names = []

		resolutions =  [4*2**x for x in range(9)]

		level_names = [["Conv0_up", "Const"],
						["Conv1", "ToRGB"]]

		position = 0

		for res in resolutions:
			root_name = f"G_synthesis/{res}x{res}/"
			for level, level_suffixes in enumerate(level_names):
				for suffix in level_suffixes:
					search_name = root_name + suffix
					matched_names = [x for x in model_names if x.startswith(search_name)]
					to_add = [(name, f"{res}x{res}", level, position) for name in matched_names]
					conv_names.extend(to_add)
				position += 1

		return conv_names

	@staticmethod
	def blend_models(low_res_wrapper, high_res_wrapper, resolution, level, blend=0):
		low_res_model = low_res_wrapper.Gs
		high_res_model = high_res_wrapper.Gs

		result_model = low_res_model.clone()

		resolution = f'{resolution}x{resolution}'

		low_res_names = ModelBlender.extract_conv_names(low_res_model)
		high_res_names = ModelBlender.extract_conv_names(high_res_model)

		short_names = [(x[1:3]) for x in low_res_names]
		full_names = [(x[0]) for x in low_res_names]
		mid_point_idx = short_names.index((resolution, level))
		mid_point_pos = low_res_names[mid_point_idx][3]

		ys = []
		for name, resolution, level, position in low_res_names:
			x = position - mid_point_pos

			exponent = -x/blend
			y = 1 / (1 + math.exp(exponent))
			ys.append(y)

		tfutil.set_vars(
			tfutil.run(
				{result_model.vars[name]: (high_res_model.vars[name] * y + low_res_model.vars[name] * (1-y))
				for name, y
				in zip(full_names, ys)}
			)
		)

		return StyleGanWrapper().set_network((low_res_wrapper._G, low_res_wrapper._D, result_model))
