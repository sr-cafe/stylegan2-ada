import math
import moviepy.editor
from PIL import Image, ImageChops

class ImageUtils:

	img_size=(512, 512)
	thumb_size=(200, 200)

	@staticmethod
	def to_grid(images, thumb_size=thumb_size, columns=None):
		if columns is None:
			columns = math.floor(math.sqrt(len(images)))

		rows = math.ceil(len(images) / columns)

		image = Image.new('RGB', (columns * thumb_size[0], rows * thumb_size[1]))

		y = 0

		for i, thumb in enumerate(images):
			x = (i % columns)
			coords = (x * thumb_size[0], y * thumb_size[1])

			image.paste(thumb.resize(thumb_size), coords)

			if x == columns - 1:
				y += 1

		return image

	@staticmethod
	def to_video(images, output_path, size=img_size, fps=24):
		video = moviepy.editor.ImageSequenceClip(images, fps=fps)
		video.write_videofile(output_path)

	@staticmethod
	def difference(img1, img2):
		return ImageChops.difference(img2, img1)

	@staticmethod
	def to_difference_grid(original, images, thumb_size=thumb_size, columns=None):
		compos = []
		compo_size = (thumb_size[0] * 3, thumb_size[1])

		for i, thumb in enumerate(images):
			compos.append(
				ImageUtils.to_grid(
					[original, thumb, ImageUtils.difference(original, thumb)],
					thumb_size,
					3
				)
			)

		return ImageUtils.to_grid(compos, compo_size, columns)
