import math
import moviepy
from PIL import Image

class ImageUtils:

	img_size=512

	@staticmethod
	def to_grid(images, thumb_size=ImageUtils.img_size, columns=None):
		if columns is None:
			columns = math.floor(math.sqrt(len(images)))

		rows = math.ceil(len(images) / columns)

		image = Image.new('RGB', (columns * thumb_size, rows * thumb_size))

		y = 0

		for i, thumb in enumerate(images):
			x = (i % columns)
			coords = (x * thumb_size, y * thumb_size)

		image.paste(thumb.resize((thumb_size, thumb_size)), coords)

		if x == columns - 1:
			y += 1

		return image

	@staticmethod
	def to_video(images, output_path, size=ImageUtils.img_size, fps=24):
		clips = []

		video = moviepy.editor.ImageSequenceClip(images, fps=fps)
		video.write_videofile(output_path)
