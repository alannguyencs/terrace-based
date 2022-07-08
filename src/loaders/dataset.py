from constants import *




class FoodDataset(Dataset):
	def __init__(self, ann_file, transform=None):
		self.ann_file = DATA_PATH + 'train_val/' + ann_file
		self.data_path = DATA_PATH + 'image/'
		self.mask_path = DATA_PATH + 'mask/'
		self.transform = transform
		self.dataset = list(open(self.ann_file, 'r'))
		print ("data_length:", len(self.dataset))

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		sample = self.extract_content(self.dataset[idx])
		if self.transform:
			sample = self.transform(sample)
		
		sample_data = tuple([s_data for s_data in sample.values()])
		return sample_data


	def extract_content(self, content_):
		[image_path, counting, aug_methods] = content_.strip().split(',,,')[:3]
		aug_methods = json.loads(aug_methods)['aug_flip']

		content = OrderedDict()
		content['image_path'] = image_path
		content['image'] = self.get_image(image_path, aug_methods)
		content['counting'] = int(counting)
		
		for mask_type in MASK_TYPES:
			mask_path = self.mask_path + mask_type + '/' + image_path.replace('.jpg', '.png')
			content[mask_type] = self.get_mask(mask_path, aug_methods)

		return content

	def get_image(self, image_path, aug_methods):
		image = Image.open(self.data_path + image_path).convert("RGB")
		for aug_method in aug_methods:
			image = image.transpose(Image.__getattribute__(aug_method))
		return image

	def get_mask(self, mask_path, aug_methods):
		if not os.path.isfile(mask_path):
			return Image.new('L', (SEG_OUTPUT_SIZE, SEG_OUTPUT_SIZE))

		mask = Image.open(mask_path)
		for aug_method in aug_methods:
			mask = mask.transpose(Image.__getattribute__(aug_method))
		return mask



class Rescale(object):
	def __init__(self, output_size):
		assert isinstance(output_size, tuple)
		self.output_size = output_size

	def __call__(self, sample):
		image = sample['image']
		assert isinstance(image, Image.Image)
		sample['image'] = image.resize(self.output_size, Image.ANTIALIAS)
		return sample

class Rotate(object):
	def __init__(self, is_training):
		self.is_training = is_training

	def __call__(self, sample):
		if self.is_training:
			image = sample['image']
			assert isinstance(image, Image.Image)
			rotate_degrees = [0, 90, 180, 270]
			shuffle(rotate_degrees)
			rotate_degree = rotate_degrees[0]
			if rotate_degree > 0:				
				sample['image'] = image.transpose(
					Image.__getattribute__("ROTATE_{}".format(rotate_degree)))

				for sample_key in sample.keys():
					if sample_key in MASK_TYPES:
						sample[sample_key] = sample[sample_key].transpose(
											Image.__getattribute__("ROTATE_{}".format(rotate_degree)))
		return sample


class ToNumpy(object):

	def __call__(self, sample):
		sample['image'] = np.array(sample['image'])
		sample['counting'] = np.array([sample['counting']], dtype='int64')
		for sample_key in sample.keys():
			if sample_key in MASK_TYPES:
				sample[sample_key] = np.array(sample[sample_key])


		return sample

class AddGaussianNoise(object):
	def __init__(self, is_training):
		self.is_training = is_training
		self.gaussian_mean = 0.0
		self.gaussian_std = 5.0

	def __call__(self, sample):
		if self.is_training:
			image = sample['image']
			gaussian = np.random.normal(self.gaussian_mean, self.gaussian_std, image.shape).astype(np.float64)
			sample['image'] = image + gaussian

		return sample

class ToTensor(object):
	def __init__(self, classification_counting=False):
		self.classification_counting = classification_counting

	def __call__(self, sample):
		image, counting = sample['image'], sample['counting']
		image = image.transpose((2, 0, 1))
		sample['image'] = torch.from_numpy(image).float()
		sample['counting'] = torch.from_numpy(counting).long() - 1 if self.classification_counting \
							else torch.from_numpy(counting).float()
		
		for sample_key in sample.keys():
			if sample_key in MASK_TYPES:
				normalized_mask = torch.from_numpy(sample[sample_key] / 255.0)
				sample[sample_key] = (normalized_mask * 255.0 / CONTOUR_INTENSITY_SCALE).long()

		return sample


class Normalize(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, sample):
		image = sample['image']
		image = image / 255.0
		for t, m, s in zip(image, self.mean, self.std):
			t.sub_(m).div_(s)

		sample['image'] = image
		return sample






