from constants import *
from evals.segmentation import SegmentationMask, EvalBinarySegmentation
from evals.counting import EvalCounting, DetailCounting
from utils import network, util_log, util_os
from utils import terrace as post_processing

W_c = [0.5, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]

class Model():
	def __init__(self, arch, num_contour=5):
		self.model = arch.to(device)
		self.name = arch.name
		print ('Architecture Network: {}'.format(self.name))
		self.num_contour = num_contour + 1   #add background

		self.opt_model_path = None
		self.last_model_path = None
		self.num_epoch = 512
		self.max_lr = 2e-4
		self.base_lr = 1e-6
		self.lr_step = 32
		self.lr = self.max_lr		

		self.segmentation_criterion = network.cross_entropy_loss2d
		self.counting_criterion = nn.L1Loss().to(device)
		self.reg_criterion = nn.MSELoss().to(device)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

		self.loss = None
		self.cnt_loss = None
		self.seg_loss = None

		self.eval_seg = None
		self.eval_rc = None

		self.epoch_loss_ = 0  #for each of train or val loss
		self.epoch_loss = 0   #sum of train and val loss
		self.epoch_id = 0
		self.min_epoch_loss = 1e5
		self.current_time = time.time()

		self.training_phase = 0
		self.log = None
		self.model_id = None


	def train(self, train_loader, val_loader, model_id=0):
		self.set_opt_model_path_and_log(train_loader, model_id)
		self.model_id = model_id
		self.set_log(self.opt_model_path)

		for epoch_id in range(1, self.num_epoch + 1):
			self.epoch_id = epoch_id

			self.show_epoch_info(epoch_id)
			self.epoch_loss = 0

			self.run_epoch(train_loader)
			self.run_epoch(val_loader)
			
			self.save_checkpoint(epoch_id)			
			self.update_lr(epoch_id)

	def test(self, test_loader):
		self.run_epoch(test_loader)

	def test_counting(self, ckpt_path, test_loader):
		self.load_ckpt(ckpt_path)
		self.model.eval()
		self.eval_rc = EvalCounting()

		for _, batch in enumerate(test_loader.loader):
			batch_data = BatchData(batch)
			image = batch_data.image.to(device)
			gt_counting = batch_data.counting.to(device)

			if image.size(0) != test_loader.batch_size: break

			_, pred_counting = self.model(image)
			self.eval_rc.update_batch_gt_pred_counting(gt_counting, pred_counting)

		print(self.eval_rc.get_summarization())
		self.eval_rc.show_detail_counting_result()
		self.write_detail_result(ckpt_path, test_loader)


	def run_epoch(self, loader):
		if loader.is_training: self.model.train()
		else: self.model.eval()
			
		self.refresh_eval_and_loss()

		for _, batch in enumerate(loader.loader):
			self.run_batch(loader, batch)
		
		self.epoch_loss_ /= loader.length
		self.epoch_loss += self.epoch_loss_
		self.summary_epoch(loader)


	def run_batch(self, loader, batch_data):
		image = batch_data[1].to(device)
		gt_counting = batch_data[2].to(device)		
		gt_mask = batch_data[3].to(device)
		gt_full = gt_mask.clone()
		gt_full[gt_mask > 1] = 1
		if image.size(0) != loader.batch_size: return

		pred_mask, pred_counting = self.model(image)	
		self.eval_seg.update_batch_gt_pred_mask(gt_full, pred_mask)
		self.eval_rc.update_batch_gt_pred_counting(gt_counting, pred_counting)

		loss_weight = torch.tensor(W_c[:self.num_contour]).to(device)  #3
		self.seg_loss = self.segmentation_criterion(pred_mask, gt_mask, loss_weight)
		self.cnt_loss = self.counting_criterion(pred_counting, gt_counting)
		self.reg_loss = self.get_reg_loss(pred_mask, gt_mask)

		self.pick_loss()
		self.epoch_loss_ += self.loss.item()             
		
		self.backprop(loader, self.loss)


	def get_reg_loss(self, pred_mask, gt_mask):
		gt_mask = gt_mask.float()
		(BATCH_SIZE, DEPTH, S, S) = pred_mask.size()
		hA = pred_mask.view(BATCH_SIZE, DEPTH, -1)	
		hA = hA.transpose(1, 2)				
		softmax = nn.Softmax(dim=2)
		heat_map = softmax(hA)
		WEIGHT = torch.tensor([[i] for i in range(self.num_contour)], dtype=torch.float).to(device)
		heat_map = torch.matmul(heat_map, WEIGHT)
		heat_map = heat_map.view(BATCH_SIZE, S, S)
		return self.reg_criterion(gt_mask, heat_map)


	def show_epoch_info(self, epoch_id):
		self.log.info ('\nEpoch [{}/{}], lr {:.6f}, phase: {}, runtime {:.3f}'.format(epoch_id, self.num_epoch, self.lr, 
			self.training_phase, time.time()-self.current_time))
		self.current_time = time.time()

	def summary_epoch(self, loader):
		self.log.info("{}: loss={:.6f}, seg_loss={:.6f}, reg_loss={:.6f}, cnt_loss={:.6f}"\
			.format(loader.name, self.epoch_loss, self.seg_loss, self.reg_loss, self.cnt_loss))
		self.log.info(self.eval_rc.get_summarization())
		self.log.info(self.eval_seg.get_summarization())

	def save_checkpoint(self, epoch_id):
		if self.min_epoch_loss > self.epoch_loss:
			self.min_epoch_loss = self.epoch_loss
			torch.save(self.model.state_dict(), self.opt_model_path)
			self.log.info ("checkpoint saved")

		if epoch_id % 32 == 0:
			torch.save(self.model.state_dict(), self.last_model_path.replace('last', str(epoch_id)))

	def refresh_eval_and_loss(self):
		self.eval_rc = EvalCounting()
		self.eval_seg = EvalBinarySegmentation()
		self.epoch_loss_ = 0

	def pick_loss(self):
		self.loss = self.seg_loss + self.cnt_loss
		if self.epoch_id > 64: self.loss += self.reg_loss

	def backprop(self, loader, loss):
		if loader.is_training:
			self.optimizer.zero_grad()
			self.loss.backward()
			self.optimizer.step()

	def set_opt_model_path_and_log(self, train_loader, model_id):
		model_name = self.name + '_' + train_loader.name + '_' + str(model_id)
		self.opt_model_path = CKPT_PATH + model_name + '.ckpt'
		self.last_model_path = CKPT_PATH + model_name + '_last.ckpt'

		
	def set_log(self, ckpt_path):
		model_name = ckpt_path.split(CKPT_PATH)[-1].split('.ckpt')[0]
		log_path = LOG_PATH + model_name + '.txt'
		self.log = util_log.Allog(log_path)
	
	
	def load_ckpt(self, ckpt_path=None):
		if ckpt_path == "default": ckpt_path = self.opt_model_path
		self.model.load_state_dict(torch.load(ckpt_path))

	def update_lr(self, epoch_id):
		if epoch_id % self.lr_step == 0: self.max_lr *= 0.88
		self.lr = self.max_lr - (self.max_lr - self.base_lr) * (epoch_id % self.lr_step) / self.lr_step
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = self.lr

	def switch_mode(self, epoch_id):
		if epoch_id % (3 * self.lr_step) == 1: 
			self.training_phase = 0
			network.freeze_children(self.model, [i for i in range(9, 11)])

		elif epoch_id % (3 * self.lr_step) == self.lr_step + 1: 
			self.training_phase = 1
			network.unfreeze_children(self.model, [i for i in range(9, 11)])
			network.freeze_children(self.model, [i for i in range(3, 9)])

		elif epoch_id % (3 * self.lr_step) == 2 * self.lr_step + 1: 
			self.training_phase = 2
			network.unfreeze_children(self.model, [i for i in range(11)])


	def write_detail_result(self, ckpt_path, loader):
		detail_counting = DetailCounting(ckpt_path, loader)
		for _, batch in enumerate(loader.loader):
			batch_data = BatchData(batch)
			image = batch_data.image.to(device)
			image_path = batch_data.image_path
			if image.size(0) != loader.batch_size: return

			_, pred_counting = self.model(image)
			detail_counting.write_batch_result(image_path, pred_counting)


	def save_instance_map(self, ckpt_path, loader):
		self.load_ckpt(ckpt_path)
		self.model.eval()

		model_name = ckpt_path.split(CKPT_PATH)[-1][:-5]
		mask_dir = util_os.gen_dir("{}{}_on_{}_256".format(RESULT_PATH, model_name, loader.name), True)
		print ("instance maps are save at " + mask_dir)
		image_path_ids = defaultdict(lambda : -1, OrderedDict())
				
		for _, batch_data in tqdm(enumerate(loader.loader)):
			image_paths = batch_data[0]
			image = batch_data[1].to(device)			
			if image.size(0) != loader.batch_size: return

			pred_mask, pred_counting = self.model(image)
			_, pred_mask = torch.max(pred_mask.data, 1)
			np_pred_mask = pred_mask.data.cpu().numpy()

			for batch_id in range(loader.batch_size):
				image_path = image_paths[batch_id].replace('/', '_').split('.')[0]
				image_path_ids[image_path] += 1
				mask_path = "{}{}_{}.png".format(mask_dir, image_path, image_path_ids[image_path])

				np_pred_mask_ = np_pred_mask[batch_id]
				post_processing.save_instance_map(np_pred_mask_, mask_path)


	def test_speed_instance_segmentation(self, ckpt_path, ann_file):
		from utils import image_preprocessing
		self.load_ckpt(ckpt_path)
		self.model.eval()
		ann_file = DATA_PATH + 'train_val/' + ann_file
		dataset = list(open(ann_file, 'r'))
		image_paths = []
		for line in dataset:
			[image_path, _, _] = line.strip().split(',,,')
			image_paths.append(DATA_PATH + 'image/' + image_path)

		time_loading_image, time_preprocessing, time_model, time_post = 0, 0, 0, 0
		for image_id, image_path in tqdm(enumerate(image_paths)):
			mask_path = "{}{}.png".format(BUFFER_PATH, image_id)
			t_0 = time.time()

			image = Image.open(image_path).convert('RGB')
			t_1 = time.time()

			image = image.resize((256, 256), Image.ANTIALIAS)
			image = np.array(image) / 255.0
			image = image.transpose(2, 0, 1)
			image = torch.from_numpy(image).float()
			for t, m, s in zip(image, IMAGENET_MEAN, IMAGENET_STD):
				t.sub_(m).div_(s)
			image = torch.unsqueeze(image, 0).to(device)
			t_2 = time.time()

			pred_mask, pred_counting = self.model(image)
			_, pred_mask = torch.max(pred_mask.data, 1)
			np_pred_mask = pred_mask.data.cpu().numpy()
			t_3 = time.time()


			time_post += post_processing.save_instance_map(np_pred_mask[0], mask_path)
			
			time_loading_image += t_1 - t_0
			time_preprocessing += t_2 - t_1
			time_model += t_3 - t_2
			# time_post += time.time() - t_1

		time_loading_image *= 1000 / len(image_paths)
		time_preprocessing *= 1000 / len(image_paths)
		time_model *= 1000 / len(image_paths)
		time_post *= 1000 / len(image_paths) 

		total_time = time_loading_image + time_preprocessing + time_model + time_post
		print ("FPS: {:.2f}".format(1000. / total_time))
		print ("FPS: {:.2f}".format(1000. / total_time))
		print ('Runtime: loading image | preprocessing | model | postprocessing | =  \
			{:.0f} ms-{:.0f}% | {:.0f} ms-{:.0f}% | {:.0f} ms-{:.0f}% | {:.0f} ms-{:.0f}%'
			.format(time_loading_image, 100*time_loading_image/total_time, 
				time_preprocessing, 100*time_preprocessing/total_time, 
				time_model, 100*time_model/total_time, time_post, 100*time_post/total_time))


	







