import torch, torch.optim as optim
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from deepspeed.ops.adam import FusedAdam
import wandb
import logging
import gc
import torch.distributed
import gc
import torch
from utils.utils import remove_bit


class TrainingModule(LightningModule):
	def __init__(
		self,
		model =  None,
		lr: float = 1e-4,
		record = False,
		epochs: int = 5000,
		lr_scheduler = 'default',
		num_annealing_steps = 10000,
		num_warmup_steps = 1000,
		dataset = None,
		deepspeed = False,
		logger = None,
		max_num_timesteps: int = 20,
		#Figure out how to do typing here
		global_splits = None,
		random_trees = None
	):
		super().__init__()
		self.model = model
		self.lr = lr
		self.record = record
		self.epochs = epochs
		self.warmup_steps = 400
		self.current_step_value = 0
		self.lr_scheduler = lr_scheduler
		self.num_annealing_steps = num_annealing_steps
		self.num_warmup_steps = num_warmup_steps
		self.dataset = dataset
		self.max_num_timesteps = max_num_timesteps
		self.global_splits = global_splits

		# Important: This property activates manual optimization.
		# Turning off automatic optimization so I can catch out of memory errors!
		self.automatic_optimization = False
		self.deepspeed = deepspeed
		self.logger_ = logger

	def forward(
		self,
		batched_tokenized_trees,
		t,
		phyla_embeddings
	):
		velocity, mask = self.model(batched_tokenized_trees, t, phyla_embeddings = phyla_embeddings, return_leafs_only = False, return_edges_only = True)
		edge_split_masks = batched_tokenized_trees[-1]
		edge_mask = batched_tokenized_trees[-2]
		return velocity, edge_split_masks, edge_mask

	def step(self, batch, eval = False):
		logs = {}
		v_pred, edge_split_masks, edge_mask = self.forward(batch['tokenized_trees'], batch['batched_time'], batch['phyla_embeddings'])
		velocity_labels = batch['batched_velocity']
		num_leaves = batch['num_leaves'] 

		for num in range(len(velocity_labels)):
			# if len(velocity_labels[num]) != len(edge_split_masks[num]):
			# 	#This is fine since velocity is defined on internal splits while edge split masks includes frivolous internal edges
			# 	print(f"Mismatch between edge masks length {len(edge_split_masks[num])} and velocity labels {len(velocity_labels[num])}!")
			num_leave = num_leaves[num]
			for i in velocity_labels[num]:
				real_max_bit = max(m.bit_length() for m in edge_split_masks[num])
				vel = i
				if vel.bit_length() == real_max_bit+1:
					vel = remove_bit(vel, num_leave+1)
				elif vel.bit_length() > real_max_bit+1:
					print(f"Whoa there is a big problem with this split mask {i} vs real max {real_max_bit}!")
					import pdb; pdb.set_trace()

				if vel not in edge_split_masks[num]:
					print(f"This split {vel} from velocity labels is not in edge splits {edge_split_masks[num]}!")
					import pdb; pdb.set_trace()
				else:
					print("WOOO ONE FOUND")
		
		print("Wow congrats")
		import pdb; pdb.set_trace()
		loss = ((v_pred - batch['batched_velocity'])**2).mean()
		logs['loss'] = loss
		return logs
			
		
	def training_step(self, batch, _):
		opt = self.optimizers()
		opt.zero_grad()

		if self.deepspeed:

			success = False
			num = 0
			failed = False
			
			#Logic, if we have an out of memmory error we just resample with a smaller subtree and rerun
			while not success:
				self.logger_.log(f"Entering step {num}", level=logging.INFO)
				error_tensor = torch.zeros(1).cuda()
				if num > 1:
					self.logger_.log(f"Batch is too large decreasing max tree size by a factor of 2 and num sequences", level=logging.INFO)
					if 'loss' in locals():
						loss = loss.detach()
						del loss 
						gc.collect()
					
					if num > 10:
						return torch.tensor(0)

					torch.cuda.empty_cache()
					torch.distributed.barrier()
					index, sub_tree_size, num_subtrees = self.dataset.chosen_tree

					if sub_tree_size <= 5:
						self.logger_.log(f"We have reached the minimum tree size", level=logging.INFO)
						num_subtrees = 5
						sub_tree_size = 5
					elif num_subtrees > 100:
						self.logger_.log(f"Number of subtrees way too big {torch.distributed.get_rank()}", level=logging.INFO)
						num_subtrees = 50
					else:
						num_subtrees = int(num_subtrees//2)
						sub_tree_size = int(sub_tree_size//2)

						if sub_tree_size < 5:
							sub_tree_size = 5
						if num_subtrees < 1:
							num_subtrees = 1

					sub_batch = self.dataset.__getitem__(index, preset_subtree_size = sub_tree_size)
					batch  = self.dataset.collate_fn([sub_batch], preset_subtree_num = num_subtrees)

					# TODO: Run without adaptive batch size speedup
					# TODO: Run with adaptive batch size speedup
					if num <= 2:
						new_max_aa = num_subtrees*sub_tree_size * self.dataset.return_max_length(self.dataset.name_to_seq)
						self.logger_.log(f"Updating the adaptive batch size sampler with this new information of the max aa of {new_max_aa}", level=logging.INFO)
						self.dataset.size_detector.update_max_aa(new_max_aa)
					
					torch.distributed.barrier()
					self.logger_.log(f"We have all recreated our batches now moving on", level=logging.INFO)
				try:
					loss_status_tensor = torch.zeros(torch.distributed.get_world_size()).cuda()
					logs = self.step(batch)
					if logs is not None:
						loss = logs['loss']
						memory_error_tensor = torch.zeros(1).cuda()

						#Go through every GPU get memmory used, if it is above 70% we will abort the manual backward and fail
						stop_manual_backward = False
						for i in range(torch.cuda.device_count()):
							# Get the current memory usage
							current_memory = torch.cuda.memory_allocated(i)
							# Get the total memory
							total_memory = torch.cuda.get_device_properties(i).total_memory
							fraction = current_memory/total_memory
							if fraction > 0.75:
								self.logger_.log(f"We detected that {i} device is above 75% memory usage!, will avoid manual backward!", level=logging.INFO)
								stop_manual_backward = True
								memory_error_tensor[0] = 1
							self.logger_.log(f"Device {i} is using {fraction} of its memory", level=logging.INFO)
							torch.distributed.barrier()
						
						#If one at least fails the memory check then we will scuttle the backward
						torch.distributed.all_reduce(memory_error_tensor)
						if memory_error_tensor[0] > 0:
							self.logger_.log(f"Wow some is about to OOM we are scuttling the backward", level=logging.INFO)
							stop_manual_backward = True

						#Okay what if one passes the memory check and still fails?
						#loss_status_tensor = torch.zeros(1).cuda()

						if not stop_manual_backward:
							self.manual_backward(loss)
							success = True
							failed = False
							self.logger_.log(f"Succeded!", level=logging.INFO)
							loss_status_tensor[torch.distributed.get_rank()] = 1
						else:
							self.logger_.log(f"Skipping backward!", level=logging.INFO)
							failed = True
							success = False 
							num += 1
							logs = None
							loss_status_tensor[torch.distributed.get_rank()] = 1

					else:
						self.logger_.log(f"Failed!", level=logging.INFO)
						num += 1
						loss_status_tensor[torch.distributed.get_rank()] = 1
				except RuntimeError as e:
					if 'out of memory' in str(e):
						self.logger_.log(f'WARNING: out of memory', level=logging.INFO)
						error_tensor[0] = 1
						failed = True	
						logs = {'loss':torch.tensor(0)}
						num += 1
						loss_status_tensor[torch.distributed.get_rank()] = 1
						self.logger_.log(f'Set up my status', level=logging.INFO)
					else:
						self.logger_.log(f'RAISING NEW ERROR {e}', level=logging.INFO)
						raise e
				finally:
					self.logger_.log(f'Entering check for the loss', level=logging.INFO)

					while loss_status_tensor.sum() != torch.distributed.get_world_size():
						torch.distributed.all_reduce(loss_status_tensor)
						self.logger_.log(f"Waiting for everyone to finish\t{loss_status_tensor.sum()}\t{loss_status_tensor}", level=logging.INFO)


					torch.distributed.barrier()
					torch.distributed.all_reduce(error_tensor)
					if error_tensor[0] > 0:
						self.logger_.log("Ooops someone had a OOM we should scuttle", level=logging.INFO)
						failed = True
						success = False
						num += 1
						
					#print("Waiting")
					torch.distributed.barrier()

				num += 1
				torch.distributed.barrier()
		else:
			success = False
			num = 0

			#Logic, if we have an out of memmory error we just resample with a smaller subtree and rerun
			while not success:

				#If fail will call zero grad again, may need this for deepspeed?
				opt.zero_grad()
				if num > 1:
					print("Batch is too large decreasing max tree and number of subtrees by a factor of 1.2")
					index, sub_tree_size, num_subtrees = self.dataset.chosen_tree
					new_sub_tree_size = sub_tree_size
					new_num_subtrees = int(num_subtrees//1.2)

					if new_num_subtrees == 0:
						new_num_subtrees = 1
						new_sub_tree_size = int(sub_tree_size//1.2)
					
					if new_sub_tree_size < 5:
						new_sub_tree_size = 5
						new_num_subtrees = 1
					
					if num <= 2:
						new_max_aa = new_num_subtrees*new_sub_tree_size * self.dataset.return_max_length(self.dataset.name_to_seq)
						print(f"Updating the adaptive batch size sampler with this new information of the max aa of {new_max_aa}")
						self.dataset.size_detector.update_max_aa(new_max_aa)
					
					if num > 10:
						print("We are spiraling, moving on")
						return torch.tensor(0)
					
					sub_batch = self.dataset.__getitem__(index, preset_subtree_size = new_sub_tree_size)
					batch  = self.dataset.collate_fn([sub_batch], preset_subtree_num = new_num_subtrees)
					print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
					print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024 ** 2} MB")

					gc.collect()
				try:
					print(f"Memory allocated before step: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
					print(f"Memory reserved before step: {torch.cuda.memory_reserved() / 1024 ** 2} MB")

					logs = self.step(batch)
					loss = logs['loss']

					print(f"Memory allocated before backward: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
					print(f"Memory reserved before backward: {torch.cuda.memory_reserved() / 1024 ** 2} MB")

					self.manual_backward(loss)
					success = True
					failed  = False

					print(f"Memory allocated after backward: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
					print(f"Memory reserved after backward: {torch.cuda.memory_reserved() / 1024 ** 2} MB")

				except RuntimeError as e:
					if 'out of memory' in str(e):
						print('WARNING: out of memory')
						if hasattr(torch.cuda, 'empty_cache'):
							#Not sure about this
							torch.cuda.empty_cache()

						print(f"Memory allocated after OOM: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
						print(f"Memory reserved after OOM: {torch.cuda.memory_reserved() / 1024 ** 2} MB")	

						num += 1
					else:
						raise e

		#print(f"Entering a new world with status {failed}")
		if not failed and logs is not None:
			for k, v in logs.items():        
				self.log(
							k, v.to("cuda"), on_step=True, on_epoch=False, prog_bar=True, logger=True,
							sync_dist=True
							)

			index, sub_tree_size, num_subtrees = self.dataset.chosen_tree
			lr = opt.optimizer.param_groups[0]["lr"]
			self.log('num_seq_per_subtree', sub_tree_size)
			logs['num_seq_per_subtree'] = sub_tree_size
			self.log('num_subtrees', num_subtrees)
			logs['num_subtrees'] = num_subtrees
			self.log('lr', lr)
			logs['lr'] = lr
			self.logger_.log(logs, level=logging.INFO)


		if logs is not None:
			if self.record:
				# wandb.log(logs)
				wandb.log(logs, step=self.global_step)
			if not self.dataset.msa_distance:
				self.dataset.update_normrf(logs['norm_rf_distance'])
			self.clip_gradients(
				opt,
				gradient_clip_val=1.0,             # tighten / loosen here
				gradient_clip_algorithm="norm"
			)

			self.current_step_value += 1
			opt.step()
			#print("Hi Im here waiting!")
			if self.deepspeed:
				torch.distributed.barrier()
	
			# Perform learning rate schedling
			if self.lr_scheduler == "cosine":
				sch1 = self.lr_schedulers()
				sch1.step()
			elif self.lr_scheduler == "cosine_warmup":
				sch1, sch2 = self.lr_schedulers()
				# Perform warmup
				if self.num_warmup_steps > 0:
					sch1.step()
					self.num_warmup_steps -= 1
				# Perform cosine annealing
				else:
					sch2.step()
			elif self.lr_scheduler == "warmup":
				sch1 = self.lr_schedulers()
				# Perform warmup
				if self.num_warmup_steps > 0:
					sch1.step()
					self.num_warmup_steps -= 1

			#ADD CODE HERE TO UPDATE ADAPTIVE BATCH SIZE SAMPLER
			return logs['loss']
		else:
			return torch.tensor(0)

	def validation_step(self, batch, batch_idx):
		print("Wow congrats you made it here!")
		#raise Exception("NOW FACE THE FOLLY OF YOUR EFFORTS!")

	def on_before_optimizer_step(self, optimizer):
		# Compute the 2-norm for each layer
		# If using mixed precision, the gradients are already unscaled here
		norms = grad_norm(self, norm_type=2)
		total = norms['grad_2.0_norm_total']

		layer_norms = {k: v for k, v in norms.items() if "total" not in k}
		max_grad = max(layer_norms.values())
		mean_grad = torch.mean(torch.stack(list(layer_norms.values())))

		self.log("grad_norm_max", max_grad, prog_bar=True, on_step=True)
		self.log("grad_norm_mean", mean_grad, prog_bar=False, on_step=True)

		# Optional: Print a warning if exploding
		if max_grad > 1:
			print(f"[Warning] Gradient norm unusually high: max={max_grad:.2e}, mean={mean_grad:.2e}")

		self.log("grad_norm_total",total)
		print(f"step {self.global_step:4d}  total_grad_norm = {total:.2f} mean is {mean_grad:.2f} max is {max_grad:.2f}")
		if self.record:
			wandb.log({"grad_norm_total": total}, step=self.global_step)
			wandb.log({"grad_norm_max": max_grad}, step=self.global_step)
			wandb.log({"grad_norm_mean": mean_grad}, step=self.global_step)

	def configure_optimizers(self):
		if self.deepspeed:
			optimizer = FusedAdam(self.parameters(), lr=self.lr)
		else:
			optimizer = optim.AdamW(self.parameters(), lr=self.lr)

		if self.lr_scheduler == 'cosine':
			sch1 = CosineAnnealingLR(optimizer, T_max=self.num_annealing_steps) # Set to current number of steps for training 7 days
			return [optimizer], [sch1]
		elif self.lr_scheduler == "cosine_warmup":
			sch1 = LinearLR(optimizer, start_factor=self.lr, total_iters=self.num_warmup_steps)
			sch2 = CosineAnnealingLR(optimizer, T_max=self.num_annealing_steps)
			return [optimizer], [sch1, sch2]
		elif self.lr_scheduler == "warmup":
			sch1 = LinearLR(optimizer, start_factor=self.lr, total_iters=self.num_warmup_steps)
			return [optimizer], [sch1]
		else:
			scheduler = []
			return optimizer
		
