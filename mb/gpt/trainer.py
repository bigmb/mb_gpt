from dataclasses import dataclass
from typing import List, Optional
from mb.utils.yaml_reader import read_yaml
import torch
from .utils.yaml_config import TrainParams, DataParams, ModelParams, OutputParams
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from mb.utils.logging import logg,logger
from mb.gpt.utils.gpu_tools import get_gpus_by_least_usage
from mb.gpt.utils.train_summary import TrainSummary

__all__ = ['Trainer','DDPTrainer']

class Trainer:

    def __init__(self,
                 TrainParams: TrainParams, 
                 DataParams: DataParams,
                 ModelParams: ModelParams):
        self.TrainParams = TrainParams
        self.DataParams = DataParams
        self.ModelParams = ModelParams
    
    def train_model(self,model,optimizer,train_loader,device):
        for epoch in range(self.TrainParams.epochs):
            avg_loss = self.train_single_epoch(model,optimizer,train_loader,device)
            print(f"Epoch {epoch+1}/{self.TrainParams.epochs}, Loss: {avg_loss:.4f}")\


class DDPTrainer:
    
    def __init__(self,TrainParams:TrainParams,DataParams:DataParams,ModelParams:ModelParams,OutputParams: OutputParams):
        self.TrainParams = TrainParams
        self.DataParams = DataParams
        self.ModelParams = ModelParams
        
        ## Setting DDP
        self.debug = self.TrainParams.debug
        if self.debug:
            self.backend = 'gloo'
        else:
            self.backend = 'nccl'
            gpus = [str(i) for i in self.TrainParams.gpu if i in self.TrainParams.gpus]           
            os.environ['CUDA_VISIBLE_DEVICES']=",".join(gpus)
            self.rank=0
            self.world_size=len(gpus)
        self.device = f'cuda:{self.rank}'
        torch.cuda.set_device(self.device)

        ## Setting DataLoader


        ## Setting Model
        model = get_model(ModelParams)
        self.model = model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.rank])
        
        self.compute_loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.TrainParams.learning_rate)
        
        ## Setting Trianig Summary and data savepoints
        self.folder = self.DataParams.save_dir
        self.logger = logger
        self.trainsummary = TrainSummary(self.folder,logger=self.logger)
        
    def _setup_ddp(self):
        os.environ['MASTER_ADDR']='localhost'        
        os.environ['NASTER_PORT']='12355'
        
        dist.init_process_group(
        backend=self.backend, 
        rank=self.rank,
        world_size=self.world_size)
            
    def _clean_up(self):
        dist.destroy_process_group()

    def train_single_epoch(self,model,train_loader):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch
            self.optimizer.zero_grad()
            outputs = model(input_ids)
            loss = self.compute_loss(outputs, input_ids)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def val_single_epoch(self,model,val_loader):
        pass
    
    def train_model(self):
       
         
        for epoch in range(self.TrainParams.epochs):
            ##Training
            
            train_loss = self.train_single_epoch(epoch)
            self.trainsummary(train_loss)
            
            ##Validation
            val_loss = self.val_single_epoch(epoch)
            self.trainsummary(val_loss)
            
        self._clean_up()
