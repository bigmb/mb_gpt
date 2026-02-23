from dataclasses import dataclass
from typing import List, Optional
from mb.utils.yaml_reader import read_yaml
import torch
from .utils.yaml_config import TrainParams, DataParams, ModelParams



class Trainer:
    def __init__(self,TrainParams:TrainParams,DataParams:DataParams,ModelParams:ModelParams):
        self.TrainParams = TrainParams
        self.DataParams = DataParams
        self.ModelParams = ModelParams

    def train_single_epoch(self,model,optimizer,train_loader,device):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = self.compute_loss(outputs, input_ids)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def train_model(self,model,optimizer,train_loader,device):
        for epoch in range(self.TrainParams.epochs):
            avg_loss = self.train_single_epoch(model,optimizer,train_loader,device)
            print(f"Epoch {epoch+1}/{self.TrainParams.epochs}, Loss: {avg_loss:.4f}")\
    
