import os
import cv2
import timeit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import onnx
import onnxruntime as ort
from onnxruntime.transformers import optimizer

print(os.getcwd())

from dataloading_utils import load_data, train_test_split, create_dataloader, create_synth_data
from models import SimpleCNN, SimpleCNN2

def sign_magnitude_loss(output, target):
    '''
    Sign magnitude loss function
    '''
    # Split the target and output into their respective components
    target_forward_speed = target[:, 0]
    target_steering_speed = target[:, 1]

    output_forward_speed = output[:, 0]
    output_steering_magnitude = output[:, 1]
    output_steering_sign = output[:, 2]

    forward_speed_loss = F.mse_loss(output_forward_speed, target_forward_speed)

    steering_magnitude_loss = F.mse_loss(output_steering_magnitude, torch.abs(target_steering_speed))

    target_steering_sign = torch.sign(target_steering_speed)
    target_steering_prob = (target_steering_sign + 1) / 2  # Convert -1/1 to 0/1 probabilities

    bce_logits_loss = F.binary_cross_entropy(output_steering_sign, target_steering_prob)

    total_loss = (forward_speed_loss + steering_magnitude_loss) * 1.5 + bce_logits_loss

    return total_loss

class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, lr: float = 1e-3, weight_decay: float = 1e-5, epochs: int = 10) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
        self.epochs = epochs
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = sign_magnitude_loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        
        self.loss_history = []
        self.test_history = []
        self.epoch_loss = []
        self.batch_inference_time = []
        
    def train(self):
        
        self.best_model = None
        best_mse = float('inf')
        best_epoch = 0
        patience = 5
        
        for epoch in range(self.epochs):
            self.model.train()
            
            loss_epoch = 0
            
            for i, (data, labels) in enumerate(self.train_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                self.loss_history.append(loss.item())
                loss_epoch += loss.item()
                
                loss.backward()
                self.optimizer.step()
                
                if i % 10 == 0:
                    print(f'Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}')
                    
            mse = self.test()
            self.test_history.append(mse)
            self.epoch_loss.append(loss_epoch / len(self.train_loader))
            
            if mse < best_mse:
                best_mse = mse
                self.best_model = self.model.state_dict()
                best_epoch = epoch
                
            if epoch - best_epoch > patience:
                print('Early stopping')
                break
            
                    
    def test(self):
        self.model.eval()
        with torch.no_grad():
            
            mse = 0
            
            for data, labels in self.test_loader:
                # Calculate mse
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                # take the last component of the output and get the sign
                sign = torch.sign(outputs[:, 2])
                outputs[:, 1] = outputs[:, 1] * sign
                outputs = outputs[:, :2]
                loss = nn.functional.mse_loss(outputs, labels)
                mse += loss.item()
                
            # Calculate inference time
            self.measure_inference_time()
                
            mse /= len(self.test_loader)
            
            print(f'Mean Squared Error: {mse}')
            print(f'Average inference time: {self.average_inference_time * 1000:.3f} ms')
            
        return mse
            
    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def save_and_optimize_onnx(self, path: str):
        onnx_model_path = path.replace('.pth', '.onnx')
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        torch.onnx.export(self.model, dummy_input, onnx_model_path, 
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        print(f'Model exported to ONNX at {onnx_model_path}')
        
        onnx_model = onnx.load(onnx_model_path)
        
        optimized_model_path = path.replace('.pth', '_optimized.onnx')
        optimization_options = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        optimized_model = optimizer.optimize_model(onnx_model, optimization_options)
        optimized_model.save_model_to_file(optimized_model_path)
        print(f'Optimized ONNX model saved at {optimized_model_path}')
        
    def load_model(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))
        
    def measure_inference_time(self) -> None:
        random_data = torch.rand((1, 3, 224, 224)).to(self.device)
        # Timeit 1000 times
        self.average_inference_time = timeit.timeit(lambda: self.model(random_data), number=1000) / 1000
        


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    dataset_path = 'dataset'
    debug = True
    batch_size = 128
    lr = 2e-4
    w_decay = 1e-5
    epochs = 1
    
    data, labels = load_data(dataset_path, debug=debug)
    
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, debug=debug)  
    
    flip_train_data, flip_train_labels = create_synth_data(train_data, train_labels)
    
    train_data = np.concatenate([train_data, flip_train_data])
    train_labels = np.concatenate([train_labels, flip_train_labels])
    
    # Free up some memory (it crashes on my 16GB RAM laptop if I don't do this)
    del flip_train_data
    del flip_train_labels
    del data
    del labels
    
    print(f'Training data shape: {train_data.shape}, training labels shape: {train_labels.shape}')
    print(f'Testing data shape: {test_data.shape}, testing labels shape: {test_labels.shape}')

    train_loader: DataLoader = create_dataloader(train_data, train_labels, batch_size=batch_size, shuffle=True)
    test_loader: DataLoader = create_dataloader(test_data, test_labels, batch_size=batch_size, shuffle=False)
    
    model = SimpleCNN()
    
    trainer = Trainer(model, train_loader, test_loader, lr=lr, weight_decay=w_decay, epochs=epochs)
    trainer.train()
    
    # Load the best model
    trainer.model.load_state_dict(trainer.best_model)
    
    trainer.test()

    trainer.save_model('models/model.pth')
    
    trainer.save_and_optimize_onnx('models/model.pth')
    
    # Plot the epoch train test loss
    plt.plot(trainer.epoch_loss, label='Train loss')
    plt.plot(trainer.test_history, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    
    trainer.measure_inference_time()
    print(f'Average inference time: {trainer.average_inference_time}')
    
    
