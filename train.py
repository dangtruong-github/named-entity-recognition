from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from charcnn import WordDataset, SimpleCNN

# Hyperparameters
learning_rate = 1e-4
batch_size = 32
num_epochs = 5

# Dataset + Data Loader
train_dataset = WordDataset(type_ds="train")
val_dataset = WordDataset(type_ds="testa")
test_dataset = WordDataset(type_ds="testb")

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss, Optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train
for epoch in range(num_epochs):
    correct_samples = 0 
    total_samples = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        # Data to CUDA if possible
        #data = data.to(device=device)
        #label = label.to(device=device)
        
        #print(data.shape) # torch.Size([64, 1, 28, 28]) --> 1 color channel --> grey-scale image
        
        # Data to correct shape
        #data = data.reshape(data.shape[0], -1)
        
        # Forward
        output = model(data)
        loss = criterion(output, label)
        
        #print(output.shape)
        
        # Backprop 
        optimizer.zero_grad()
        loss.backward()
        
        # gradient descent or adam step
        optimizer.step()
        
        # Calculate accuracy through training
        #if epoch % 10 == 9:    
        
        _, cur_predictions = output.max(1)
        correct_samples += (cur_predictions == label).sum()
        total_samples += output.size(0)

        if batch_idx % 10 == 9:
            right_pred = (cur_predictions == label).sum()
            all_pred = output.size(0)
            print(f"Batch {batch_idx + 1}: Accuracy {float(right_pred) / float(all_pred):.2f}")
    
    # check accuracy through training
    #if epoch % 10 == 9:
    acc = float(correct_samples)/float(total_samples) * 100.0
    print(f"Epoch {epoch+1}: Got {correct_samples} / {total_samples} with accuracy {acc:.2f}%")