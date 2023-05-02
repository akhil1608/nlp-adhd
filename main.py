import torch

from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, random_split

from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import f1_score, hamming_loss

from models import BERTModel, LSTMModel
from util import CustomDataset, get_data

def train(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs, mask=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss, train_f1, train_hl = 0.0, 0.0, 0.0
        model.train()
        for inputs, masks, labels in train_loader:
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, masks) if mask else model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_f1 += f1_score(labels.cpu().numpy(), (outputs > 0.5).float().cpu().numpy(), average='micro')
            train_hl += hamming_loss(labels.cpu().numpy(), (outputs > 0.5).float().cpu().numpy())
        train_loss /= len(train_loader.dataset)
        train_f1 /= len(train_loader)
        train_hl /= len(train_loader)
        val_loss, val_f1, val_hl = evaluate(model, val_loader, criterion, mask)
        print(f'Epoch {epoch+1}/{num_epochs} : Train - Loss = {train_loss:.4f}, F1 = {train_f1:.4f} HL = {train_hl:.4f}; Val - Loss = {val_loss:.4f}, F1 = {val_f1:.4f}, HL = {val_hl:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'saved_models/{model.name}.pt')
        scheduler.step()
    model.load_state_dict(torch.load(f'saved_models/{model.name}.pt'))
    return model

def evaluate(model, dataloader, criterion, mask, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss, f1, hl = 0.0, 0.0, 0.0
    with torch.no_grad():
        for inputs, masks, labels in dataloader:
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            outputs = model(inputs, masks) if mask else model(inputs)
            loss += criterion(outputs, labels.float()).item() * inputs.size(0)
            f1 += f1_score(labels.cpu().numpy(), (outputs > 0.5).float().cpu().numpy(), average='micro')
        loss /= len(dataloader.dataset)
        f1 /= len(dataloader)
        hl /= len(dataloader)
    if verbose:
        print(f'Test - Loss = {loss:.4f}, F1 = {f1:.4f}, HL = {hl:.4f}')
    return loss, f1, hl

# model parameters
num_classes, lstm_dim, num_layers, max_length, dropout = 4, 64, 1, 128, 1e-1
# training parameters
num_epochs, batch_size, bert_lr, lstm_lr, l2_reg = 20, 32, 1e-5, 1e-1, 1e-5

# prepare datasets
raw_data = get_data()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = CustomDataset(raw_data["data"][:200], raw_data["labels"][:200], tokenizer, max_length)
train_dataset, val_dataset, test_dataset = random_split(dataset, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(24))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# BERT
model = BERTModel(num_classes, dropout)
optimizer = AdamW(model.parameters(), lr=bert_lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*10)
criterion = BCEWithLogitsLoss()
model = train(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs, True)
evaluate(model, test_loader, criterion, mask=True, verbose=True)

# reset to use LSTM
model = LSTMModel(len(tokenizer), max_length, lstm_dim, num_classes, num_layers, l2_reg)
optimizer = Adam(model.parameters(), lr=lstm_lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*10)
criterion = BCEWithLogitsLoss()
model = train(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs)
evaluate(model, test_loader, criterion, mask=False, verbose=True)