import csv, torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # Apply tokenization and padding to the input text
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        return input_ids, attention_mask, torch.tensor(label)

def get_data():
    raw_data = {
        "data": [],
        "labels": []
    }
    categories = ["ADHD", "depression", "Anxiety", "OCD"]
    # populate
    for i, category in enumerate(categories):
        with open("data/{0}.csv".format(category), 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # skip the header row
            for row in csv_reader:
                raw_data["data"].extend([row[0]])
                raw_data["labels"].extend([list(map(int, row[1:]))])
    return raw_data