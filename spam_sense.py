import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset

# Load dataset from CSV
df = pd.read_csv('spam_ham_dataset.csv')
# Clean the data - remove any NaN values and duplicate entries
df = df.dropna()
df = df.drop_duplicates()

# Convert labels to numeric format (0 for ham, 1 for spam)
df['label'] = (df['label'] == 'spam').astype(int)

# Split the dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to Hugging Face datasets
train_data = Dataset.from_pandas(train_df)
test_data = Dataset.from_pandas(test_df)

# Tokenize the dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

train_data = train_data.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)

# Load BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = torch.argmax(torch.tensor(predictions), dim=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

# Training arguments
training_args = TrainingArguments(
    output_dir='./results', 
    num_train_epochs=3, 
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=16,   
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",    
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

# Save the model and tokenizer
model.save_pretrained('./models/spam_model')
tokenizer.save_pretrained('./models/spam_model')
