from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import pandas as pd

# Load the fine-tuned model and tokenizer
model_name = './fine_tuned_model'  # Path to your fine-tuned model
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Load the expanded dataset
file_path = 'expanded_training_data.csv'  # Replace with your file path
expanded_data = pd.read_csv(file_path)

# Prepare the dataset for training
class ExpandedTravelDataset(TextDataset):
    def __init__(self, tokenizer, file_path, block_size):
        prompts = expanded_data['Prompt'].tolist()
        responses = expanded_data['Response'].tolist()
        lines = [prompts[i] + ' ' + responses[i] for i in range(len(prompts))]

        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line + '\n')

        super().__init__(tokenizer=tokenizer, file_path=file_path, block_size=block_size)

# Initialize the dataset
dataset = ExpandedTravelDataset(tokenizer, 'intermediate_training_file.txt', block_size=128)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,  # Adjust the number of epochs if needed
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Continue training
trainer.train()

# Save the further fine-tuned model
model.save_pretrained('./further_fine_tuned_model')
tokenizer.save_pretrained('./further_fine_tuned_model')
