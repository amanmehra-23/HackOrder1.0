from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import pandas as pd

# Load the pre-trained model and tokenizer
model_name = "gpt2"  # Replace with your model of choice, e.g., "gpt3"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Load your preprocessed data
file_path = 'final_training_data.csv'  # Replace with the path to your downloaded CSV file
final_training_data = pd.read_csv(file_path)

# Prepare the dataset
class TravelDataset(TextDataset):
    def __init__(self, tokenizer, file_path, block_size):
        prompts = final_training_data['Prompt'].tolist()
        responses = final_training_data['Response'].tolist()
        lines = [prompts[i] + ' ' + responses[i] for i in range(len(prompts))]

        with open('intermediate_training_file.txt', 'w') as f:  # This file will be created in your working directory
            for line in lines:
                f.write(line + '\n')

        super().__init__(tokenizer=tokenizer, file_path='intermediate_training_file.txt', block_size=block_size)

# Specify the block size
block_size = 128  # You can adjust this based on the model's max input length and your available memory

# Initialize the dataset
dataset = TravelDataset(tokenizer, 'intermediate_training_file.txt', block_size)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
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

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
