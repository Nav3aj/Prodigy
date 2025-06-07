"""task1_gpt2_finetune.py
Instruction: Fine-tune GPT-2 on a custom dataset for Task 01.
"""

# Topic: Check dependencies
import importlib
missing = [lib for lib in ("transformers", "datasets") if importlib.util.find_spec(lib) is None]
if missing:
    print(f"Missing dependencies: {', '.join(missing)}. Install via: pip install transformers datasets")
else:
    # Topic: Imports
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
    from datasets import load_dataset

    # Topic: Load dataset
    def load_text_data(dataset_name, split='train'):
        return load_dataset(dataset_name, split=split)

    # Topic: Tokenization
    def tokenize_function(examples, tokenizer):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    # Topic: Model initialization
    def initialize_model(model_name='gpt2'):
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, model

    # Topic: Prepare dataset
    def prepare_dataset(dataset, tokenizer):
        tokenized = dataset.map(lambda ex: tokenize_function(ex, tokenizer), batched=True)
        tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        return tokenized

    # Topic: Training setup
    def setup_training_args(output_dir, epochs=3, batch_size=4, lr=5e-5):
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            weight_decay=0.01,
            save_steps=500,
            save_total_limit=1,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
        )

    # Topic: Train model
    def train_model(model, tokenizer, train_dataset, training_args):
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer
        )
        trainer.train()
        return trainer

    # Topic: Generate text
    def generate_text(trainer, tokenizer, prompt, max_length=100):
        inputs = tokenizer(prompt, return_tensors='pt')
        outputs = trainer.model.generate(**inputs, max_length=max_length)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Topic: Tests
if __name__ == '__main__':
    # Check imports
    if missing:
        print('Cannot run tests: missing dependencies.')
    else:
        tokenizer, model = initialize_model()
        sample = ['Hello world!', 'Testing GPT-2.']
        enc = tokenizer(sample, return_tensors='pt', padding=True, truncation=True)
        assert 'input_ids' in enc and 'attention_mask' in enc, 'Tokenization failed'
        print('GPT-2 imports and tokenization OK')
