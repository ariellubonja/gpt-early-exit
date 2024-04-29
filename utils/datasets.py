from datasets import load_dataset
# ChatGPT - take 100 shortest entries from Tiny Shakespeare
def get_top_n_tiny_shakespeare(n=25, mode='longest'):
    # Load the dataset
    train_dataset = load_dataset("Trelis/tiny-shakespeare", split="train")

    # Define a function to get the length of each input
    def input_length(example):
        return len(example['Text'])

    # Add a new field to each example in the dataset containing the length of the input
    train_dataset = train_dataset.map(lambda x: {"length": input_length(x)})

    # Sort the dataset by the newly added 'length' field
    if mode == 'longest':
        sorted_dataset = train_dataset.sort("length", reverse=True)
    elif mode == 'shortest':
        sorted_dataset = train_dataset.sort("length")

    # Get the top 100 shortest inputs
    shortest_100 = sorted_dataset.select(range(n))

    # Print the shortest 100 inputs (optional)
    # for example in shortest_100:
    #     print(len(example['Text']))

    return shortest_100 