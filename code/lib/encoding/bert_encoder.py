import os
from dataclasses import dataclass

import torch
from transformers import BertModel, BertTokenizer  # type: ignore

"""
Okay so
this needs to serve two purposes: 1: convert all the documents to token tensors and SAVE them for faster training
But we also want the code to be able to do the whole pipeline if need be. Take a document, tokenize it, vectorize it,
and then feed it straight into the RNN. The first approach will save us time in training and testing, but the second
functionality is much better for later deployment of the model

How this is done:
Let the encoder have multiple functions:
THIS ONE IS DONE
1: generate_embeddings(filepath): returns vector embeddings for a SINGLE document

2: generate_dataset(cutoff_size: Optional[int] = None): given the base directory for text_classification:
generate a dataset to disk of all documents, splitting them into train/test and LABELLING them. if cutoff_size is
specified, only generate cutoff_size instances of train and test and then stop. NOTE: this function will ALWAYS
generate the dataset from scratch. This is intended as a preprocessing step. Returns a TensorDataset

3: generate_toy_dataset(): equivalent to calling generate_dataset(10)

4: load_dataset(): returns an already existing dataset at processed_path

5: dataset_exists(): returns True if a dataset matching name exists at processeed_path

6: generate_if_nonexistent(): helper function, if dataset_exists then load, if not generate and then load

Make Dataset_Loader next
"""


@dataclass
class bert_config:
    dataset_name: str
    unprocessed_path: str
    processed_path: str


class bert_encoder:
    dataset_name: str
    unprocessed_path: str
    processed_path: str

    def __init__(self, config: bert_config):
        self.dataset_name = config.dataset_name
        self.unprocessed_path = config.unprocessed_path
        self.processed_path = config.processed_path

    def generate_embeddings(self, filepath) -> torch.Tensor:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)

        with open(f"{filepath}") as file:
            contents = file.read()
            tokens = torch.tensor([], dtype=torch.int64)
            attention_mask = torch.tensor([], dtype=torch.int64)

            # NOTE: this truncates longer texts to only 512 tokens. We could look into batching the files if we need to
            # alternatively we can take the first 256 and last 256 tokens rather than first 512
            encoded_dict = tokenizer.encode_plus(
                contents,
                add_special_tokens=True,
                max_length=512,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            print(encoded_dict)
            tokens = torch.cat((tokens, encoded_dict["input_ids"]), 0)
            attention_mask = torch.cat((attention_mask, encoded_dict["attention_mask"]), 0)
        model.eval()

        # now feed our information into the model
        with torch.no_grad():
            # indices into BERTS dictionary of tokens

            # tells model to ignore padding
            # Not using this, its used to tell two seperate sentence contexts apart. not feasible for big documents
            # token_type_ids = encoded_dict["token_type_ids"]
            output = model(tokens, token_type_ids=None, attention_mask=attention_mask)

            # print(output)
            hidden_states = output[2]

            # At this point, hidden_states is a tuple of tensors. this makes it one big tensor
            embeddings = torch.stack(hidden_states, dim=0)

            # rearranges it to batch, tokens, layers, units
            embeddings = embeddings.permute(1, 2, 0, 3)
            print(embeddings.size())

            # next up: sum last 4 layers to get a token vector for each one.
            sum_token_vectors = []
            for batch in embeddings:  # only 1 batch rn, but may change later
                for token in batch:
                    # sum the last four rows of features. There are other ways to extract word information,
                    # but this has the best balance of accuracy with lowering unnecessary input complexity
                    sum_vec = torch.sum(token[-4:], dim=0)
                    sum_token_vectors.append(sum_vec)

            # stack into one 512 x 768 tensor
            final_output = torch.stack([tens for tens in sum_token_vectors])
            return final_output


if 1:
    readpath: str = "./data/stage_4_data/text_classification/"
    writepath: str = "./data/stage_4_data/processed/text_classification"
    testfile: str = "train/pos/0_9.txt"
    # creates some dirs
    if 1:
        if not os.path.exists(f"{writepath}/test/"):
            os.makedirs(f"{writepath}/test/")
        if not os.path.exists(f"{writepath}/train/"):
            os.makedirs(f"{writepath}/train/")

    # a very basic BERT model, good for just extracting vector embeddings from text
