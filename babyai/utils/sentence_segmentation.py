import torch
import torch.nn as nn
import torch.optim as optim

def hidden_zeros(seq):
    return torch.zeros_like(seq[:, 0, ...]).unsqueeze(0)

class SentenceMask(torch.nn.Module):
    def __init__(self, n_words, hid_size):
        super().__init__()
        self.emb = torch.nn.Embedding(n_words, hid_size)
        self.lstm = torch.nn.LSTM(hid_size, hid_size, 1, batch_first=True)
        self.projection = torch.nn.Linear(hid_size, 2)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        x = self.emb(x)
        outputs, _ = self.lstm(
            x,
            (hidden_zeros(x), hidden_zeros(x))
        )
        outputs = self.projection(outputs.squeeze(1))

        # B x S x 2
        return outputs.reshape(batch_size, seq_len, 2)

def split_sentence_by_model(model, idxs, sentence):
    split_points = model(torch.tensor([
        [
            idxs[w] for w in sentence.split()
        ]
    ])).argmax(dim=-1).numpy().flatten().tolist()

    sentence_w = []
    collect = 0
    for w, p in zip(sentence.split(), split_points):
        if p == 1:
            collect = 1

            if sentence_w:
                yield sentence_w

            sentence_w = []

        if collect == 1:
            sentence_w.append(w)

    yield sentence_w

def load_split_model(name):
    idxs_fname = name + ".idx"
    idxs = torch.load(idxs_fname)
    model = SentenceMask(len(idxs), 4)
    model.load_state_dict(torch.load(name))

    if torch.cuda.is_available():
        model = model.to('cuda')

    return model, idxs
