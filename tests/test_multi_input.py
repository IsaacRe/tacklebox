import torch
from transformers import XLMTokenizer, XLMWithLMHeadModel
import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import torch.nn as nn
from tacklebox.hook_management import HookManager


if __name__ == '__main__':
    # load data
    english = spacy.load('en')
    german = spacy.load('de')

    def tokenize_en(text):
        return [tok.text for tok in english.tokenizer(text)]
    def tokenize_de(text):
        return [tok.text for tok in german.tokenizer(text)]

    en_text = Field(sequential=True, use_vocab=True, tokenize=tokenize_en, lower=True)
    de_text = Field(sequential=True, use_vocab=True, tokenize=tokenize_de, lower=True)

    train, val, test = Multi30k.splits(root='../../data', exts=('.en', '.de'), fields=(en_text, de_text))

    en_text.build_vocab(train, max_size=30000, min_freq=3)
    de_text.build_vocab(train, max_size=30000, min_freq=3)
    vocab_en = en_text.vocab
    vocab_de = de_text.vocab
    pad_idx = vocab_de.stoi['<pad>']

    train_ldr, val_ldr, test_ldr = BucketIterator.splits((train, val, test),
                                                        batch_size=5)

    # load model
    xlm = XLMWithLMHeadModel.from_pretrained('xlm-mlm-ende-1024')
    xlm.transformer.embeddings = nn.Embedding(len(vocab_en), xlm.config.emb_dim, padding_idx=pad_idx)
    xlm.pred_layer.proj = nn.Linear(xlm.config.emb_dim, len(vocab_de), bias=True)
    xlm.cuda()

    xent = nn.CrossEntropyLoss()

    batch = next(iter(train_ldr))
    src, trg = batch.src.to(0), batch.trg.to(0)

    def mt_loss(out, target):
        # only compute loss for non-padding indices
        min_idx = min([out.shape[0], target.shape[0]])
        out, target = out[:min_idx], target[:min_idx]
        mask = (target != pad_idx).type(torch.bool)
        return xent(out[mask], target[mask])


    out, = xlm(src)
    mt_loss(out, trg).backward()

    named_modules = {
        'embedding': xlm.transformer.embeddings,
        'final_attn': xlm.transformer.attentions[-1],
    }

    hookmngr = HookManager()

    # dealing with multiple hook functions

    # forward pre-hook function signature: (module, inputs)
    def zero_input(module, inputs):
        print('Set %s input to zero' % module.name)
        ret = []
        for input in inputs:
            if type(input) == torch.Tensor and input.dtype == torch.float:
                ret += [input - input]
            else:
                ret += [input]
        return tuple(ret)


    def print_mean(module, inputs, outputs):
        print('%s input mean = %.2f, output mean = %.2f' % (module.name,
                                                            inputs[0].sum().item() / inputs[0].numel(),
                                                            outputs[0].sum().item() / outputs[0].numel()))


    hookmngr.register_forward_pre_hook(zero_input, hook_fn_name='zero_input', activate=False, **named_modules)

    # to pass modules without names, pass as args instead of as kwargs
    hookmngr.register_forward_hook(print_mean, *named_modules.values(), hook_fn_name='print_mean', activate=False, )

    # note we could use separate hook manager for XLM

    with hookmngr.hook_all_context() + torch.no_grad():
        xlm(src)

    # lets only activate our forward hooks
    with hookmngr.hook_all_context(category='forward_hook') + torch.no_grad():
        xlm(src)
