import torch as t

class TwoPMLP(t.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.embedding = t.nn.Embedding(params.p1 + params.p2, params.embed_dim)
        
        self.linear1r = t.nn.Linear(params.embed_dim, params.hidden_size, bias=True)
        self.linear1l = t.nn.Linear(params.embed_dim, params.hidden_size, bias=True)
        
        self.tie_unembed = params.tie_unembed
        if params.tie_unembed:
            self.linear2 = t.nn.Linear(params.hidden_size, params.embed_dim, bias=True)
        else:
            self.linear2 = t.nn.Linear(params.hidden_size, params.p1 + params.p2, bias=False)
        
        if params.activation == "relu":
            self.act = t.nn.ReLU()
        elif params.activation == "gelu":
            self.act = t.nn.GELU()
        elif params.activation == "quad":
            self.act = lambda x: x ** 2
        else:
            raise ValueError(f"Unknown activation function {params.activation}")

        self.vocab_size = params.p1 + params.p2

        self.linear1r.weight.data *= params.scale_linear_1_factor
        self.linear1l.weight.data *= params.scale_linear_1_factor
        self.embedding.weight.data *= params.scale_embed

        self.saved_activations = {}
        self.params = params

    def forward(self, x1, x2, x3, x4):
        emb1 = self.embedding(x1)
        emb2 = self.embedding(x2)
        if self.params.linear_1_tied:
            out1 = self.linear1r(emb1)
            out2 = self.linear1r(emb2)
        else:
            out1 = self.linear1l(emb1)
            out2 = self.linear1r(emb2)
        pair1 = out1 + out2
        emb3 = self.embedding(x3)
        emb4 = self.embedding(x4)
        if self.params.linear_1_tied:
            out3 = self.linear1r(emb3)
            out4 = self.linear1r(emb4)
        else:
            out3 = self.linear1l(emb3)
            out4 = self.linear1r(emb4)
        pair2 = out3 + out4

        x = pair1 + pair2

        x = self.act(x)
        x = self.linear2(x)

        if self.params.save_activations:
            if x1.dim() == 0:
                x1 = x1.unsqueeze(0)
                x2 = x2.unsqueeze(0)
                x3 = x3.unsqueeze(0)
                x4 = x4.unsqueeze(0)
            key = (x1[0].item(), x2[0].item(), x3[0].item(), x4[0].item())
            self.saved_activations[key] = x[0].cpu().clone().detach()

        if self.tie_unembed:
            x = x @ self.embedding.weight.T
        return x
