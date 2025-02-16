import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import MLP
from .word_embedding import load_word_embeddings

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.multiprocessing.set_sharing_strategy('file_system')


class AOGN(nn.Module):
    def __init__(self, dataset, args):
        super(AOGN, self).__init__()
        self.args = args
        self.dataset = dataset

        self.val_forward = self.val_forward
        self.train_forward = self.train_forward

        self.num_attrs, self.num_objs, self.num_pairs = len(dataset.attrs), len(dataset.objs), len(dataset.pairs)
        self.pairs = dataset.pairs

        self.obj_head = MLP(dataset.feat_dim, args.emb_dim, num_layers=args.obj_fclayers, relu=args.relu,
                            dropout=args.dropout, norm=args.norm, layers=args.obj_emb)
        self.attr_head = MLP(dataset.feat_dim, args.emb_dim, num_layers=args.attr_fclayers, relu=args.relu,
                             dropout=args.dropout, norm=args.norm, layers=args.attr_emb)

        obj_words = list(dataset.objs)
        attr_words = list(dataset.attrs)

        self.obj_to_idx = {word: idx for idx, word in enumerate(dataset.objs)}
        self.attr_to_idx = {word: idx for idx, word in enumerate(dataset.attrs)}

        obj_embeddings = load_word_embeddings(args.emb_init, obj_words).to(device)
        attr_embeddings = load_word_embeddings(args.emb_init, attr_words).to(device)

        self.obj_embeddings = obj_embeddings.unsqueeze(0)
        self.attr_embeddings = attr_embeddings.unsqueeze(0)
        # Encoder-obj
        obj_encoder_layer = nn.TransformerEncoderLayer(d_model=args.emb_dim, nhead=args.obj_nhead)
        obj_transformer_encoder = nn.TransformerEncoder(obj_encoder_layer, num_layers=args.obj_nlayer)
        self.trans_obj = obj_transformer_encoder
        # Encoder-attr
        attr_encoder_layer = nn.TransformerEncoderLayer(d_model=args.emb_dim, nhead=args.attr_nhead)
        attr_transformer_encoder = nn.TransformerEncoder(attr_encoder_layer, num_layers=args.attr_nlayer)
        self.trans_attr = attr_transformer_encoder

    # loss
    def compute_loss(self, preds, labels):
        loss = F.cross_entropy(preds, labels)
        return loss

    def train_forward(self, x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]

        obj_feats = self.obj_head(img[:, 0, :])
        attr_feats = self.attr_head(img[:, 0, :])

        obj_embeddings = self.trans_obj(self.obj_embeddings).squeeze(0)
        attr_embeddings = self.trans_attr(self.attr_embeddings).squeeze(0)

        obj_embed = obj_embeddings.permute(1, 0)
        attr_embed = attr_embeddings.permute(1, 0)

        obj_pred = torch.matmul(obj_feats, obj_embed)
        attr_pred = torch.matmul(attr_feats, attr_embed)

        loss_obj = self.compute_loss(obj_pred, objs)
        loss_attr = self.compute_loss(attr_pred, attrs)

        loss = loss_obj + loss_attr
        return loss, None

    def val_forward(self, x):
        img = x[0]

        obj_feats = self.obj_head(img[:, 0, :])
        attr_feats = self.attr_head(img[:, 0, :])

        obj_embeddings = self.trans_obj(self.obj_embeddings).squeeze(0)
        attr_embeddings = self.trans_attr(self.attr_embeddings).squeeze(0)

        obj_embed = obj_embeddings.permute(1, 0)
        attr_embed = attr_embeddings.permute(1, 0)

        score_obj = torch.matmul(obj_feats, obj_embed)
        score_attr = torch.matmul(attr_feats, attr_embed)

        score = torch.bmm(score_attr.unsqueeze(2), score_obj.unsqueeze(1)).view(score_attr.shape[0], -1)
        scores = {}
        for itr, (attr, obj) in enumerate(self.dataset.pairs):
            attr_id, obj_id = self.dataset.attr2idx[attr], self.dataset.obj2idx[obj]
            idx = obj_id + attr_id * len(self.dataset.objs)
            scores[(attr, obj)] = score[:, idx]
        return score, scores

    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
        return loss, pred
