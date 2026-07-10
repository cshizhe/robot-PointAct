import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

try:
    import flash_attn
except:
    print('No flash attn')

from pointact.model.ptv3.concerto.structure import Point
from pointact.model.ptv3.concerto.module import PointModule, PointSequential
from pointact.model.ptv3.concerto.model import (
    PointTransformerV3, GridPooling, GridUnpooling, Embedding, Block, MLP
)
from pointact.model.ptv3.concerto.utils import (
    offset2bincount, gen_seq_masks,
)


class CrossAttention(PointModule):
    def __init__(
        self, 
        channels, 
        num_heads, 
        kv_channels=None, 
        attn_drop=0, 
        proj_drop=0, 
        qk_norm=False, 
        enable_flash=True
    ):
        super().__init__()
        if kv_channels is None:
            kv_channels = channels
        assert channels % num_heads == 0

        self.q = nn.Linear(channels, channels, bias=True)
        self.kv = nn.Linear(kv_channels, channels * 2, bias=True)
        self.attn_drop = attn_drop

        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qk_norm = qk_norm
        self.enable_flash = enable_flash

        # TODO: eps should be 1 / 65530 if using fp16 (eps=1e-6)
        self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6) if self.qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6) if self.qk_norm else nn.Identity()

    def forward(
        self, 
        query: torch.Tensor, 
        context: torch.Tensor,
        query_offset: torch.Tensor,
        context_offset: torch.Tensor,
    ):
        device = query.device

        q = self.q(query).view(-1, self.num_heads, self.head_dim)
        kv = self.kv(context).view(-1, 2, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(kv[:, 0])
        kv = torch.stack([k, kv[:, 1]], dim=1)

        if self.enable_flash:
            cu_seqlens_q = torch.cat([torch.zeros(1).int().to(device), query_offset.int()], dim=0)
            cu_seqlens_k = torch.cat([torch.zeros(1).int().to(device), context_offset.int()], dim=0)
            max_seqlen_q = offset2bincount(query_offset).max()
            max_seqlen_k = offset2bincount(context_offset).max()

            feat = flash_attn.flash_attn_varlen_kvpacked_func(
                q.half(), 
                kv.half(), 
                cu_seqlens_q, 
                cu_seqlens_k, 
                max_seqlen_q, 
                max_seqlen_k,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale
            ).reshape(-1, self.channels)
            feat = feat.to(q.dtype)
        else:
            # q: (#all points, #heads, #dim)
            # kv: (#all words, k/v, #heads, #dim)
            npoints_in_batch = offset2bincount(query_offset).data.cpu().numpy().tolist()
            nwords_in_batch = offset2bincount(context_offset).data.cpu().numpy().tolist()
            word_padded_masks = torch.from_numpy(
                gen_seq_masks(nwords_in_batch)
            ).to(q.device).logical_not()

            q_pad = pad_sequence(
                torch.split(q, npoints_in_batch, dim=0),
                batch_first=True,
                padding_value=0,
            )
            kv_pad = pad_sequence(
                torch.split(kv, nwords_in_batch),
                batch_first=True,
                padding_value=0,
            )
            # q_pad: (batch_size, #points, #heads, #dim)
            # kv_pad: (batch_size, #words, k/v, #heads, #dim)
            logits = torch.einsum('bphd,bwhd->bpwh', q_pad, kv_pad[:, :, 0]) * self.scale
            logits.masked_fill_(word_padded_masks.unsqueeze(1).unsqueeze(-1), -1e4)
            attn_probs = torch.softmax(logits, dim=2)
            feat = torch.einsum('bpwh,bwhd->bphd', attn_probs, kv_pad[:, :, 1])
            feat = torch.cat([ft[:npoints_in_batch[i]] for i, ft in enumerate(feat)], 0)
            feat = feat.reshape(-1, self.channels).float()

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        return feat
    

class CABlock(PointModule):
    def __init__(
        self, 
        channels, 
        num_heads, 
        kv_channels=None, 
        attn_drop=0.0, 
        proj_drop=0.0,
        mlp_ratio=4.0, 
        qk_norm=True,
        norm_layer=nn.LayerNorm, 
        act_layer=nn.GELU, 
        pre_norm=True,
        enable_flash=True, 
        attn_class=CrossAttention, 
        apply_point_ca=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        self.apply_point_ca = apply_point_ca

        if self.apply_point_ca:
            self.norm1 = PointSequential(norm_layer(channels))
            self.norm2 = PointSequential(norm_layer(channels))
            self.mlp = PointSequential(
                MLP(
                    in_channels=channels,
                    hidden_channels=int(channels * mlp_ratio),
                    out_channels=channels,
                    act_layer=act_layer,
                    drop=proj_drop,
                )
            )

        self.attn = attn_class(
            channels=channels,
            num_heads=num_heads,
            kv_channels=kv_channels,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qk_norm=qk_norm,
            enable_flash=enable_flash,
        )

    def forward(self, point: Point):
        if self.apply_point_ca:
            shortcut = point.feat
            if self.pre_norm:
                point = self.norm1(point)
            point.feat = self.attn(
                point.feat, point.context, point.offset, point.context_offset
            )
            point.feat = shortcut + point.feat
            if not self.pre_norm:
                point = self.norm1(point)

            shortcut = point.feat
            if self.pre_norm:
                point = self.norm2(point)
            point = self.mlp(point)
            point.feat = shortcut + point.feat
            if not self.pre_norm:
                point = self.norm2(point)
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)

        return point


class PointTransformerV3CA(PointTransformerV3):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        ctx_channels=256,
        qkv_bias=True,
        qk_scale=None,
        qk_norm=True,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        layer_scale=None,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=False,
        freeze_encoder=False,
    ):
        PointModule.__init__(self)

        self.num_stages = len(enc_depths)
        self.num_dec_stages = len(dec_depths)
        self.order = [order] if isinstance(order, str) else order
        self.enc_mode = enc_mode
        self.shuffle_orders = shuffle_orders
        self.freeze_encoder = freeze_encoder

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.enc_mode or self.num_dec_stages == len(dec_depths)
        assert self.enc_mode or self.num_dec_stages == len(dec_channels)
        assert self.enc_mode or self.num_dec_stages == len(dec_num_head)
        assert self.enc_mode or self.num_dec_stages == len(dec_patch_size)

        # normalization layer
        ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=ln_layer,
            act_layer=act_layer,
            mask_token=mask_token,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    GridPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        layer_scale=layer_scale,
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
                enc.add(
                    CABlock(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        kv_channels=ctx_channels,
                        mlp_ratio=mlp_ratio,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        qk_norm=qk_norm,
                        pre_norm=pre_norm,
                        enable_flash=enable_flash,
                    ),
                    name=f"ca_block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.enc_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_dec_stages)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    GridUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s + self.num_stages - self.num_dec_stages - 1],
                        out_channels=dec_channels[s],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        traceable=traceable
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            layer_scale=layer_scale,
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                    dec.add(
                        CABlock(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            kv_channels=ctx_channels,
                            mlp_ratio=mlp_ratio,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            qk_norm=qk_norm,
                            enable_flash=enable_flash,
                        ),
                        name=f"ca_block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

        if self.freeze_encoder:
            for p in self.embedding.parameters():
                p.requires_grad = False
            for p in self.enc.parameters():
                p.requires_grad = False
        self.apply(self._init_weights)

    def forward(self, data_dict):
        """
        A data_dict is a dictionary containing properties of a batched point cloud.
        It should contain the following properties for PTv3:
        1. "feat": feature of point cloud
        2. "grid_coord": discrete coordinate after grid sampling (voxelization) or "coord" + "grid_size"
        3. "offset" or "batch": https://github.com/Pointcept/Pointcept?tab=readme-ov-file#offset
        """
        point = Point(data_dict)
        point = self.embedding(point)

        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.enc(point)
        if not self.enc_mode:
            point = self.dec(point)
        return point
    

if __name__ == '__main__':
    enc_channels = (64, 128, 256, 512, 768)
    dec_channels = (128, 128, 256, 512)
    patch_size = 128
    ctx_embed_size = 256

    model = PointTransformerV3CA(
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(1, 1, 1, 1, 1),
        enc_channels=enc_channels,
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(patch_size, patch_size, patch_size, patch_size, patch_size),
        dec_depths=(1, 1, 1, 1),
        dec_channels=dec_channels,
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(patch_size, patch_size, patch_size, patch_size),
        mlp_ratio=4,
        ctx_channels=ctx_embed_size,
        qkv_bias=True,
        qk_norm=True,
        attn_drop=0.1,
        proj_drop=0.1,
        drop_path=0.,
        pre_norm=True,
        shuffle_orders=True,
        enable_flash=True,
        enc_mode=True,
    ).cuda()

    print(model)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_parameters / 1e6:.2f}M")
