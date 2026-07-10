import torch
import torch.nn as nn

try:
    import flash_attn
except:
    print('No flash attn')

from pointact.model.ptv3.concerto.structure import Point
from pointact.model.ptv3.concerto.module import PointModule, PointSequential
from pointact.model.ptv3.concerto.model import (
    GridPooling, GridUnpooling, Embedding, Block, MLP, SerializedAttention
)
from pointact.model.ptv3.concerto.model_ca import (
    CABlock, PointTransformerV3CA,    
)
from pointact.model.ptv3.concerto.utils import offset2bincount


class GridPoolingWithAction(GridPooling):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.action_proj = nn.Linear(in_channels, out_channels)
        norm_layer = kwargs.get("norm_layer", None)
        if norm_layer is not None:
            self.action_norm = PointSequential(norm_layer(out_channels))
        
    def forward(self, point: Point):
        point = super().forward(point)

        action_feat = self.action_proj(point.action_feat)
        if self.norm is not None:
            action_feat = self.action_norm(action_feat)
        if self.act is not None:
            action_feat = self.act(action_feat)

        point.action_feat = action_feat
        return point
    
class GridUnpoolingWithAction(GridUnpooling):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels, 
        **kwargs
    ):
        super().__init__(in_channels, skip_channels, out_channels, **kwargs)

        self.action_proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.action_proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        norm_layer = kwargs.get("norm_layer", None)
        if norm_layer is not None:
            self.action_proj.add(norm_layer(out_channels))
            self.action_proj_skip.add(norm_layer(out_channels))
        act_layer = kwargs.get("act_layer", None)
        if act_layer is not None:
            self.action_proj.add(act_layer())
            self.action_proj_skip.add(act_layer())

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pooling_inverse
        feat = point.feat

        parent.action_feat = self.action_proj(point.action_feat) + \
                             self.action_proj_skip(parent.action_feat)

        parent = self.proj_skip(parent)
        parent.feat = parent.feat + self.proj(point).feat[inverse]
        parent.sparse_conv_feat = parent.sparse_conv_feat.replace_feature(parent.feat)

        if self.traceable:
            point.feat = feat
            parent["unpooling_parent"] = point
        return parent
    
class SerializedAttentionWithAction(SerializedAttention):
    def forward(self, point):
        bincount = offset2bincount(point.offset)

        if not self.enable_flash:
            self.patch_size = min(
                bincount.min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        # print('point feat', point.feat.size(), point.action_feat.size())
        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)
        # print(point.feat.size(), point.offset)
        # print(pad.size(), unpad.size(), cu_seqlens)
        patch_size_list = torch.diff(cu_seqlens)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        # Add action tokens: (batch, num_actions, D)
        num_actions = point.action_feat.size(1)
        action_qkv = self.qkv(point.action_feat) # (batch, num_actions, 3 * channels)
        repeat_size = torch.div(
            bincount + self.patch_size - 1,
            self.patch_size,
            rounding_mode="trunc",
        )
        action_qkv = action_qkv.repeat_interleave(
            repeat_size, dim=0
        )    # (batch*repeat, num_actions, 3*channels)
        # print(action_qkv.size(), action_qkv)

        if not self.enable_flash:
            # the previous padding only pads point when num of points larger than patch_size
            # but the patch size is set as the min npoints in batch
            qkv = torch.cat(
                [
                    action_qkv.reshape(-1, num_actions, 3, H, C // H),
                    qkv.reshape(-1, K, 3, H, C // H)
                ],
                dim=1
            )
            # encode and reshape qkv: (N', num_actions+K, 3, H, C') => (3, N', H, num_actions+K, C')
            q, k, v = (
                qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            # print(attn.size(), attn.max(), torch.norm(q), torch.norm(k))
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2)   # (N', H, K, C) -> (N', K, H, C)
            
            action_feat = torch.split(
                feat[:, :num_actions], repeat_size.data.cpu().numpy().tolist(), dim=0
            )
            action_feat = torch.stack(
                [torch.mean(x, dim=0) for x in action_feat], 0
            ).reshape(-1, num_actions, C)
            # action_feat = torch.stack(
            #     [torch.max(x, dim=0)[0] for x in action_feat], 0
            # ).reshape(-1, num_actions, C)
            feat = feat[:, num_actions:].reshape(-1, C)

        else:
            # # This is correct only when #points are larger patch_size
            # action_qkv = action_qkv.reshape(-1, num_actions, 3, H, C // H)
            # qkv = qkv.reshape(-1, K, 3, H, C // H)  # (N'*K=#points, 3, H, C'), K=patch_size
            # qkv = torch.cat([action_qkv, qkv], dim=1).reshape(-1, 3, H, C // H)

            # (npoints, 3*channels) -> [each item [patch_size, C]]
            qkv = torch.split(qkv, patch_size_list.data.cpu().numpy().tolist(), dim=0)
            qkv = [
                torch.cat([i_action_qkv, i_qkv], 0) \
                    for i_qkv, i_action_qkv in zip(qkv, action_qkv)
            ]
            qkv = torch.cat(qkv, 0).reshape(-1, 3, H, C // H)

            patch_size_list = patch_size_list + num_actions
            cu_seqlens = torch.cumsum(patch_size_list, 0).int()
            # cu_seqlens = torch.cumsum(torch.diff(cu_seqlens) + num_actions, 0).int()
            cu_seqlens = torch.cat([torch.zeros(1).int().to(cu_seqlens.device), cu_seqlens], dim=0)

            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half(), #.reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size+num_actions,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            # print(feat.size())

            feat = torch.split(
                feat, patch_size_list.data.cpu().numpy().tolist(), dim=0
            )
            action_feat = torch.stack([x[:num_actions] for x in feat], 0)
            feat = torch.cat([x[num_actions:] for x in feat], 0)
            feat = feat.to(qkv.dtype)

            action_feat = torch.split(
                action_feat, repeat_size.data.cpu().numpy().tolist(), dim=0
            )
            action_feat = torch.stack([torch.mean(x, dim=0) for x in action_feat], 0)
            # action_feat = torch.stack([torch.max(x, dim=0)[0] for x in action_feat], 0)
            action_feat = action_feat.reshape(-1, num_actions, C)
            action_feat = action_feat.to(qkv.dtype)
            # print('final', feat.size(), action_feat.size())

        # print(inverse)
        feat = feat[inverse]
        # print(feat.size())

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat

        action_feat = self.proj(action_feat)
        action_feat = self.proj_drop(action_feat)
        point.action_feat = action_feat
        # print('SerializedAttentionWithAction', feat.size(), action_feat.size())
        return point
 
class BlockWithAction(Block):
    def __init__(
        self, channels, num_heads, **kwargs
    ):
        super().__init__(
            channels, 
            num_heads, 
            attn_class=SerializedAttentionWithAction, 
            **kwargs
        )

        norm_layer = kwargs["norm_layer"]
        self.action_proj = nn.Linear(channels, channels)
        self.action_norm0 = norm_layer(channels)
        self.action_norm1 = norm_layer(channels)
        self.action_norm2 = norm_layer(channels)
        self.action_mlp = MLP(
            in_channels=channels,
            hidden_channels=int(channels * kwargs["mlp_ratio"]),
            out_channels=channels,
            act_layer=kwargs["act_layer"],
            drop=kwargs["proj_drop"],
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)

        # action
        action_feat = point.action_feat
        point.action_feat = action_feat + self.action_norm0(
            self.action_proj(action_feat)
        )
        action_shortcut = point.action_feat
        if self.pre_norm:
            point.action_feat = self.action_norm1(action_feat)

        point = self.drop_path(self.ls1(self.attn(point)))

        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.ls2(self.mlp(point)))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)

        # action
        point.action_feat = action_shortcut + point.action_feat
        if not self.pre_norm:
            point.action_feat = self.action_norm1(point.action_feat)
        action_shortcut = point.action_feat
        if self.pre_norm:
            point.action_feat = self.action_norm2(point.action_feat)
        point.action_feat = self.drop_path(self.action_mlp(point.action_feat))
        point.action_feat = action_shortcut + point.action_feat
        if not self.pre_norm:
            point.action_feat = self.action_norm2(point.action_feat)

        return point

class CABlockWithAction(CABlock):
    def __init__(
        self, channels, num_heads, apply_point_ca=True, **kwargs
    ):
        super().__init__(
            channels, num_heads, apply_point_ca=apply_point_ca, **kwargs
        )
        self.action_norm1 = kwargs["norm_layer"](channels)
        self.action_norm2 = kwargs["norm_layer"](channels)
        self.action_mlp = MLP(
            in_channels=channels,
            hidden_channels=int(channels * kwargs["mlp_ratio"]),
            out_channels=channels,
            act_layer=kwargs["act_layer"],
            drop=kwargs["proj_drop"],
        )

    def forward(self, point: Point):
        # point cross attention
        point = super().forward(point)
    
        # action corss attention
        action_shortcut = point.action_feat
        if self.pre_norm:
            point.action_feat = self.action_norm1(point.action_feat)
        
        batch_size, num_actions, _ = point.action_feat.size()
        device = point.action_feat.device
        action_offset = torch.cumsum(
            torch.LongTensor([num_actions] * batch_size).to(device), 0
        )
        point.action_feat = self.attn(
            point.action_feat.reshape(batch_size * num_actions, -1), 
            point.context, 
            action_offset, 
            point.context_offset
        ).reshape(batch_size, num_actions, -1)
        point.action_feat = action_shortcut + point.action_feat
        if not self.pre_norm:
            point.action_feat = self.action_norm1(point.action_feat)

        action_shortcut = point.action_feat
        if self.pre_norm:
            point.action_feat = self.action_norm2(point.action_feat)
        point.action_feat = self.action_mlp(point.action_feat)
        point.action_feat = action_shortcut + point.action_feat
        if not self.pre_norm:
            point.action_feat = self.action_norm2(point.action_feat)

        # print('CABlockWithAction', point.feat.size(), point.action_feat.size())
        return point


class PointTransformerV3CAWithAction(PointTransformerV3CA):
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
        apply_point_ca=True,
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
                    GridPoolingWithAction(
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
                    BlockWithAction(
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
                    CABlockWithAction(
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
                        apply_point_ca=apply_point_ca
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
                    GridUnpoolingWithAction(
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
                        BlockWithAction(
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
                        CABlockWithAction(
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
                            apply_point_ca=apply_point_ca
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
    # test
    import time

    torch.manual_seed(0)

    model = PointTransformerV3CAWithAction(
        enc_patch_size=(5, 5, 5, 5, 5),
        dec_patch_size=(5, 5, 5, 5),
        shuffle_orders=False,
        enable_flash=True,
        enc_mode=False,
        apply_point_ca=True
    ).eval().cuda()

    pc_fts = torch.rand(10, 6)
    voxel_size = 0.01
    point_offset = torch.LongTensor([4, 10])
    point_batch_idxs = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    context = torch.rand(5, 256)
    context_offset = torch.LongTensor([3, 5])
    action_feat = torch.rand(2, 3, 32)

    data_dict = {
        'coord': pc_fts[:, :3],
        'grid_size': voxel_size,
        'offset': point_offset,
        'batch': point_batch_idxs,
        'feat': pc_fts,
        'context': context,
        'context_offset': context_offset,
        'action_feat': action_feat,
    }
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            data_dict[k] = v.cuda()

    # print(data_dict)
    start_time = time.time()
    outs = model(data_dict)
    duration = time.time() - start_time
    # print(outs)
    print(outs.feat.size(), outs.action_feat.size())
    print(outs.feat, outs.action_feat)

    print('time', duration)
