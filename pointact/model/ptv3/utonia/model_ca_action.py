import torch
import torch.nn as nn

try:
    import flash_attn
except ImportError:
    flash_attn = None

from .structure import Point
from .module import PointModule, PointSequential
from .model import (
    GridPooling,
    GridUnpooling,
    Embedding,
    Block,
    MLP,
    SerializedAttention,
)
from .model_ca import CABlock, PointTransformerV3CA
from .utils import offset2bincount


class GridPoolingWithAction(GridPooling):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.action_proj = nn.Linear(in_channels, out_channels)
        norm_layer = kwargs.get("norm_layer", None)
        if norm_layer is not None:
            self.action_norm = PointSequential(norm_layer(out_channels))

    def forward(self, point: Point):
        action_feat = self.action_proj(point.action_feat)
        if hasattr(self, "action_norm"):
            action_feat = self.action_norm(action_feat)
        if hasattr(self, "act"):
            action_feat = self.act(action_feat)

        point = super().forward(point)
        point.action_feat = action_feat
        return point


class GridUnpoolingWithAction(GridUnpooling):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        **kwargs,
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

        parent.action_feat = self.action_proj(point.action_feat) + self.action_proj_skip(
            parent.action_feat
        )

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
            self.patch_size = min(bincount.min().tolist(), self.patch_size_max)

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)
        patch_size_list = torch.diff(cu_seqlens)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        point_qkv = self.qkv(point.feat)[order]
        point_qkv_dtype = point_qkv.dtype
        rope_coord = point.coord[order].clone()

        point_qkv = point_qkv.reshape(-1, 3, H, C // H)
        q, k, v = point_qkv.unbind(dim=1)
        q, k = self.rope(q, k, rope_coord)
        point_qkv = torch.stack([q, k, v], dim=1)

        num_actions = point.action_feat.size(1)
        action_qkv = self.qkv(point.action_feat).reshape(
            -1, num_actions, 3, H, C // H
        )
        repeat_size = torch.div(
            bincount + self.patch_size - 1,
            self.patch_size,
            rounding_mode="trunc",
        )
        action_qkv = action_qkv.repeat_interleave(repeat_size, dim=0)

        if not self.enable_flash:
            point_qkv = point_qkv.reshape(-1, K, 3, H, C // H)
            qkv = torch.cat([action_qkv, point_qkv], dim=1)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)

            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if self.enable_rpe:
                rpe = self.rpe(self.get_rel_pos(point, order))
                attn[:, :, num_actions:, num_actions:] = (
                    attn[:, :, num_actions:, num_actions:] + rpe
                )
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2)

            action_feat = torch.split(
                feat[:, :num_actions],
                repeat_size.cpu().tolist(),
                dim=0,
            )
            action_feat = torch.stack([torch.mean(x, dim=0) for x in action_feat], dim=0)
            action_feat = action_feat.reshape(-1, num_actions, C)
            feat = feat[:, num_actions:].reshape(-1, C)
        else:
            point_qkv = torch.split(
                point_qkv,
                patch_size_list.cpu().tolist(),
                dim=0,
            )
            qkv = torch.cat(
                [
                    torch.cat([i_action_qkv, i_point_qkv], dim=0)
                    for i_point_qkv, i_action_qkv in zip(point_qkv, action_qkv)
                ],
                dim=0,
            )

            patch_size_list = patch_size_list + num_actions
            cu_seqlens = torch.cumsum(patch_size_list, dim=0).int()
            cu_seqlens = torch.cat(
                [torch.zeros(1, dtype=torch.int32, device=cu_seqlens.device), cu_seqlens],
                dim=0,
            )

            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.to(torch.bfloat16),
                cu_seqlens,
                max_seqlen=self.patch_size + num_actions,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)

            feat = torch.split(feat, patch_size_list.cpu().tolist(), dim=0)
            action_feat = torch.stack([x[:num_actions] for x in feat], dim=0)
            feat = torch.cat([x[num_actions:] for x in feat], dim=0)
            feat = feat.to(point_qkv_dtype)

            action_feat = torch.split(
                action_feat,
                repeat_size.cpu().tolist(),
                dim=0,
            )
            action_feat = torch.stack(
                [torch.mean(x, dim=0) for x in action_feat], dim=0
            )
            action_feat = action_feat.reshape(-1, num_actions, C)
            action_feat = action_feat.to(point_qkv_dtype)

        feat = feat[inverse]

        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat

        action_feat = self.proj(action_feat)
        action_feat = self.proj_drop(action_feat)
        point.action_feat = action_feat
        return point


class BlockWithAction(Block):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        rope_base=10,
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
    ):
        super().__init__(
            channels=channels,
            num_heads=num_heads,
            patch_size=patch_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            layer_scale=layer_scale,
            norm_layer=norm_layer,
            act_layer=act_layer,
            pre_norm=pre_norm,
            order_index=order_index,
            cpe_indice_key=cpe_indice_key,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            rope_base=rope_base,
            shift_coords=shift_coords,
            jitter_coords=jitter_coords,
            rescale_coords=rescale_coords,
            attn_class=SerializedAttentionWithAction,
        )

        self.action_proj = nn.Linear(channels, channels)
        self.action_norm0 = norm_layer(channels)
        self.action_norm1 = norm_layer(channels)
        self.action_norm2 = norm_layer(channels)
        self.action_mlp = MLP(
            in_channels=channels,
            hidden_channels=int(channels * mlp_ratio),
            out_channels=channels,
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)

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
    def __init__(self, channels, num_heads, apply_point_ca=True, **kwargs):
        super().__init__(
            channels,
            num_heads,
            apply_point_ca=apply_point_ca,
            **kwargs,
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
        point = super().forward(point)

        action_shortcut = point.action_feat
        if self.pre_norm:
            point.action_feat = self.action_norm1(point.action_feat)

        batch_size, num_actions, _ = point.action_feat.size()
        device = point.action_feat.device
        action_offset = torch.cumsum(
            torch.full(
                (batch_size,),
                num_actions,
                dtype=torch.long,
                device=device,
            ),
            dim=0,
        )
        point.action_feat = self.attn(
            point.action_feat.reshape(batch_size * num_actions, -1),
            point.context,
            action_offset,
            point.context_offset,
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
        return point


class PointTransformerV3CAWithAction(PointTransformerV3CA):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(3, 3, 3, 3),
        dec_channels=(108, 108, 216, 432),
        dec_num_head=(6, 6, 12, 32),
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
        traceable=True,
        mask_token=False,
        enc_mode=True,
        apply_point_ca=True,
        freeze_encoder=False,
        rope_base=10,
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
    ):
        PointModule.__init__(self)

        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.enc_mode = enc_mode
        self.shuffle_orders = shuffle_orders
        self.freeze_encoder = freeze_encoder

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.enc_mode or self.num_stages == len(dec_depths) + 1
        assert self.enc_mode or self.num_stages == len(dec_channels) + 1
        assert self.enc_mode or self.num_stages == len(dec_num_head) + 1
        assert self.enc_mode or self.num_stages == len(dec_patch_size) + 1

        ln_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=ln_layer,
            act_layer=act_layer,
            mask_token=mask_token,
        )

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
                        rope_base=rope_base,
                        shift_coords=shift_coords,
                        jitter_coords=jitter_coords,
                        rescale_coords=rescale_coords,
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
                        apply_point_ca=apply_point_ca,
                    ),
                    name=f"ca_block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        if not self.enc_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    GridUnpoolingWithAction(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        traceable=traceable,
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
                            rope_base=rope_base,
                            shift_coords=shift_coords,
                            jitter_coords=jitter_coords,
                            rescale_coords=rescale_coords,
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
                            apply_point_ca=apply_point_ca,
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
        point = Point(data_dict)
        point = self.embedding(point)

        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.enc(point)
        if not self.enc_mode:
            point = self.dec(point)
        return point


def test(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    npoints = torch.tensor([6, 7], device=device)
    offset = torch.cumsum(npoints, dim=0).long()
    batch = torch.arange(len(npoints), device=device).repeat_interleave(npoints)
    feat = torch.randn(int(offset[-1]), 6, device=device)
    context_lens = torch.tensor([3, 2], device=device)
    context_offset = torch.cumsum(context_lens, dim=0).long()
    context = torch.randn(int(context_offset[-1]), 8, device=device)
    action_feat = torch.randn(len(npoints), 2, 6, device=device)

    model = PointTransformerV3CAWithAction(
        in_channels=6,
        order=("z", "z-trans"),
        enc_depths=(1, 1, 1, 1, 1),
        enc_channels=(6, 12, 24, 48, 96),
        enc_num_head=(1, 2, 4, 8, 16),
        enc_patch_size=(4, 4, 4, 4, 4),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(12, 12, 24, 48),
        dec_num_head=(2, 2, 4, 8),
        dec_patch_size=(4, 4, 4, 4),
        ctx_channels=8,
        enable_flash=False,
        enc_mode=True,
        shuffle_orders=False,
    ).to(device)
    model.eval()

    data_dict = {
        "coord": feat[:, :3],
        "grid_size": 0.05,
        "offset": offset,
        "batch": batch,
        "feat": feat,
        "context": context,
        "context_offset": context_offset,
        "action_feat": action_feat,
    }
    with torch.no_grad():
        output = model(data_dict)
    print(
        "PointTransformerV3CAWithAction test:",
        output.feat.shape,
        output.action_feat.shape,
    )
    return output


if __name__ == "__main__":
    test()
