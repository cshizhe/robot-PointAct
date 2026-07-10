import torch
import torch.nn as nn


def get_ptv3_model_cls(ptv3_backend, with_action=False):
    ptv3_backend = (ptv3_backend or "concerto").lower()
    if ptv3_backend == "concerto":
        if with_action:
            from pointact.model.ptv3.concerto.model_ca_action import (
                PointTransformerV3CAWithAction,
            )

            return PointTransformerV3CAWithAction
        from pointact.model.ptv3.concerto.model_ca import PointTransformerV3CA

        return PointTransformerV3CA
    if ptv3_backend == "utonia":
        if with_action:
            from pointact.model.ptv3.utonia.model_ca_action import (
                PointTransformerV3CAWithAction,
            )

            return PointTransformerV3CAWithAction
        from pointact.model.ptv3.utonia.model_ca import PointTransformerV3CA

        return PointTransformerV3CA
    raise ValueError(
        f"Unsupported ptv3_backend={ptv3_backend!r}; expected 'concerto' or 'utonia'."
    )


class PointTransformerUnet(nn.Module):
    '''Point Transformer v3'''

    def __init__(
        self, 
        input_size, 
        ctx_embed_size, 
        voxel_size=0.01,
        enc_channels=(32, 64, 128, 256, 512),
        enc_depths=(1, 1, 1, 1, 1),
        enc_num_head=(2, 4, 8, 16, 32),
        dec_channels = (64, 64, 128, 256),
        dec_depths=(1, 1, 1, 1),
        dec_num_head=(4, 4, 8, 16),
        patch_size=256,
        enc_mode=False,
        ptv3_backend="concerto",
    ):
        super().__init__()

        ptv3_model_cls = get_ptv3_model_cls(ptv3_backend, with_action=False)
        self.ptv3_model = ptv3_model_cls(
            in_channels=input_size,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=[patch_size] * len(enc_depths),
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=[patch_size] * len(dec_depths),
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
            enc_mode=enc_mode,
        )
        if enc_mode:
            self.output_size = enc_channels[-1]
        else:
            self.output_size = dec_channels[0]
        self.voxel_size = voxel_size

    def prepare_ptv3_batch(self, pc_fts, npoints_in_batch, ctx_embeds, ctx_lens):
        device = pc_fts.device

        point_offset = torch.cumsum(npoints_in_batch, dim=0).long()
        point_batch_idxs = torch.arange(
            len(npoints_in_batch), device=device, dtype=torch.long
        ).repeat_interleave(npoints_in_batch)
        outs = {
            'coord': pc_fts[:, :3],
            'grid_size': self.voxel_size,
            'offset': point_offset,
            'batch': point_batch_idxs,
            'feat': pc_fts,
        }
        # encode context for each point cloud        
        outs['context'] = torch.cat(
            [ctx_embed[:ctx_len] for ctx_embed, ctx_len in zip(ctx_embeds, ctx_lens)],
            dim=0
        )
        outs['context_offset'] = torch.cumsum(ctx_lens, dim=0).to(device)

        return outs
        
    def forward(self, pc_fts, npoints_in_batch, ctx_embeds, ctx_lens):
        ptv3_batch = self.prepare_ptv3_batch(
            pc_fts, npoints_in_batch, ctx_embeds, ctx_lens
        )
        
        point_outs = self.ptv3_model(ptv3_batch)
        
        return point_outs.feat, point_outs.coord, point_outs.offset



class PointTransformerUnetWithAction(nn.Module):
    '''Point Transformer v3'''

    def __init__(
        self, 
        input_size, 
        ctx_embed_size, 
        voxel_size=0.01,
        enc_channels=(64, 128, 256, 512, 768),
        enc_depths=(1, 1, 1, 1, 1),
        enc_num_head=(2, 4, 8, 16, 32),
        dec_channels=(128, 128, 256, 512),
        dec_depths=(1, 1, 1, 1),
        dec_num_head=(4, 4, 8, 16),
        enc_mode=False,
        patch_size=128,
        apply_point_ca=True,
        ptv3_backend="concerto",
    ):
        super().__init__()

        ptv3_model_cls = get_ptv3_model_cls(ptv3_backend, with_action=True)
        self.ptv3_model = ptv3_model_cls(
            in_channels=input_size,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=[patch_size] * len(enc_depths),
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=[patch_size] * len(dec_depths),
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
            enc_mode=enc_mode,
            apply_point_ca=apply_point_ca
        )
        self.enc_channels = enc_channels
        if enc_mode:
            self.output_size = enc_channels[-1]
        else:
            self.output_size = dec_channels[0]
        self.voxel_size = voxel_size

    def prepare_ptv3_batch(
        self, pc_fts, npoints_in_batch, ctx_embeds, ctx_lens, action_feat, 
        time_embeds=None,
    ):
        device = pc_fts.device

        point_offset = torch.cumsum(npoints_in_batch, dim=0).long()
        point_batch_idxs = torch.arange(
            len(npoints_in_batch), device=device, dtype=torch.long
        ).repeat_interleave(npoints_in_batch)
        outs = {
            'coord': pc_fts[:, :3],
            'grid_size': self.voxel_size,
            'offset': point_offset,
            'batch': point_batch_idxs,
            'feat': pc_fts,
        }
        # encode context for each point cloud        
        outs['context'] = torch.cat(
            [ctx_embed[:ctx_len] for ctx_embed, ctx_len in zip(ctx_embeds, ctx_lens)],
            dim=0
        )
        outs['context_offset'] = torch.cumsum(ctx_lens, dim=0).to(device)
        outs['action_feat'] = action_feat
        if time_embeds is not None:
            outs['time_embeds'] = time_embeds

        return outs
        
    def forward(
        self, pc_fts, npoints_in_batch, ctx_embeds, ctx_lens, action_embeds, 
        time_embeds=None
    ):

        ptv3_batch = self.prepare_ptv3_batch(
            pc_fts, npoints_in_batch, ctx_embeds, ctx_lens,
            action_embeds, time_embeds=time_embeds
        )
        
        # print(self.ptv3_model)
        # for k, v in ptv3_batch.items():
        #     if isinstance(v, torch.Tensor):
        #         print(k, v.size())
        point_outs = self.ptv3_model(ptv3_batch)

        action_out_embeds = point_outs.action_feat
        
        return point_outs.feat, point_outs.coord, point_outs.offset, action_out_embeds
