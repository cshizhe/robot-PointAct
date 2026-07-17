# from scipy.spatial.transform import Rotation as R
# import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

def get_best_pos_from_disc_pos(
    disc_pos_prob, xyz, pos_bin_size=0.01, pos_bins=100
):
    """Args:
        disc_pos_prob: (3, npoints*pos_bins)
        xyz: (npoints, 3)
    """
    device = xyz.device
    half_pos_bins = pos_bins // 2
    # (pos_bins, )
    shift = torch.arange(-half_pos_bins, half_pos_bins).to(device) * pos_bin_size 
    # (npoints, 3, pos_bins)
    cands_pos = shift.unsqueeze(0).repeat(3, 1).unsqueeze(0) + xyz.unsqueeze(-1)

    cands_pos = einops.rearrange(cands_pos, "n c b -> c (n b)") # (3, npoints*pos_bins)
    idxs = torch.argmax(disc_pos_prob, -1)
    best_pos = cands_pos[torch.arange(3).long(), idxs]

    return best_pos

def get_disc_gt_pos_prob(
    xyz, gt_pos, pos_bin_size=0.01, pos_bins=100, heatmap_type="plain"
):
    """
    Args:
        xyz: torch.Tensor, (npoints, 3)
        gt_pos: torch.Tensor, (3, )
        heatmap_type:
            - plain: the same prob for all voxels with distance to gt_pos within pos_bin_size
            - dist: prob for each voxel is propotional to its distance to gt_pos
    Return:
        disc_pos_prob: (3, npoints * pos_bins)
    """
    assert pos_bins % 2 == 0
    half_pos_bins = pos_bins // 2
    device = xyz.device
    # (pos_bins, )
    shift = torch.arange(-half_pos_bins, half_pos_bins, device=device, dtype=xyz.dtype) * pos_bin_size
    # (npoints, 3, pos_bins)
    cands_pos = shift.unsqueeze(0).repeat(3, 1).unsqueeze(0) + xyz.unsqueeze(-1)
    # (npoints, 3, pos_bins)
    dists = torch.abs(gt_pos.unsqueeze(0).unsqueeze(-1) - cands_pos)
    dists = einops.rearrange(dists, "n c b -> c (n b)") # (3, npoints*pos_bins)
    
    if heatmap_type == "plain":
        disc_pos_prob = torch.zeros_like(dists)
        # bins close to the target within one bin size
        disc_pos_prob[dists < pos_bin_size] = 1
        for i in range(3):
            if torch.sum(disc_pos_prob[i]) == 0:
                print("zero")
                disc_pos_prob[i, torch.argmin(dists[i])] = 1
        disc_pos_prob = disc_pos_prob / torch.sum(disc_pos_prob, -1).unsqueeze(-1)
    else:
        disc_pos_prob = 1 / dists.clamp_min(1e-4)
        # disc_pos_prob[dists > 2 * pos_bin_size] = 0
        disc_pos_prob[dists > pos_bin_size] = 0
        for i in range(3):
            if torch.sum(disc_pos_prob[i]) == 0:
                disc_pos_prob[i, torch.argmin(dists[i])] = 1
        disc_pos_prob = disc_pos_prob / torch.sum(disc_pos_prob, -1).unsqueeze(-1)
    
    return disc_pos_prob


class PointMoEClassificationActionHeadBase(nn.Module):
    """
    The action chunk embeds will be concatenated with point embeds for position prediction
    """
    def __init__(
        self,
        hidden_size, 
        action_chunk_size, 
        dropout=0, 
        euler_resolution=5, 
        pos_bin_size=0.01, 
        pos_bins=100, 
        pos_heatmap_type="plain",
    ) -> None:
        super().__init__()
        # action: pos_bins * 3 on each axis, euler_bins * 3, openness 1d
        self.hidden_size = hidden_size
        self.action_chunk_size = action_chunk_size
        self.dropout = dropout

        self.pos_bin_size = pos_bin_size
        self.pos_bins = pos_bins
        self.pos_heatmap_type = pos_heatmap_type    # plan, dist
        self.euler_resolution = euler_resolution * torch.pi / 180
        self.euler_bins = 360 // euler_resolution

        self.build_model()

    def build_model(self):
        raise NotImplementedError("implement build_model")
    
    def compute_continuous_action_from_discrete_bins(
        self, xt, xr, xo, point_coords, npoints_in_batch_list
    ):
        batch_size = len(npoints_in_batch_list)
            
        # [(action_chunk_size, 3, npoints, pos_bins)]
        pred_pos_logit_list = torch.split(xt, npoints_in_batch_list, dim=2)
        # [(npoints, 3)]
        point_coords_list = torch.split(point_coords, npoints_in_batch_list, dim=0)

        cont_pred_pos = []
        for i in range(batch_size):
            cont_pred_pos.append([])
            for j in range(self.action_chunk_size):
                pred_pos_prob = torch.softmax(
                    pred_pos_logit_list[i][j].reshape(3, -1), dim=-1
                )
                cont_pred_pos[-1].append(
                    get_best_pos_from_disc_pos(
                        pred_pos_prob, 
                        point_coords_list[i], 
                        pos_bin_size=self.pos_bin_size, 
                        pos_bins=self.pos_bins, 
                    )
                )
            # (action_chunk_size, 3)
            cont_pred_pos[-1] = torch.stack(cont_pred_pos[-1], 0)
        # (batch_size, action_chunk_size, 3)
        cont_pred_pos = torch.stack(cont_pred_pos, 0)

        # (batch_size, action_chunk_size, 3)
        pred_rot = torch.argmax(xr, dim=2) * self.euler_resolution

        pred_actions = torch.cat(
            [cont_pred_pos, pred_rot, torch.sigmoid(xo.unsqueeze(-1))], dim=-1
        )
        return pred_actions
    
    def prepare_classification_labels(
        self, tgt_actions, point_coords_list,
    ):
        """Ensure the tgt_action is the absolute position and rotation (not normalized)
        Args:
            tgt_actions: (batch_size, action_chunk_size, dim_actions=7)
        """
        batch_size, action_chunk_size, _ = tgt_actions.size()

        tgt_pos = tgt_actions[:, :, :3]
        tgt_rot = tgt_actions[:, :, 3:6]
        tgt_open = tgt_actions[:, :, 6]

        tgt_disc_prob_list = []
        for i in range(batch_size):
            tgt_disc_prob_list.append([])
            for j in range(action_chunk_size):
                tgt_disc_prob_list[-1].append(get_disc_gt_pos_prob(
                    point_coords_list[i], tgt_pos[i, j], pos_bin_size=self.pos_bin_size,
                    pos_bins=self.pos_bins, heatmap_type=self.pos_heatmap_type,
                ))
            # (action_chunk_size, 3, npoints*pos_bins)
            tgt_disc_prob_list[-1] = torch.stack(tgt_disc_prob_list[-1], 0)
        
        # (batch_size, traj_len, 3) long
        tgt_rot = tgt_rot % (2 * torch.pi) # convert to [0, 2pi]
        tgt_rot = torch.round(tgt_rot / self.euler_resolution).long()
        tgt_rot[tgt_rot == self.euler_bins] = 0

        return tgt_disc_prob_list, tgt_rot, tgt_open

    def compute_loss(
        self, pred_actions, tgt_actions, action_masks, 
        npoints_in_batch, point_coords,
    ):
        batch_size = tgt_actions.size(0)
        npoints_in_batch = npoints_in_batch.data.cpu().numpy().tolist()
        point_coords_list = torch.split(point_coords, npoints_in_batch, dim=0)
        
        pred_pos, pred_rot, pred_open = pred_actions
        tgt_pos, tgt_rot, tgt_open = self.prepare_classification_labels(
            tgt_actions, point_coords_list, 
        )

        # position loss
        # [(action_chunk_size, 3, #npoints, pos_bins)]
        pred_pos_list = torch.split(pred_pos, npoints_in_batch, dim=2)
        pos_loss = 0
        for i in range(batch_size):
            # (action_chunk_size, 3)
            mask = action_masks[i].unsqueeze(1).expand(-1, 3).reshape(-1)
            pos_loss += torch.sum(F.cross_entropy(
                einops.rearrange(pred_pos_list[i], "t c n b -> (t c) (n b)"), 
                einops.rearrange(tgt_pos[i], "t c b -> (t c) b"), 
                reduction="none"
            ) * mask) / mask.sum()
        pos_loss /= batch_size

        # rotation loss
        rot_loss = F.cross_entropy(
            einops.rearrange(pred_rot, "n t d c -> (n t c) d"), 
            einops.rearrange(tgt_rot, "n t c -> (n t c)"), 
            reduction="none"
        )
        rot_loss = einops.rearrange(rot_loss, "(n t c) -> n t c", c=3, n=batch_size) 
        rot_loss = torch.sum(rot_loss * action_masks.unsqueeze(-1)) / action_masks.sum()

        # openness state loss
        open_loss = F.binary_cross_entropy_with_logits(
            pred_open, tgt_open, reduction="none"
        ) 
        open_loss = (open_loss * action_masks).sum() / action_masks.sum()

        total_loss = pos_loss + rot_loss + open_loss
        
        return total_loss, (pos_loss, rot_loss, open_loss)
 

class PointMoEClassificationMLPActionHead(PointMoEClassificationActionHeadBase):

    def build_model(self):
        self.traj_embedding = nn.Embedding(
            self.action_chunk_size, 
            self.hidden_size
        )
        
        # use MLP layers for action prediction
        self.heatmap_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LeakyReLU(0.02),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 3 * self.pos_bins)
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LeakyReLU(0.02),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.euler_bins * 3 + 1)
        )
    
    def forward(
        self, point_embeds, point_coords, npoints_in_batch, return_cont_actions=False
    ):
        """
        Args:
            point_embeds: (# all points, dim)
            point_coords: (# all points, 3)
            npoints_in_batch: (batch_size, )
        Return:
            pred_actions: (batch, action_chunk_size, dim_actions)
        """ 
        device = point_embeds.device
        npoints_in_batch_list = npoints_in_batch.data.cpu().numpy().tolist()

        point_embeds = point_embeds.unsqueeze(1).expand(-1, self.action_chunk_size, -1)

        traj_embeds = self.traj_embedding(torch.arange(self.action_chunk_size).to(device))
        # (#points, action_chunk_size, dim)
        point_embeds = torch.cat(
            [point_embeds, traj_embeds.expand(point_embeds.size(0), -1, -1)], -1
        )
        
        xt = self.heatmap_mlp(point_embeds) # (npoints, action_chunk_size, 3*pos_bins)
        xt = einops.rearrange(xt, "n t (c b) -> t c n b", c=3) # (action_chunk_size, 3, #npoints, pos_bins)

        split_point_embeds = torch.split(point_embeds, npoints_in_batch_list)
        pc_embeds = torch.stack([torch.max(x, 0)[0] for x in split_point_embeds], 0)
        # (batch_size, action_chunk_size, dim)
        action_embeds = self.action_mlp(pc_embeds)
        
        # (batch_size, action_chunk_size, dim)
        xr = action_embeds[..., :self.euler_bins*3]
        # (batch_size, action_chunk_size, euler_bins, 3)
        xr = einops.rearrange(xr, "n t (b c) -> n t b c", c=3)
        
        # (batch_size, action_chunk_size)
        xo = action_embeds[..., -1]

        # generate final continuous actions
        pred_actions = None
        if return_cont_actions:
            pred_actions = self.compute_continuous_action_from_discrete_bins(
                xt, xr, xo, point_coords, npoints_in_batch_list
            )
            
        return xt, xr, xo, pred_actions

class PointCenteredClassificationActionHeadBase(nn.Module):
    """
    Directly use action chunk embeds for position prediction
    """
    def __init__(
        self, 
        hidden_size, 
        action_chunk_size,
        dropout=0, 
        euler_resolution=5, 
        pos_bin_size=0.01, 
        pos_bins=100, 
        pos_heatmap_type="plain",
    ) -> None:
        super().__init__()
        # action: pos_bins * 3 on each axis, euler_bins * 3, openness 1d
        self.hidden_size = hidden_size
        self.action_chunk_size = action_chunk_size
        self.dropout = dropout

        self.pos_bin_size = pos_bin_size
        self.pos_bins = pos_bins
        self.pos_heatmap_type = pos_heatmap_type    # plan, dist
        self.euler_resolution = euler_resolution * torch.pi / 180
        self.euler_bins = 360 // euler_resolution
        
        self.build_model()

    def build_model(self):
        raise NotImplementedError("should implement build_model()")
    
    def compute_continuous_action_from_discrete_bins(self, xt, xr, xo, center_coords=None):
        device = xt.device

        half_pos_bins = self.pos_bins // 2
        shift = torch.arange(-half_pos_bins, half_pos_bins, device=device, dtype=xt.dtype) * self.pos_bin_size
        # (batch_size, action_chunk_size, 3, pos_bins)
        cands_pos = shift.view(1, 1, 1, self.pos_bins).expand(*xt.shape[:-1], -1)
        # (batch_size, action_chunk_size, 3, 1)
        best_idxs = torch.argmax(xt, -1).unsqueeze(-1)
        # (batch_size, action_chunk_size, 3)
        cont_pred_pos = torch.gather(cands_pos, dim=-1, index=best_idxs).squeeze(-1)
        if center_coords is not None:
            cont_pred_pos = cont_pred_pos + center_coords.unsqueeze(1)

        # (batch_size, action_chunk_size, 3)
        pred_rot = torch.argmax(xr, dim=3) * self.euler_resolution

        pred_actions = torch.cat(
            [cont_pred_pos, pred_rot, torch.sigmoid(xo.unsqueeze(-1))], dim=-1
        )
        return pred_actions
    
    def prepare_classification_labels(
        self, tgt_actions, center_coords=None
    ):
        """Ensure the tgt_action is the absolute position and rotation (not normalized)
        Args:
            tgt_actions: (batch_size, action_chunk_size, dim_actions=7)
            center_coords: (batch_size, 3)
        """
        batch_size, action_chunk_size, _ = tgt_actions.size()

        tgt_pos = tgt_actions[:, :, :3]
        tgt_rot = tgt_actions[:, :, 3:6]
        tgt_open = tgt_actions[:, :, 6]

        tgt_disc_prob_list = []
        for i in range(batch_size):
            tgt_disc_prob_list.append([])
            for j in range(action_chunk_size):
                if center_coords is None:
                    center_point = torch.zeros(1, 3, dtype=tgt_actions.dtype, device=tgt_actions.device)
                else:
                    center_point = center_coords[i].unsqueeze(0)
                tgt_disc_prob_list[-1].append(get_disc_gt_pos_prob(
                    center_point, tgt_pos[i, j], pos_bin_size=self.pos_bin_size,
                    pos_bins=self.pos_bins, heatmap_type=self.pos_heatmap_type,
                ))
            # (action_chunk_size, 3, pos_bins)
            tgt_disc_prob_list[-1] = torch.stack(tgt_disc_prob_list[-1], 0)
        # (batch_size, action_chunk_size, 3, pos_bins)
        tgt_disc_prob_list = torch.stack(tgt_disc_prob_list)
        
        # (batch_size, action_chunk_size, 3) long
        tgt_rot = tgt_rot % (2 * torch.pi) # convert to [0, 2pi]
        tgt_rot = torch.round(tgt_rot / self.euler_resolution).long()
        tgt_rot[tgt_rot == self.euler_bins] = 0

        return tgt_disc_prob_list, tgt_rot, tgt_open

    def compute_loss(
        self, pred_actions, tgt_actions, action_masks, center_coords=None
    ):
        batch_size = tgt_actions.size(0)
        pred_pos, pred_rot, pred_open = pred_actions
        tgt_pos, tgt_rot, tgt_open = self.prepare_classification_labels(
            tgt_actions, center_coords, 
        )

        # position loss
        # (batch_size, action_chunk_size, 3, pos_bins)]
        pos_loss = F.cross_entropy(
            einops.rearrange(pred_pos, "n t c b -> (n t c) b"), 
            einops.rearrange(tgt_pos, "n t c b -> (n t c) b"), 
            reduction="none"
        )
        pos_loss = einops.rearrange(pos_loss, "(n t c) -> n t c", c=3, n=batch_size)
        pos_loss = torch.sum(pos_loss * action_masks.unsqueeze(-1)) / action_masks.sum()

        # rotation loss
        rot_loss = F.cross_entropy(
            einops.rearrange(pred_rot, "n t c b -> (n t c) b"), 
            einops.rearrange(tgt_rot, "n t c -> (n t c)"), 
            reduction="none"
        )
        rot_loss = einops.rearrange(rot_loss, "(n t c) -> n t c", c=3, n=batch_size) 
        rot_loss = torch.sum(rot_loss * action_masks.unsqueeze(-1)) / action_masks.sum()

        # openness state loss
        open_loss = F.binary_cross_entropy_with_logits(
            pred_open, tgt_open, reduction="none"
        ) 
        open_loss = (open_loss * action_masks).sum() / action_masks.sum()

        total_loss = pos_loss + rot_loss + open_loss
        
        return total_loss, (pos_loss, rot_loss, open_loss)


class PointWithActionMoEClassificationMLPActionHead(PointMoEClassificationActionHeadBase):
    
    def build_model(self):
        # use MLP layers for action prediction
        self.heatmap_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LeakyReLU(0.02),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 3 * self.pos_bins)
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.02),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.euler_bins * 3 + 1)
        )

    def forward(
        self, action_embeds, point_embeds, point_coords, npoints_in_batch, return_cont_actions=False
    ):
        """
        Args:
            action_embeds: (batch_size, action_chunk_size, dim)
            point_embeds: (npoints, dim)
        """
        npoints_in_batch_list = npoints_in_batch.data.cpu().numpy().tolist()

        point_embeds_list = torch.split(point_embeds, npoints_in_batch_list)

        out_point_embeds = []
        for i, point_embed in enumerate(point_embeds_list):
            action_embed = action_embeds[i].unsqueeze(0).expand(
                point_embed.size(0), -1, -1
            )
            point_embed = point_embed.unsqueeze(1).expand(
                -1, action_embed.size(1), -1
            )
            out_point_embed = torch.cat([action_embed, point_embed], dim=-1)
            out_point_embeds.append(out_point_embed)
        out_point_embeds = torch.cat(out_point_embeds, 0)
        # (npoints, action_chunk_size, 3*pos_bins)
        xt = self.heatmap_mlp(out_point_embeds) 
        xt = einops.rearrange(xt, "n t (c b) -> t c n b", c=3) # (action_chunk_size, 3, #npoints, pos_bins)

        # (batch_size, action_chunk_size, dim)
        pred_actions = self.action_mlp(action_embeds)

        # (batch_size, action_chunk_size, dim)
        xr = pred_actions[..., :self.euler_bins*3]
        # (batch_size, action_chunk_size, euler_bins, 3)
        xr = einops.rearrange(xr, "n t (b c) -> n t b c", c=3)
        
        # (batch_size, action_chunk_size)
        xo = pred_actions[..., -1]

        # generate final continuous actions
        pred_actions = None
        if return_cont_actions:
            pred_actions = self.compute_continuous_action_from_discrete_bins(
                xt, xr, xo, point_coords, npoints_in_batch_list
            )
            
        return xt, xr, xo, pred_actions
    

class PointWithActionCenteredClassificationMLPActionHead(PointCenteredClassificationActionHeadBase):
    
    def build_model(self):
        self.action_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.02),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 3 * self.pos_bins + 3 * self.euler_bins + 1)
        )

    def forward(
        self, action_embeds, center_coords=None, return_cont_actions=False
    ):
        """
        Args:
            point_embeds: (# all points, dim)
            npoints_in_batch: (batch_size, )
        Return:
            pred_actions: (batch, action_chunk_size, dim_actions)
        """ 
        action_embeds = self.action_mlp(action_embeds)
        xt = action_embeds[..., :3*self.pos_bins]
        xr = action_embeds[..., 3*self.pos_bins:3*self.pos_bins+3*self.euler_bins]
        xo = action_embeds[..., -1]

        # (batch_size, action_chunk_size, 3, self.pos_bins)
        xt = einops.rearrange(xt, "n t (c b) -> n t c b", c=3)
        xr = einops.rearrange(xr, "n t (c b) -> n t c b", c=3)

        # generate final continuous actions
        pred_actions = None
        if return_cont_actions:   
            pred_actions = self.compute_continuous_action_from_discrete_bins(
                xt, xr, xo, center_coords=center_coords
            )
            
        return xt, xr, xo, pred_actions
