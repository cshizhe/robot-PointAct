import torch
import torch.nn as nn


class PointMoERegressionMLPActionHead(nn.Module):
    def __init__(
        self, 
        hidden_size, 
        dim_action, 
        action_chunk_size=1, 
        dropout=0, 
        heatmap_temp=1,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.action_chunk_size = action_chunk_size
        self.heatmap_temp = heatmap_temp

        # TODO: initialize with sincos embedding
        self.traj_embedding = nn.Embedding(action_chunk_size, hidden_size)

        self.heatmap_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(0.02),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        self.action_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(0.02),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, dim_action)
        )

    def forward(self, point_embeds, npoints_in_batch, point_coords=None, **kwargs):
        '''
        Args:
            point_embeds: (# all points, dim)
            npoints_in_batch: (batch_size, )
        Return:
            pred_actions: (batch, num_steps, dim_actions)
        ''' 
        device = point_embeds.device
        point_embeds = point_embeds.unsqueeze(1).expand(-1, self.action_chunk_size, -1)

        traj_embeds = self.traj_embedding(torch.arange(self.action_chunk_size).to(device))
        # (#points, action_chunk_size, dim)
        point_embeds = torch.cat(
            [point_embeds, traj_embeds.expand(point_embeds.size(0), -1, -1)], -1
        )

        pred_heatmaps = self.heatmap_mlp(point_embeds)  # (npoints, action_chunk_size, 1)
        pred_actions = self.action_mlp(point_embeds)    # (npoints, action_chunk_size, dim_action)

        npoints_in_batch_list = npoints_in_batch.data.cpu().numpy().tolist()

        pred_heatmap_list = torch.split(pred_heatmaps, npoints_in_batch_list)
        pred_heatmap_list = [torch.softmax(x / self.heatmap_temp, dim=0) for x in pred_heatmap_list]

        pred_action_list = torch.split(pred_actions, npoints_in_batch_list)
        pred_action_list = [torch.sum(h*a, 0) for h, a in zip(pred_heatmap_list, pred_action_list)]

        # (batch_size, action_chunk_size, dim_action)
        pred_actions = torch.stack(pred_action_list, 0)

        return pred_actions


class PointWithActionRegressionMLPActionHead(nn.Module):
    def __init__(
        self, 
        hidden_size,
        dim_action,
        action_chunk_size,
        dropout=0, 
        use_heatmap=False,
        heatmap_temp=1,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.action_chunk_size = action_chunk_size
        self.heatmap_temp = heatmap_temp
        self.use_heatmap = use_heatmap

        if self.use_heatmap:
            self.heatmap_mlp = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LeakyReLU(0.02),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1 + 3)
            )
            self.action_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(0.02),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, dim_action - 3)
            )
        else:
            self.action_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(0.02),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, dim_action)
            )

    def forward(
        self, 
        action_embeds,
        point_embeds=None, 
        npoints_in_batch=None
    ):
        '''
        Args:
            action_embeds: (batch_size, action_chunk_size, dim)
            point_embeds: (# all points, dim)
            npoints_in_batch: (batch_size, )
        Return:
            pred_actions: (batch, num_steps, dim_actions)
        ''' 
        action_chunk_size = self.action_chunk_size

        if self.use_heatmap:
            npoints_in_batch_list = npoints_in_batch.data.cpu().numpy().tolist()
            point_embeds_list = torch.split(point_embeds, npoints_in_batch_list)

            xt = []
            for i, point_embed in enumerate(point_embeds_list):
                point_embed = point_embed.unsqueeze(1).expand(
                    -1, action_chunk_size, -1
                )
                action_embed = action_embeds[i].unsqueeze(0).expand(
                    point_embed.size(0), -1, -1
                )
                out_point_embed = torch.cat(
                    [action_embed, point_embed], dim=-1
                )
                # (npoints, action_chunk_size, 4)
                heatmap_and_pos = self.heatmap_mlp(out_point_embed)
                heatmap = torch.softmax(heatmap_and_pos[:, :, 0] / self.heatmap_temp, dim=0)
                pos = heatmap_and_pos[:, :, 1:]    # (npoints, action_chunk_size, 3)
                
                # (action_chunk_size, 3)
                pos = torch.sum(pos * heatmap.unsqueeze(-1), dim=0)
                xt.append(pos)
            # (batch_size, action_chunk_size, 3)
            xt = torch.stack(xt, 0)    

            # (batch_size, action_chunk_size, dim)
            pred_actions = self.action_mlp(action_embeds)
            pred_actions = torch.cat([xt, pred_actions], dim=-1)

        else:
            # (batch_size, action_chunk_size, dim)
            pred_actions = self.action_mlp(action_embeds)
            return pred_actions

        return pred_actions

