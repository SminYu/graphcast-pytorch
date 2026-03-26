import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
import torch_scatter

from typing import Optional

def mlp_builder(input_dim, hidden_dim, output_dim,
                num_hidden=2, activation=nn.ReLU, norm_last=True):

    layers = [nn.Linear(input_dim, hidden_dim), activation()]

    for _ in range(num_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), activation()]

    layers.append(nn.Linear(hidden_dim, output_dim))
    if norm_last:
        layers.append(nn.LayerNorm(output_dim))

    return nn.Sequential(*layers)

class Graphcast(nn.Module):
    def __init__(self, 
                 input_dim_grid, input_dim_mesh, input_dim_edge, 
                 output_dim_grid, hidden_dim, activation,
                 encoder_hidden, decoder_hidden, gnn_hidden, num_layers):
        super(Graphcast, self).__init__()

        """
        This model consists of five parts: 
        (1) Encoder (for grids, meshes, edges, and mappings)
        (2) Grid-to-Mesh mapping
        (3) Processor
        (4) Mesh-to-Grid mapping
        (5) Decoder (for grids only)
        """

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.activation = activation

        if output_dim_grid is None:
            output_dim_grid = input_dim_grid

        # embed raw inputs at grids into latent space
        self.grid_encoder = mlp_builder(input_dim=input_dim_grid,
                                        hidden_dim=hidden_dim,
                                        output_dim=hidden_dim,
                                        num_hidden=encoder_hidden,
                                        activation=self.activation,
                                        norm_last=True)
        self.mesh_encoder = mlp_builder(input_dim=input_dim_mesh,
                                        hidden_dim=hidden_dim,
                                        output_dim=hidden_dim,
                                        num_hidden=encoder_hidden,
                                        activation=self.activation,
                                        norm_last=True)

        self.g2m_encoder = mlp_builder(input_dim=input_dim_edge,
                                        hidden_dim=hidden_dim,
                                        output_dim=hidden_dim,
                                        num_hidden=encoder_hidden,
                                        activation=self.activation,
                                        norm_last=True)
        self.m2g_encoder = mlp_builder(input_dim=input_dim_edge,
                                        hidden_dim=hidden_dim,
                                        output_dim=hidden_dim,
                                        num_hidden=encoder_hidden,
                                        activation=self.activation,    
                                        norm_last=True)


        # mapping lat-lon grid to multi-mesh
        self.grid2mesh = HeteroMessagePassingNetwork(input_dim=hidden_dim, 
                                                    output_dim=hidden_dim, 
                                                    num_hidden=gnn_hidden,
                                                    activation=self.activation)

        # embed raw inputs at edges into latent space
        self.edge_encoder = mlp_builder(input_dim=input_dim_edge,
                                        hidden_dim=hidden_dim,
                                        output_dim=hidden_dim,
                                        num_hidden=encoder_hidden,
                                        activation=self.activation,
                                        norm_last=True)
        
        assert (self.num_layers >= 1), 'Number of message passing layers is not >=1'
        
        # processing at multi-meshes
        self.processor = nn.ModuleList()
        for _ in range(self.num_layers):
            self.processor.append(MessagePassingNetwork(input_dim=hidden_dim, 
                                                        output_dim=hidden_dim, 
                                                        activation=self.activation,
                                                        num_hidden=gnn_hidden,
                                                        edge_update=True))

        self.mesh2grid = HeteroMessagePassingNetwork(input_dim=hidden_dim, 
                                                    output_dim=hidden_dim, 
                                                    num_hidden=gnn_hidden,
                                                    activation=self.activation)

        # decoding from latent space to physical space
        self.grid_decoder = mlp_builder(input_dim=hidden_dim,
                                        hidden_dim=hidden_dim,
                                        output_dim=output_dim_grid,
                                        num_hidden=decoder_hidden,
                                        activation=self.activation,
                                        norm_last=False)
                
        self.reset_parameters()

    def reset_parameters(self):
        #* Reset all parameters
        for _network in [self.grid_encoder, self.edge_encoder, self.mesh_encoder, 
                        self.g2m_encoder, self.m2g_encoder, self.grid_decoder]:
            for _layer in _network:
                if hasattr(_layer, "reset_parameters"):
                    _layer.reset_parameters()

        for _network in [self.grid2mesh, self.mesh2grid]:
            _network.reset_parameters()
        
        for _layer in self.processor:
            _layer.reset_parameters()

        print('Reset parameters')
        
    def forward(self,
                grid_attr0: Tensor,
                grid_attr1: Tensor,
                grid_static: OptTensor,
                mesh_static: Tensor,
                g2m_index: Tensor,
                g2m_static: Tensor,
                m2g_index: Tensor,
                m2g_static: Tensor,
                edge_index: Tensor,
                edge_static: Tensor,
                ) -> Tensor:
        """
        Encoder encodes graph (cell/edge features) into latent vectors (cell/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """
        if grid_static is not None:
            x_grid = torch.cat([grid_attr0, grid_attr1, grid_static], dim=-1)
        else:
            x_grid = torch.cat([grid_attr0, grid_attr1], dim=-1)
        
        # Encoder: encode cell/edge features into latent embeddings
        x_grid = self.grid_encoder(x_grid)
        x_mesh = self.mesh_encoder(mesh_static)
        edge_attr = self.edge_encoder(edge_static)
        g2m_static = self.g2m_encoder(g2m_static)
        m2g_static = self.m2g_encoder(m2g_static)

        # Merge grid values and zero mesh values into a single grid
        # Mapping grid values into mesh
        x_mesh = self.grid2mesh(x_grid, x_mesh, g2m_index, g2m_static)

        # Processor: perform message passing iteratively with latent embeddings
        for process_step in range(self.num_layers):
            x_mesh, edge_attr = self.processor[process_step](x=x_mesh, 
                                                            edge_index=edge_index, 
                                                            edge_attr=edge_attr)

        # Merge mesh values and zero grid values into a single grid
        # Mapping mesh values to grid
        x_grid = self.mesh2grid(x_mesh, x_grid, m2g_index, m2g_static)
        
        # Decoder decode latent cell embeddings into physical quantities of interest with residual connection
        x_grid = grid_attr1 + self.grid_decoder(x_grid)
        
        # Dummy output for autoregressive model
        return grid_attr1, x_grid

class MessagePassingNetwork(MessagePassing):
    def __init__(self, input_dim, output_dim, activation, num_hidden, edge_update:bool, **kwargs):
        super(MessagePassingNetwork, self).__init__(  **kwargs )
        # input(self embedding, adjacent cell embedding, edge embedding)
        # update edges
        # f(x_i, e_ij, x_j) -> e_ij'
        self.activation = activation
        self.num_hidden = num_hidden
        self.edge_update = edge_update

        self.edge_mlp = mlp_builder(input_dim=3 * input_dim,
                                    hidden_dim=output_dim,
                                    output_dim=output_dim,
                                    num_hidden=self.num_hidden,
                                    activation=self.activation,
                                    norm_last=True)

        # input(self embedding, sum of processced adjacent embedding)
        # update cells
        # g(x_i, sum(f(e_ij, x_i, x_j))) -> x_i'
        self.cell_mlp = mlp_builder(input_dim=2 * input_dim,
                                    hidden_dim=output_dim,
                                    output_dim=output_dim,
                                    num_hidden=self.num_hidden,
                                    activation=self.activation,
                                    norm_last=True)

    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        for _network in [self.edge_mlp, self.cell_mlp]:
            for _layer in _network:
                if not isinstance(_layer, self.activation):
                    _layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr, size = None) -> Tensor:
        """
        Handle the pre and post-processing of cell features/embeddings,
        as well as initiates message passing by calling the propagate function.

        Note that message passing and aggregation are handled by the propagate
        function, and the update

        x has shpae [cell_num , in_channels] (cell embeddings)
        edge_index: [2, edge_num]
        edge_attr: [E, in_channels]
        """
        if self.edge_update:
            out, edge_out = self.propagate(edge_index, 
                                            x = x, 
                                            edge_attr = edge_attr) 
        else:
            out = self.propagate(edge_index, 
                                x = x, 
                                edge_attr = edge_attr) 

        out = self.cell_mlp(torch.cat([x, out], dim=-1))

        return x + out, edge_attr + edge_out if self.edge_update else x + out # residual connection

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        """
        destination cell: x_i has the shape of [E, in_channels]
        source cell: x_j has the shape of [E, in_channels]
        edge attributes: edge_attr has the shape of [E, out_channels]
        """
        # shape of [E, 3 * in_channels]
        updated_edges = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim = 1))
        return updated_edges

    def aggregate(self, 
                  updated_edges: Tensor, 
                  edge_index: Tensor) -> Tensor:
        """
        updated_edges: [E, channels]
        out: [E, channels]
        """
        # The axis along which to index number of cells.
        cell_dim = 0
                
        out = torch_scatter.scatter(src=updated_edges, 
                                    index=edge_index[1, :],
                                    dim=cell_dim, 
                                    reduce='sum')

        return out, updated_edges if self.edge_update else out


class HeteroMessagePassingNetwork(MessagePassing):
    def __init__(self, input_dim, output_dim, activation, num_hidden, **kwargs):
        super(HeteroMessagePassingNetwork, self).__init__(  **kwargs )
        # input(self embedding, adjacent cell embedding, edge embedding)
        # update edges
        # f(x_i, e_ij, x_j) -> e_ij'
        self.activation = activation
        self.num_hidden = num_hidden

        self.edge_mlp = mlp_builder(input_dim=3 * input_dim,
                                    hidden_dim=output_dim,
                                    output_dim=output_dim,
                                    num_hidden=self.num_hidden,
                                    activation=self.activation,
                                    norm_last=True)

        # input(self embedding, sum of processced adjacent embedding)
        # update cells
        # g(x_i, sum(f(e_ij, x_i, x_j))) -> x_i'
        self.cell_mlp = mlp_builder(input_dim=2 * output_dim,
                                    hidden_dim=output_dim,
                                    output_dim=output_dim,
                                    num_hidden=self.num_hidden,
                                    activation=self.activation,
                                    norm_last=True)
                                    
    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        for _network in [self.edge_mlp, self.cell_mlp]:
            for _layer in _network:
                if not isinstance(_layer, self.activation):
                    _layer.reset_parameters()

    def forward(self, x_src, x_dst, edge_index, edge_attr) -> Tensor:
        """
        Note that message passing and aggregation are handled by the propagate
        function, and the update

        x has shpae [cell_num , in_channels] (cell embeddings)
        edge_index: [2, edge_num]
        edge_attr: [E, in_channels]
        """
        out = self.propagate(edge_index, 
                            x = torch.cat([x_dst, x_src], axis=0), 
                            edge_attr = edge_attr)

        out = self.cell_mlp(torch.cat([x_dst, out], axis=-1))

        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        """
        destination cell: x_i has the shape of [E, in_channels]
        source cell: x_j has the shape of [E, in_channels]
        edge attributes: edge_attr has the shape of [E, out_channels]
        """
        # shape of [E, 3 * in_channels]
        updated_edges = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim = 1))
        return updated_edges

    def aggregate(self, 
                  updated_edges: Tensor, 
                  edge_index: Tensor) -> Tensor:
        """
        updated_edges: [E, channels]
        out: [E, channels]
        """
        # The axis along which to index number of cells.
        cell_dim = 0

        # note that the destination and source are switched for hetero message passing for automated output tensor shaping
        out = torch_scatter.scatter(src=updated_edges, 
                                    index=edge_index[0, :],
                                    dim=cell_dim, 
                                    reduce='sum')
        return out

