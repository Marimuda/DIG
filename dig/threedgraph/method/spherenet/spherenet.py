from typing import Optional
import torch
from torch import Tensor
from torch import nn
from torch.nn import Linear, Embedding, ReLU, Sequential, ModuleList
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn import radius_graph
from torch_geometric.utils import degree
from torch_scatter import scatter
from math import sqrt

from ...utils import swish, xyz_to_dat, FCLayer, MLP
from .features import dist_emb, angle_emb, torsion_emb

from .aggregators import AGGREGATORS
from .scalers import SCALERS

try:
    import sympy as sym
except ImportError:
    sym = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class emb(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, cutoff_g, envelope_exponent, fix=False):
        super(emb, self).__init__()
        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent, fix)
        self.dist_emb_g = dist_emb(num_radial, cutoff_g, envelope_exponent, fix)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent, fix)
        self.torsion_emb = torsion_emb(
            num_spherical, num_radial, cutoff, envelope_exponent, fix
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.dist_emb.reset_parameters()

    def forward(self, dist, angle, torsion, idx_kj):
        dist_emb = self.dist_emb(dist)
        dist_emb_g = self.dist_emb_g(dist)
        angle_emb = self.angle_emb(dist, angle, idx_kj)
        torsion_emb = self.torsion_emb(dist, angle, torsion, idx_kj)
        return dist_emb, angle_emb, torsion_emb, dist_emb_g


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class init(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, act=swish):
        super(init, self).__init__()
        self.act = act
        self.emb = Embedding(95, hidden_channels)
        self.lin_rbf_0 = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf_0.reset_parameters()
        self.lin.reset_parameters()
        glorot_orthogonal(self.lin_rbf_1.weight, scale=2.0)

    def forward(self, x, emb, i, j):
        rbf, _, _, _ = emb
        x = self.emb(x)
        rbf0 = self.act(self.lin_rbf_0(rbf))

        x_tmp = torch.cat([x[i], x[j], rbf0], dim=-1)

        e1 = self.act(self.lin(x_tmp))
        # Embeddings block ends here.
        e2 = self.lin_rbf_1(rbf)
        e2 = e2 * e1

        return e1, e2


class update_eT(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        int_emb_size,
        basis_emb_size_dist,
        basis_emb_size_angle,
        num_bilinear,  # 64
        num_spherical,
        num_radial,
        act=swish,
        use_bilinear=True
    ):
        super(update_eT, self).__init__()
        self.use_bilinear = use_bilinear
        self.hidden_channels = hidden_channels
        self.act = act
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size_dist, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size_dist, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(
            num_spherical * num_radial, int_emb_size, bias=False
        )

        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        if use_bilinear:
            #Bilinear layer, num_bilinear 32 quad, 64 trip
            self.W = nn.Parameter(
                     torch.Tensor(int_emb_size, num_bilinear, int_emb_size))
        else:
            self.lin_sbf2 = nn.Linear(basis_emb_size_angle, int_emb_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        if self.use_bilinear:
            # Bilinear layer
            self.W.data.normal_(mean=0.0, std=2 / self.W.size(0))
        else:
            glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

    def forward(self, x, emb, idx_kj, idx_ji):
        rbf0, sbf, t = emb
        x1, _ = x

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf1(sbf)

        if self.use_bilinear:
            x_kj = torch.einsum('wj,wl,ijl->wi', sbf, x_kj[idx_kj], self.W)
        else:
            sbf = self.lin_sbf2(sbf)
            x_kj = x_kj[idx_kj] * sbf

        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        e = x_ji + x_kj

        return e

class update_eG(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        int_emb_size,
        basis_emb_size_dist,
        num_radial,
        act=swish,
    ):
        super(update_eG, self).__init__()

        self.act = act
        self.lin_rbf1_g = nn.Linear(num_radial, basis_emb_size_dist, bias=False)
        self.lin_rbf2_g = nn.Linear(basis_emb_size_dist, hidden_channels, bias=False)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1_g.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2_g.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

    def forward(self, x1, emb, idx_kj, idx_ji):
        _, _, _, rbf0_g = emb

        #TODO: split global and quad jump into different modules, whereas global module feeds into quad / tri module.

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        rbf_g = self.lin_rbf1_g(rbf0_g)
        rbf_g = self.lin_rbf2_g(rbf_g)
        x_kj = x_kj * rbf_g

        # Down-project embeddings and generate triplet embeddings
        x_kj = self.act(self.lin_down(x_kj))
        x_kj = x_kj[idx_kj]

        # Aggregate interactions and up-project embeddings
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        # Transformation before skip connection
        x2_g = x_ji + x_kj

        return x2_g, x_kj, x_ji


class update_eQ(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        int_emb_size,
        basis_emb_size_dist,
        basis_emb_size_angle,
        basis_emb_size_torsion,
        num_bilinear,  # 32
        num_spherical,
        num_radial,
        act=swish,
        use_bilinear=True
    ):
        super(update_eQ, self).__init__()
        self.use_bilinear = use_bilinear
        self.act = act
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size_dist, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size_dist, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(
            num_spherical * num_radial, basis_emb_size_angle, bias=False
        )
        self.lin_sbf2 = nn.Linear(basis_emb_size_angle, int_emb_size, bias=False)
        self.lin_t1 = nn.Linear(
            num_spherical * num_spherical * num_radial,
            int_emb_size,  # basis_emb_size_torsion,
            bias=False,
        )
        #self.lin_t2 = nn.Linear(basis_emb_size_torsion, int_emb_size, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        #self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        #self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)
        
        if use_bilinear:
            #Bilinear layer, num_bilinear 32 quad, 64 trip
            self.W = nn.Parameter(
                     torch.Tensor(int_emb_size, num_bilinear, int_emb_size))
        else:
            self.lin_t2 = nn.Linear(basis_emb_size_torsion, int_emb_size, bias=False)


        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_t1.weight, scale=2.0)

        #glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        #self.lin_kj.bias.data.fill_(0)
        #glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        #self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)

        if self.use_bilinear:
            #Bilinear layer
            self.W.data.normal_(mean=0.0, std=2 / self.W.size(0))
        else:
            glorot_orthogonal(self.lin_t2.weight, scale=2.0)

    def forward(self, x1, emb, x_kj, x_ji, idx_kj, idx_ji):
        rbf0, sbf, t, _ = emb

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        t = self.lin_t1(t)

        if self.use_bilinear:
            t = torch.einsum('wj,wl,ijl->wi', t, x_kj, self.W)
        else:
            t = self.lin_t2(t)
            x_kj = x_kj * t

        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        return x_kj


class update_e(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        int_emb_size_T,  # 64
        int_emb_size_Q,  # 32
        basis_emb_size_dist,
        basis_emb_size_angle,
        basis_emb_size_torsion,
        num_bilinear_T,  # 64
        num_bilinear_Q,  # 32
        num_spherical,
        num_radial,
        num_before_skip,
        num_after_skip,
        act=swish,
        use_bilinear=True
    ):
        super(update_e, self).__init__()
        self.act = act
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.lin_skip = nn.Linear(hidden_channels, hidden_channels)
        #self.mponejump = update_eT(hidden_channels, int_emb_size_T, basis_emb_size_dist, basis_emb_size_angle, num_bilinear_T, num_spherical, num_radial, act=act, use_bilinear=use_bilinear)
        self.mpjump_g = update_eG(hidden_channels, int_emb_size_Q, basis_emb_size_dist, num_radial, act=act)
        self.mptwojump = update_eQ(hidden_channels, int_emb_size_Q, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion, num_bilinear_Q, num_spherical, num_radial, act=act, use_bilinear=use_bilinear)

        self.layers_before_skip = torch.nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)]
        )
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)]
        )

        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)
        #self.lin_rbf_g = nn.Linear(num_radial, hidden_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin.weight, scale=2.0)
        glorot_orthogonal(self.lin_skip.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        #glorot_orthogonal(self.lin_rbf_g.weight, scale=2.0)

    def forward(self, x, emb, x_kj, x_ji):
        x1,_ = x
        rbf0, _, _, rbf0_g = emb

        x_old = x1
        #x1 = self.lin(x1)

        #tmp = self.mponejump(x, emb, x_kj, x_ji)
        qmpg, x_kj_g, x_ji_g = self.mpjump_g(x1, emb, x_kj, x_ji)

        for layer in self.layers_before_skip:
            qmpg = layer(qmpg)
        qmpg = self.act(self.lin_skip(qmpg)) + x_old
        for layer in self.layers_after_skip:
            qmpg = layer(qmpg)

        x_kj = self.mptwojump(x1, emb, x_kj_g, x_ji_g, x_kj, x_ji)

        e1 = x_ji_g + x_kj   #+ tmp

        for layer in self.layers_before_skip:
            e1 = layer(e1)

        e1 = self.act(self.lin_skip(e1)) + qmpg

        for layer in self.layers_after_skip:
            e1 = layer(e1)

        e2 = self.lin_rbf(rbf0)
        #gg = self.lin_rbf_g(rbf0_g)

        #e2 = g * gg
        e2 = e2 * e1

        #e2 = self.lin_rbf(rbf0) * e1
        return e1, e2



class update_v(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        out_emb_channels,
        out_channels,
        num_output_layers,
        act,
        output_init,
    ):
        super(update_v, self).__init__()
        self.act = act
        self.output_init = output_init
        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        self.lins = torch.nn.ModuleList()

        for _ in range(num_output_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_init == "zeros":
            self.lin.weight.data.fill_(0)
        if self.output_init == "GlorotOrthogonal":
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(self, e, i, num_nodes):
        _, e2 = e
        v = scatter(e2, i, dim=0)
        v = self.lin_up(v)
        for lin in self.lins:
            v = self.act(lin(v))
        v = self.lin(v)
        return v


class update_u(torch.nn.Module):
    def __init__(self):
        super(update_u, self).__init__()

    def forward(self, u, v, batch):
        u += scatter(v, batch, dim=0)
        return u

class update_u2(torch.nn.Module):
    def __init__(
        self,
        aggregators,
        scalers,
        deg,
    ):
        super(update_u, self).__init__()
        self.aggregators = [AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [SCALERS[scale] for scale in scalers]

        deg = deg.to(torch.float)
        self.avg_deg = {
            'lin': deg.mean().item(),
            'log': (deg + 1).log().mean().item(),
            'exp': deg.exp().mean().item(),
        }
        self.posttrans = MLP(in_size=(len(aggregators) * len(scalers) + 1),
                             hidden_size=(len(aggregators) * len(scalers) + 1)//2, out_size=1, layers=1,
                             mid_activation='swish', last_activation='none')

    def forward(self, u: Tensor, inputs: Tensor, index: Tensor, dim_size: Optional[int] = None) -> Tensor: #v, batch
        outs = [aggr(inputs, index, dim_size) for aggr in self.aggregators]
        out = torch.cat(outs, dim=-1)

        deg = degree(index, dim_size, dtype=inputs.dtype)
        deg = deg.clamp_(1).view(-1, 1)
        #deg = deg.clamp_(1).view(-1, 1, 1)

        outs = [scale(out, deg, self.avg_deg) for scale in self.scalers]
        out = torch.cat(outs, dim=-1)
        out = torch.cat([u, out], dim=-1)
        u = self.posttrans(out)
        #u += res
        #u += scatter(v, batch, dim=0)
        return u


class SphereNet(torch.nn.Module):
    r"""
     The spherical message passing neural network SphereNet from the `"Spherical Message Passing for 3D Graph Networks" <https://arxiv.org/abs/2102.05013>`_ paper.

    Args:
        energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the negative of the derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)
        cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`5.0`)
        num_layers (int, optional): Number of building blocks. (default: :obj:`4`)
        hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
        out_channels (int, optional): Size of each output sample. (default: :obj:`1`)
        int_emb_size (int, optional): Embedding size used for interaction triplets. (default: :obj:`64`)
        basis_emb_size_dist (int, optional): Embedding size used in the basis transformation of distance. (default: :obj:`8`)
        basis_emb_size_angle (int, optional): Embedding size used in the basis transformation of angle. (default: :obj:`8`)
        basis_emb_size_torsion (int, optional): Embedding size used in the basis transformation of torsion. (default: :obj:`8`)
        out_emb_channels (int, optional): Embedding size used for atoms in the output block. (default: :obj:`256`)
        num_spherical (int, optional): Number of spherical harmonics. (default: :obj:`7`)
        num_radial (int, optional): Number of radial basis functions. (default: :obj:`6`)
        envelop_exponent (int, optional): Shape of the smooth cutoff. (default: :obj:`5`)
        num_before_skip (int, optional): Number of residual layers in the interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip (int, optional): Number of residual layers in the interaction blocks before the skip connection. (default: :obj:`2`)
        num_output_layers (int, optional): Number of linear layers for the output blocks. (default: :obj:`3`)
        act: (function, optional): The activation funtion. (default: :obj:`swish`)
        output_init: (str, optional): The initialization fot the output. It could be :obj:`GlorotOrthogonal` and :obj:`zeros`. (default: :obj:`GlorotOrthogonal`)

    """

    def __init__(
        self,
        energy_and_force=False,
        cutoff_i=10.0,  # cutoff interaction
        cutoff_e=5.0,  # cutoff embedding
        cutoff_g=7.0, # cutoff global layer
        num_layers=4,
        hidden_channels=128,
        out_channels=1,
        int_emb_size_T=64,
        int_emb_size_Q=32,
        basis_emb_size_dist=8,
        basis_emb_size_angle=8,
        basis_emb_size_torsion=8,
        out_emb_channels=256,
        num_bilinear_T=64,
        num_bilinear_Q=32,
        num_spherical=7,
        num_radial=6,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        act=swish,
        output_init="GlorotOrthogonal",
        fix=False,
        use_bilinear=True,
        aggregators=[],
        scalers=[],
        deg = torch.zeros([2, 2], dtype=torch.int32),
    ):
        super(SphereNet, self).__init__()
        self.scalers = scalers
        self.deg = deg

        self.fix = fix
        self.cutoff_i = cutoff_i  # cutoff interaction
        self.energy_and_force = energy_and_force

        self.init_e = init(num_radial, hidden_channels, act)
        self.init_v = update_v(
            hidden_channels,
            out_emb_channels,
            out_channels,
            num_output_layers,
            act,
            output_init,
        )
        self.init_u = update_u()
        self.emb = emb(num_spherical, num_radial, cutoff_e, cutoff_g, envelope_exponent, fix)

        self.update_vs = torch.nn.ModuleList(
            [
                update_v(
                    hidden_channels,
                    out_emb_channels,
                    out_channels,
                    num_output_layers,
                    act,
                    output_init,
                )
                for _ in range(num_layers)
            ]
        )

        self.update_es = torch.nn.ModuleList(
            [
                update_e(
                    hidden_channels,
                    int_emb_size_T,
                    int_emb_size_Q,
                    basis_emb_size_dist,
                    basis_emb_size_angle,
                    basis_emb_size_torsion,
                    num_bilinear_T,
                    num_bilinear_Q,
                    num_spherical,
                    num_radial,
                    num_before_skip,
                    num_after_skip,
                    act,
                    use_bilinear,
                )
                for _ in range(num_layers)
            ]
        )

        self.update_us = torch.nn.ModuleList([update_u() for _ in range(num_layers)])
        self.reset_parameters()

    def reset_parameters(self):
        self.init_e.reset_parameters()
        self.init_v.reset_parameters()
        self.emb.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()

    def forward(self, batch_data):
        z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
        if self.energy_and_force:
            pos.requires_grad_()
        edge_index = radius_graph(pos, r=self.cutoff_i, batch=batch)
        num_nodes = z.size(0)
        dist, angle, torsion, i, j, idx_kj, idx_ji = xyz_to_dat(
            pos, edge_index, num_nodes, use_torsion=True
        )

        emb = self.emb(dist, angle, torsion, idx_kj)

        # Initialize edge, node, graph features
        e = self.init_e(z, emb, i, j)
        v = self.init_v(e, i, num_nodes)
        u = self.init_u(
            torch.zeros_like(scatter(v, batch, dim=0)), v, batch
        )  # scatter(v, batch, dim=0)

        for update_e, update_v, update_u in zip(
            self.update_es, self.update_vs, self.update_us
        ):
            e = update_e(e, emb, idx_kj, idx_ji)
            v = update_v(e, i, num_nodes)
            u = update_u(u, v, batch)  # u += scatter(v, batch, dim=0)
            #breakpoint()

        return u
