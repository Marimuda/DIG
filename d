[1mdiff --git a/dig/3dgraph/dimenetpp/model.py b/dig/3dgraph/dimenetpp/model.py[m
[1mindex 28882d7..e77bdec 100644[m
[1m--- a/dig/3dgraph/dimenetpp/model.py[m
[1m+++ b/dig/3dgraph/dimenetpp/model.py[m
[36m@@ -25,6 +25,8 @@[m [mtry:[m
 except ImportError:[m
     sym = None[m
 [m
[32m+[m[32mdevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')[m
[32m+[m
 class emb(torch.nn.Module):[m
     def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent):[m
         super(emb, self).__init__()[m
[36m@@ -85,29 +87,36 @@[m [mclass init(torch.nn.Module):[m
 [m
         return e1, e2[m
 [m
[31m-[m
[32m+[m[32m# InteractionPPBlock[m
 class update_e(torch.nn.Module):[m
     def __init__(self, hidden_channels, int_emb_size, basis_emb_size, num_spherical, num_radial, [m
         num_before_skip, num_after_skip, act=swish):[m
         super(update_e, self).__init__()[m
         self.act = act[m
[32m+[m
[32m+[m[32m        # Transformations of Bessel and spherical basis representations[m
         self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size, bias=False)[m
         self.lin_rbf2 = nn.Linear(basis_emb_size, hidden_channels, bias=False)[m
         self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size, bias=False)[m
         self.lin_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)[m
         self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)[m
 [m
[32m+[m[32m        # Dense transformations of input messages[m
         self.lin_kj = nn.Linear(hidden_channels, hidden_channels)[m
         self.lin_ji = nn.Linear(hidden_channels, hidden_channels)[m
 [m
[32m+[m[32m        # Embedding projections for interaction triplets[m
         self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)[m
         self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)[m
 [m
[32m+[m[32m        # Resudual layers before skip connection[m
         self.layers_before_skip = torch.nn.ModuleList([[m
             ResidualLayer(hidden_channels, act)[m
             for _ in range(num_before_skip)[m
         ])[m
         self.lin = nn.Linear(hidden_channels, hidden_channels)[m
[32m+[m
[32m+[m[32m        # Residual layers after skip connection[m
         self.layers_after_skip = torch.nn.ModuleList([[m
             ResidualLayer(hidden_channels, act)[m
             for _ in range(num_after_skip)[m
[36m@@ -142,33 +151,45 @@[m [mclass update_e(torch.nn.Module):[m
         rbf0, sbf = emb[m
         x1,_ = x[m
 [m
[32m+[m[32m        # Initial transformation[m
         x_ji = self.act(self.lin_ji(x1))[m
         x_kj = self.act(self.lin_kj(x1))[m
 [m
[32m+[m[32m        # Transformation via Bessel basis[m
         rbf = self.lin_rbf1(rbf0)[m
         rbf = self.lin_rbf2(rbf)[m
         x_kj = x_kj * rbf[m
 [m
[32m+[m[32m        # Down-project embeddings and generate interaction triplet embeddings[m
         x_kj = self.act(self.lin_down(x_kj))[m
 [m
[32m+[m[32m        # Transform via 2D spherical basis[m
         sbf = self.lin_sbf1(sbf)[m
         sbf = self.lin_sbf2(sbf)[m
         x_kj = x_kj[idx_kj] * sbf[m
 [m
[32m+[m[32m        # Aggregate interactions and up-project embeddings[m
         x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))[m
         x_kj = self.act(self.lin_up(x_kj))[m
 [m
[32m+[m[32m        # Transformations before skip connection[m
         e1 = x_ji + x_kj[m
         for layer in self.layers_before_skip:[m
             e1 = layer(e1)[m
[32m+[m
[32m+[m[32m        # Skip connection[m
         e1 = self.act(self.lin(e1)) + x1[m
[32m+[m
[32m+[m[32m        # Transformations after skip connection[m
         for layer in self.layers_after_skip:[m
             e1 = layer(e1)[m
[31m-        e2 = self.lin_rbf(rbf0) * e1[m
 [m
[31m-        return e1, e2 [m
[32m+[m[32m        # TODO: Verify that this is a part of the original solution[m
[32m+[m[32m        e2 = self.lin_rbf(rbf0) * e1[m
 [m
[32m+[m[32m        return e1, e2[m
 [m
[32m+[m[32m# OutputPPBlock[m
 class update_v(torch.nn.Module):[m
     def __init__(self, hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init):[m
         super(update_v, self).__init__()[m
[36m@@ -195,10 +216,14 @@[m [mclass update_v(torch.nn.Module):[m
 [m
     def forward(self, e, i, num_nodes=None):[m
         _, e2 = e[m
[32m+[m
[32m+[m[32m        # Aggregate interactions and up-project embeddings[m
         v = scatter(e2, i, dim=0) #, dim_size=num_nodes[m
         v = self.lin_up(v)[m
[32m+[m
         for lin in self.lins:[m
             v = self.act(lin(v))[m
[32m+[m
         v = self.lin(v)[m
         return v[m
 [m
[36m@@ -228,7 +253,7 @@[m [mclass dimenetpp(torch.nn.Module):[m
         self.init_v = update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init)[m
         self.init_u = update_u()[m
         self.emb = emb(num_spherical, num_radial, self.cutoff, envelope_exponent)[m
[31m-        [m
[32m+[m
         self.update_vs = torch.nn.ModuleList([[m
             update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init) for _ in range(num_layers)])[m
 [m
[36m@@ -257,6 +282,7 @@[m [mclass dimenetpp(torch.nn.Module):[m
 [m
 [m
     def forward(self, batch_data):[m
[32m+[m
         z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch[m
         if self.energy_and_force:[m
             pos.requires_grad_()[m
[36m@@ -266,14 +292,16 @@[m [mclass dimenetpp(torch.nn.Module):[m
 [m
         emb = self.emb(dist, angle, idx_kj)[m
 [m
[31m-        #Initialize edge, node, graph features[m
[32m+[m[32m        # Initialize edge, node, graph features[m
         e = self.init_e(z, emb, i, j)[m
[32m+[m[32m        # v = index_tensor[m
         v = self.init_v(e, i, num_nodes=pos.size(0))[m
[31m-        u = self.init_u(torch.zeros_like(scatter(v, batch, dim=0)), v, batch) #scatter(v, batch, dim=0)[m
[32m+[m[32m        # u = dimension, batch = src[m
[32m+[m[32m        u = self.init_u(torch.zeros_like(scatter(v, batch, dim=0)), v, batch) # scatter(v, batch, dim=0)[m
 [m
         for update_e, update_v, update_u in zip(self.update_es, self.update_vs, self.update_us):[m
             e = update_e(e, emb, idx_kj, idx_ji)[m
             v = update_v(e, i)[m
[31m-            u = update_u(u, v, batch) #u += scatter(v, batch, dim=0)[m
[32m+[m[32m            u = update_u(u, v, batch) # u += scatter(v, batch, dim=0)[m
 [m
         return u[m
