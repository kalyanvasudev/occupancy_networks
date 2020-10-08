import torch
import torch.optim as optim
from torch import autograd
import numpy as np
from tqdm import trange
import trimesh
from im2mesh.utils import libmcubes
from im2mesh.common import make_3d_grid
from im2mesh.utils.libsimplify import simplify_mesh
from im2mesh.utils.libmise import MISE
from im2mesh.utils.libmesh import check_mesh_contains
from im2mesh.common import compute_iou
from torch.nn import functional as F
import time


class Generator3D(object):
    '''  Generator class for Occupancy Networks.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        simplify_nfaces (int): number of faces the mesh should be simplified to
        preprocessor (nn.Module): preprocessor for inputs
    '''

    def __init__(self, model, points_batch_size=100000,
                 threshold=0.5, refinement_step=0, device=None,
                 resolution0=16, upsampling_steps=3,
                 with_normals=False, padding=0.1, sample=False,
                 simplify_nfaces=None,
                 preprocessor=None):
        self.model = model.to(device)
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.padding = padding
        self.sample = sample
        self.simplify_nfaces = simplify_nfaces
        self.preprocessor = preprocessor


    def generate_mesh_from_points(self, data, return_stats=True):
        ''' Generates the output from points and occupancies.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = torch.empty(1, 0).to(device)
        kwargs = {}

        # Preprocess if requires
        if self.preprocessor is not None:
            t0 = time.time()
            with torch.no_grad():
                inputs = self.preprocessor(inputs)
            stats_dict['time (preprocess)'] = time.time() - t0

        # Encode inputs
        t0 = time.time()
        with torch.no_grad():
            c = self.model.encode_inputs(inputs)
        stats_dict['time (encode inputs)'] = time.time() - t0

        kwargs = {}
        z = self.model.infer_z(data['points'].to(device),data['points.occ'].to(device), c, **kwargs)
        z = z.loc.detach()
        
        mesh = self.generate_from_latent(z, c, stats_dict=stats_dict, **kwargs)

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh

    def optimize_z(self, z_gt, c, data, opt_batch_size=2048,
                                        opt_iters=10000,
                                        z_type='random'): 
        device = self.device
        all_points = data['points.all_points'] 
        all_occ = data['points.all_occ']
        
        # optimize for z

        # sample from a N(0,1)

        if z_type == 'prior':
            z = self.model.get_z_from_prior((1,), sample=self.sample).to(device)
            z = z.float().to(device).requires_grad_(True)
        elif z_type == 'uniform':
            z = torch.empty(z_gt.shape)
            torch.nn.init.uniform_(z,-0.1,0.1)
            z= z.float().to(device).requires_grad_(True)
        elif z_type == 'random':
            z = torch.rand(1,128).float().to(device).requires_grad_(True)   
        elif z_type == 'zeros':
            z = torch.zeros(1,128).float().to(device).requires_grad_(True)   
        


        kwargs = {}
        stats_dict = {}
        mesh = self.generate_from_latent(z, c, stats_dict=stats_dict, **kwargs)
        occ = check_mesh_contains(mesh, data['points.all_points'].squeeze(0).numpy())
        iou = compute_iou(occ, data['points.all_occ'].squeeze(0).numpy())
        print((f"####### IOU before optimization: {iou} ##########"))
        
        # sample completely random
        #z = torch.rand(z_gt.shape).float().to(device).requires_grad_(True)
        
        #optimizer = torch.optim.Adam([z], lr=0.001 )#,weight_decay=0.0001)
        optimizer = torch.optim.SGD([z], lr=0.001, momentum=0.9 )#,weight_decay=0.0001)

        for i in range(opt_iters):
            optimizer.zero_grad()
            idx = np.random.randint(all_points.shape[1], size= opt_batch_size)
            points = all_points[:, idx,:].to(device)
            occ = all_occ[:, idx].to(device)
            
            
            logits = self.model.decode(points, z, c, **kwargs).logits
            loss_i = F.binary_cross_entropy_with_logits(
                logits, occ, reduction='none')
            loss = loss_i.sum(-1).mean() #+ z.abs().mean()
            if not i %2000 or i ==0: 
                print("Opt Iter: {:4d}, Loss: {:0.3f}, Z Dist: {:0.3f}, Z_gt Norm: {:0.3f}, Z_cur Norm: {:0.3f}".format( 
                    i, loss.tolist(), torch.norm(z_gt-z).tolist(), torch.norm(z_gt).tolist(), torch.norm(z).tolist()))
            loss.backward()
            optimizer.step()

            ############# For debug only
            #logits = self.model.decode(points, z_gt, c, **kwargs).logits
            #loss_i = F.binary_cross_entropy_with_logits(
            #    logits, occ, reduction='none')
            #loss = loss_i.sum(-1).mean() #+ z.abs().mean()
            #if not i %200 or i ==0: 
            #    print("Loss from Z_git: {:0.3f}".format(loss.tolist()))
            ############# End of debug only

        return z.detach(), loss.tolist()


    def generate_mesh_from_points_optimize(self, data, 
                                        return_stats=True, 
                                        opt_batch_size=2048,
                                        opt_iters=2000,
                                        rnd_restart_num=1,
                                        z_type='random'): # avail options 'random' and 'dist'
        ''' Only batch size 1 supported!!!.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = torch.empty(1, 0).to(device)
        kwargs = {}

        # Preprocess if requires
        if self.preprocessor is not None:
            t0 = time.time()
            with torch.no_grad():
                inputs = self.preprocessor(inputs)
            stats_dict['time (preprocess)'] = time.time() - t0

        # Encode inputs
        t0 = time.time()
        with torch.no_grad():
            c = self.model.encode_inputs(inputs)
        stats_dict['time (encode inputs)'] = time.time() - t0

        kwargs = {}
        # ground truth z
        z_gt = self.model.infer_z(data['points.all_points'].to(device),data['points.all_occ'].to(device), c, **kwargs)
        z_gt = z_gt.loc.detach()

        z, loss = None, np.float('inf')

        for i in range(rnd_restart_num):
            print(f"####### Rand restart number: {i}/{rnd_restart_num} ##########")
            temp_z, temp_loss = self.optimize_z(z_gt, c, data, opt_batch_size,opt_iters, z_type)
            if temp_loss < loss:
                z = temp_z
                loss = temp_loss

        mesh = self.generate_from_latent(z, c, stats_dict=stats_dict, **kwargs)
        occ = check_mesh_contains(mesh, data['points.all_points'].squeeze(0).numpy())
        iou = compute_iou(occ, data['points.all_occ'].squeeze(0).numpy())

        #occ = self.model.decoder( data['points.all_points'].to(device), z, c, **kwargs)
        #iou_new = compute_iou(occ.squeeze(0).cpu().detach().numpy(), data['points.all_occ'].squeeze(0).numpy())
        print((f"####### IOU after optimization: {iou} ##########",loss))
        if return_stats:
            return mesh, stats_dict
        else:
            return mesh

    def optimize_c(self, data, opt_batch_size=2048,
                                        opt_iters=10000,
                                        c_type='random'): 
        device = self.device
        all_points = data['points.all_points'] 
        all_occ = data['points.all_occ']
        
        z = torch.empty(1, 0).to(device)
        
        # sample completely random
        if c_type == 'random':
            c = torch.rand(1,256).float().to(device).requires_grad_(True)   
        elif c_type == 'uniform':
            # sample from uniform in a range
            c = torch.empty(1,256)
            torch.nn.init.uniform_(c,-0.5,0.5)
            c= c.float().to(device).requires_grad_(True)
        elif c_type == 'zeros':
            c = torch.zeros(1,256).float().to(device).requires_grad_(True)   

        kwargs = {}
        stats_dict = {}
        #mesh = self.generate_from_latent(z, c, stats_dict=stats_dict, **kwargs)
        #occ = check_mesh_contains(mesh, data['points.all_points'].squeeze(0).numpy())
        #iou = compute_iou(occ, data['points.all_occ'].squeeze(0).numpy())
        #print((f"####### IOU before optimization: {iou} ##########"))
        

        optimizer = torch.optim.Adam([c], lr=0.1 )#,weight_decay=0.0001)
        #optimizer = torch.optim.SGD([c], lr=0.0001, momentum=0.9 )#,weight_decay=0.0001)

        for i in range(opt_iters):
            optimizer.zero_grad()
            z = torch.empty(1, 0).to(device)
            idx = np.random.randint(all_points.shape[1], size= opt_batch_size)
            points = all_points[:, idx,:].to(device)
            occ = all_occ[:, idx].to(device)
            
            
            logits = self.model.decode(points, z, c, **kwargs).logits
            loss_i = F.binary_cross_entropy_with_logits(
                logits, occ, reduction='none')
            loss = loss_i.sum(-1).mean() #+ z.abs().mean()
            if not i %2000 or i ==0: 
                print("Opt Iter: {:4d}, Loss: {:0.3f}".format( i, loss.tolist() ), torch.norm(c))
            loss.backward()
            optimizer.step()

        return c.detach(), loss.tolist()
    
    def generate_mesh_from_points_optimize_c(self, data, 
                                        return_stats=True, 
                                        opt_batch_size=2008,
                                        opt_iters=20000,
                                        rnd_restart_num=1,
                                        c_type='random'): # avail options 'random' and 'dist'
        ''' Only batch size 1 supported!!!.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}
        kwargs = {}
    
        c, loss = None, np.float('inf')

        for i in range(rnd_restart_num):
            print(f"####### Rand restart number: {i}/{rnd_restart_num} ##########")
            temp_c, temp_loss = self.optimize_c(data, opt_batch_size,opt_iters, c_type)
            if temp_loss < loss:
                c = temp_c
                loss = temp_loss

        #z = self.model.get_z_from_prior((1,), sample=self.sample).to(device)
        z = torch.empty(1, 0).to(device)
        mesh = self.generate_from_latent(z, c, stats_dict=stats_dict, **kwargs)
        occ = check_mesh_contains(mesh, data['points.all_points'].squeeze(0).numpy())
        iou = compute_iou(occ, data['points.all_occ'].squeeze(0).numpy())

        #occ = self.model.decoder( data['points.all_points'].to(device), z, c, **kwargs)
        #iou = compute_iou(occ.squeeze(0).cpu().detach().numpy(), data['points.all_occ'].squeeze(0).numpy())
        print((f"####### IOU after optimization: {iou} ##########", loss))
        if return_stats:
            return mesh, stats_dict
        else:
            return mesh

    def generate_mesh(self, data, return_stats=True):
        ''' Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        kwargs = {}

        # Preprocess if requires
        if self.preprocessor is not None:
            t0 = time.time()
            with torch.no_grad():
                inputs = self.preprocessor(inputs)
            stats_dict['time (preprocess)'] = time.time() - t0

        # Encode inputs
        t0 = time.time()
        with torch.no_grad():
            c = self.model.encode_inputs(inputs)
        stats_dict['time (encode inputs)'] = time.time() - t0

        z = self.model.get_z_from_prior((1,), sample=self.sample).to(device)
        mesh = self.generate_from_latent(z, c, stats_dict=stats_dict, **kwargs)

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh

    def generate_from_latent(self, z, c=None, stats_dict={}, **kwargs):
        ''' Generates mesh from latent.

        Args:
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding

        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            )
            values = self.eval_points(pointsf, z, c, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            points = mesh_extractor.query()        
            while points.shape[0] != 0:
                # Query points
                pointsf = torch.FloatTensor(points).to(self.device)
                # Normalize to bounding box
                pointsf = pointsf / mesh_extractor.resolution
                pointsf = box_size * (pointsf - 0.5)
                # Evaluate model and update
                values = self.eval_points(
                    pointsf, z, c, **kwargs).cpu().numpy()
                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()

        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0

        mesh = self.extract_mesh(value_grid, z, c, stats_dict=stats_dict)
        return mesh

    def eval_points(self, p, z, c=None, **kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points 
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                occ_hat = self.model.decode(pi, z, c, **kwargs).logits

            occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)

        return occ_hat

    def extract_mesh(self, occ_hat, z, c=None, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # Undo padding
        vertices -= 1
        # Normalize to bounding box
        vertices /= np.array([n_x-1, n_y-1, n_z-1])
        vertices = box_size * (vertices - 0.5)

        # mesh_pymesh = pymesh.form_mesh(vertices, triangles)
        # mesh_pymesh = fix_pymesh(mesh_pymesh)

        # Estimate normals if needed
        if self.with_normals and not vertices.shape[0] == 0:
            t0 = time.time()
            normals = self.estimate_normals(vertices, z, c)
            stats_dict['time (normals)'] = time.time() - t0

        else:
            normals = None

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=normals,
                               process=False)

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
            stats_dict['time (simplify)'] = time.time() - t0

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, occ_hat, z, c)
            stats_dict['time (refine)'] = time.time() - t0

        return mesh

    def estimate_normals(self, vertices, z, c=None):
        ''' Estimates the normals by computing the gradient of the objective.

        Args:
            vertices (numpy array): vertices of the mesh
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        z, c = z.unsqueeze(0), c.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            occ_hat = self.model.decode(vi, z, c).logits
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, occ_hat, z, c=None):
        ''' Refines the predicted mesh.

        Args:   
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert(n_x == n_y == n_z)
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces).to(self.device)

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-4)

        for it_r in trange(self.refinement_step):
            optimizer.zero_grad()

            # Loss
            face_vertex = v[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
            eps = torch.FloatTensor(eps).to(self.device)
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
            face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / \
                (face_normal.norm(dim=1, keepdim=True) + 1e-10)
            face_value = torch.sigmoid(
                self.model.decode(face_point.unsqueeze(0), z, c).logits
            )
            normal_target = -autograd.grad(
                [face_value.sum()], [face_point], create_graph=True)[0]

            normal_target = \
                normal_target / \
                (normal_target.norm(dim=1, keepdim=True) + 1e-10)
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = \
                (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()

        mesh.vertices = v.data.cpu().numpy()

        return mesh
