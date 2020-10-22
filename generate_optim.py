import torch
# import torch.distributions as dist
import os
import pyglet
import numpy as np
import shutil
import argparse
import trimesh
from tqdm import tqdm
import time
from collections import defaultdict
import pandas as pd
from im2mesh import config
from im2mesh.checkpoints import CheckpointIO
from im2mesh.utils.io import export_pointcloud
from im2mesh.utils.visualize import visualize_data
from im2mesh.utils.voxels import VoxelGrid


def mesh_to_img(path,mesh):
    scene = mesh.scene()
    img_pat = path + '/render.png'
    
    cam_old, _ = scene.graph[scene.camera.name]
    rotate = trimesh.transformations.rotation_matrix(
    	angle= np.radians(-45.0), direction=[1,0,0], point=scene.centroid)
    cam_new = np.dot(rotate, cam_old)
    scene.graph[scene.camera.name] = cam_new
    png = scene.save_image(resolution=[1920,1080], visible=True)
    with open(img_pat, 'wb') as f:
        f.write(png)
        f.close()

parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('sampling_type', type=str, help='Path to config file.')
parser.add_argument('model_type', type=str, help='Path to config file.')
parser.add_argument('opt_iters', type=int, help='Path to config file.')
parser.add_argument('rnd_restart_num', type=int, help='Path to config file.')

parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')



args = parser.parse_args()

print("Input argumentss : ", args.config, args.sampling_type, args.model_type, args.opt_iters, args.rnd_restart_num)


cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")



opt_batch_size = cfg['generation']['opt_batch_size']

model_type = args.model_type
opt_iters = args.opt_iters
rnd_restart_num = args.rnd_restart_num
sampling_type = args.sampling_type

#model_type = cfg['generation']['model_type']
#opt_iters = cfg['generation']['opt_iters']
#rnd_restart_num= cfg['generation']['rnd_restart_num']
#sampling_type = cfg['generation']['sampling_type']

out_dir = cfg['training']['out_dir']
out_dir += '_sample_{}_restarts_{}_optiters_{}_optbatchsize_{}'.format(sampling_type,rnd_restart_num,opt_iters,opt_batch_size)
generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
out_time_file = os.path.join(generation_dir, 'time_generation_full.pkl')
out_time_file_class = os.path.join(generation_dir, 'time_generation.pkl')

batch_size = cfg['generation']['batch_size']
input_type = cfg['data']['input_type']
vis_n_outputs = cfg['generation']['vis_n_outputs']
if vis_n_outputs is None:
    vis_n_outputs = -1

# Dataset
dataset = config.get_dataset('test', cfg, return_idx=True)

# Model
model = config.get_model(cfg, device=device, dataset=dataset)

checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])

# Generator
generator = config.get_generator(model, cfg, device=device)

# Determine what to generate
generate_mesh = cfg['generation']['generate_mesh']
generate_pointcloud = cfg['generation']['generate_pointcloud']

if generate_mesh and not hasattr(generator, 'generate_mesh'):
    generate_mesh = False
    print('Warning: generator does not support mesh generation.')

if generate_pointcloud and not hasattr(generator, 'generate_pointcloud'):
    generate_pointcloud = False
    print('Warning: generator does not support pointcloud generation.')


# Loader
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=0, shuffle=False)

# Statistics
time_dicts = []

# Generate
model.eval()

mesh_gt_root = '/private/home/kalyanv/learning_vision3d/datasets/ShapeNetCore.v1'#cfg['data'][mesh_gt_root]
# Count how many models already created
model_counter = defaultdict(int)
i = 0
for it, data in enumerate(test_loader):
    if i >= vis_n_outputs:
        break
    i += 1
    
    # input
    # Output folders
    mesh_dir = os.path.join(generation_dir, 'meshes')
    pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
    in_dir = os.path.join(generation_dir, 'input')
    generation_vis_dir = os.path.join(generation_dir, 'vis', )

    
    # Get index etc.
    idx = data['idx'].item()

    try:
        model_dict = dataset.get_model_dict(idx)
    except AttributeError:
        model_dict = {'model': str(idx), 'category': 'n/a'}
    
    modelname = model_dict['model']
    category_id = model_dict.get('category', 'n/a')

    mesh_gt_path = os.path.join(mesh_gt_root, str(category_id), modelname, 'model.obj')
    print("Iteration:", i, "Model path:", mesh_gt_path )
    try:
        category_name = dataset.metadata[category_id].get('name', 'n/a')
    except AttributeError:
        category_name = 'n/a'

    if category_id != 'n/a':
        mesh_dir = os.path.join(mesh_dir, str(category_id))
        pointcloud_dir = os.path.join(pointcloud_dir, str(category_id))
        in_dir = os.path.join(in_dir, str(category_id))

        folder_name = str(category_id)
        if category_name != 'n/a':
            folder_name = str(folder_name) + '_' + category_name.split(',')[0]

        generation_vis_dir = os.path.join(generation_vis_dir, folder_name)

    # Create directories if necessary
    if vis_n_outputs >= 0 and not os.path.exists(generation_vis_dir):
        os.makedirs(generation_vis_dir)

    if generate_mesh and not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)

    if generate_pointcloud and not os.path.exists(pointcloud_dir):
        os.makedirs(pointcloud_dir)

    if not os.path.exists(in_dir):
        os.makedirs(in_dir)
    
    # Timing dict
    time_dict = {
        'idx': idx,
        'class id': category_id,
        'class name': category_name,
        'modelname': modelname,
    }
    time_dicts.append(time_dict)

    # Generate outputs
    out_file_dict = {}
    out_file_dict['gt'] = mesh_gt_path

    if generate_mesh:
        t0 = time.time()

        if model_type == 'single':
            out = generator.generate_mesh_from_points_optimize(data,
                                        opt_batch_size=opt_batch_size,
                                        opt_iters=opt_iters,
                                        rnd_restart_num=rnd_restart_num,
                                        z_type=sampling_type)
        else:
            out = generator.generate_mesh_from_points_optimize_c(data,
                                        opt_batch_size=opt_batch_size,
                                        opt_iters=opt_iters,
                                        rnd_restart_num=rnd_restart_num,
                                        c_type=sampling_type)
        time_dict['mesh'] = time.time() - t0

        # Get statistics
        try:
            mesh, stats_dict = out
        except TypeError:
            mesh, stats_dict = out, {}
        time_dict.update(stats_dict)

        
        # Write output
        mesh_out_file = os.path.join(mesh_dir, '%s.off' % modelname)
        mesh.export(mesh_out_file)
        out_file_dict['mesh'] = mesh_out_file

        #temp_m = trimesh.load_mesh(mesh_out_file)
        #mesh_to_img(mesh_dir,mesh)
        ###### Add ground truth mesh from latent
        #out_gt = generator.generate_mesh_from_points(data)

        # Get statistics
        #try:
        #    mesh, stats_dict = out
        #except TypeError:
        #    mesh, stats_dict = out, {}
        #time_dict.update(stats_dict)

        # Write output
        #mesh_out_file = os.path.join(mesh_dir, '%s.off' % (modelname+ '_gt'))
        #mesh.export(mesh_out_file)
        #out_file_dict['mesh_gt'] = mesh_out_file


    if cfg['generation']['copy_input']:
        # Save inputs
        if input_type == 'img':
            inputs_path = os.path.join(in_dir, '%s.jpg' % modelname)
            inputs = data['inputs'].squeeze(0).cpu()
            visualize_data(inputs, 'img', inputs_path)
            out_file_dict['in'] = inputs_path
        elif input_type == 'voxels':
            inputs_path = os.path.join(in_dir, '%s.off' % modelname)
            inputs = data['inputs'].squeeze(0).cpu()
            voxel_mesh = VoxelGrid(inputs).to_mesh()
            voxel_mesh.export(inputs_path)
            out_file_dict['in'] = inputs_path
        elif input_type == 'pointcloud':
            inputs_path = os.path.join(in_dir, '%s.ply' % modelname)
            inputs = data['inputs'].squeeze(0).cpu().numpy()
            export_pointcloud(inputs, inputs_path, False)
            out_file_dict['in'] = inputs_path

    # Copy to visualization directory for first vis_n_output samples
    c_it = model_counter[category_id]
    if c_it < vis_n_outputs:
        # Save output files
        img_name = '%02d.off' % c_it
        for k, filepath in out_file_dict.items():
            ext = os.path.splitext(filepath)[1]
            out_file = os.path.join(generation_vis_dir, '%02d_%s%s'
                                    % (c_it, k, ext))
            shutil.copyfile(filepath, out_file)

    model_counter[category_id] += 1

