import os
import subprocess
import numpy as np
import copy
import argparse
from subprocess import check_call
from PIL import Image
import torch
import math
from prototree.prototree import ProtoTree
from structure import *

def gen_vis(tree: ProtoTree, folder_name: str, args: argparse.Namespace, classes:tuple):
    destination_folder=os.path.join(args.log_dir,folder_name)
    upsample_dir = os.path.join(os.path.join(args.log_dir, args.dir_for_saving_images), folder_name)
    if not os.path.isdir(destination_folder):
        os.mkdir(destination_folder)
    if not os.path.isdir(destination_folder + '/node_vis'):
        os.mkdir(destination_folder + '/node_vis')

    with torch.no_grad():
        s = 'digraph T {margin=0;ranksep=".03";nodesep="0.05";splines="false";\n'
        s += 'node [shape=rect, label=""];\n'
        s += _gen_dot_nodes(tree._root, destination_folder, upsample_dir, classes)
        s += _gen_dot_edges(tree._root, classes)[0]
        s += '}\n'

    with open(os.path.join(destination_folder,'treevis.dot'), 'w') as f:
        f.write(s)
   
    from_p = os.path.join(destination_folder,'treevis.dot')
    to_pdf = os.path.join(destination_folder,'treevis.pdf')
    check_call('dot -Tpdf -Gmargin=0 %s -o %s'%(from_p, to_pdf), shell=True)

def _node_vis(node: Node, upsample_dir: str):
    if isinstance(node, Leaf):
        return _leaf_vis(node)
    if isinstance(node, Branch):
        return _branch_vis(node, upsample_dir)


def _leaf_vis(node: Leaf):
    if node._log_probabilities:
        ws = copy.deepcopy(torch.exp(node.distribution()).cpu().detach().numpy())
    else:
        ws = copy.deepcopy(node.distribution().cpu().detach().numpy())
    
    ws = np.ones(ws.shape) - ws
    ws *= 255

    height = 24

    if ws.shape[0] < 36:
        img_size = 36
    else:
        img_size = ws.shape[0]
    scaler = math.ceil(img_size/ws.shape[0])

    img = Image.new('F', (ws.shape[0]*scaler, height))
    pixels = img.load()

    for i in range(scaler*ws.shape[0]):
        for j in range(height-10):
            pixels[i,j]=ws[int(i/scaler)]
        for j in range(height-10,height-9):
            pixels[i,j]=0 #set bottom line of leaf distribution black
        for j in range(height-9,height):
            pixels[i,j]=255 #set bottom part of node white such that class label is readable

    if scaler*ws.shape[0]>100:
        img=img.resize((100,height))
    return img


def _branch_vis(node: Branch, upsample_dir: str):
    branch_id = node.index
    
    img = Image.open(os.path.join(upsample_dir, '%s_nearest_patch_of_image.png'%branch_id))
    bb = Image.open(os.path.join(upsample_dir, '%s_bounding_box_nearest_patch_of_image.png'%branch_id))
    map = Image.open(os.path.join(upsample_dir, '%s_heatmap_original_image.png'%branch_id))
    w, h = img.size
    wbb, hbb = bb.size
    
    if wbb < 100 and hbb < 100:
        cs = wbb, hbb
    else:
        cs = 100/wbb, 100/hbb
        min_cs = min(cs)
        bb = bb.resize(size=(int(min_cs * wbb), int(min_cs * hbb)))
        wbb, hbb = bb.size

    if w < 100 and h < 100:
        cs = w, h
    else:
        cs = 100/w, 100/h
        min_cs = min(cs)
        img = img.resize(size=(int(min_cs * w), int(min_cs * h)))
        w, h = img.size

    between = 4
    total_w = w+wbb + between
    total_h = max(h, hbb)
    

    together = Image.new(img.mode, (total_w, total_h), color=(255,255,255))
    together.paste(img, (0, 0))
    together.paste(bb, (w+between, 0))

    return together


def _gen_dot_nodes(node: Node, destination_folder: str, upsample_dir: str, classes:tuple):
    img = _node_vis(node, upsample_dir).convert('RGB')
    if isinstance(node, Leaf):
        if node._log_probabilities:
            ws = copy.deepcopy(torch.exp(node.distribution()).cpu().detach().numpy())
        else:
            ws = copy.deepcopy(node.distribution().cpu().detach().numpy())
        argmax = np.argmax(ws)
        targets = [argmax] if argmax.shape == () else argmax.tolist()
        class_targets = copy.deepcopy(targets)
        for i in range(len(targets)):
            t = targets[i]
            class_targets[i] = classes[t]
        str_targets = ','.join(str(t) for t in class_targets) if len(class_targets) > 0 else ""
        str_targets = str_targets.replace('_', ' ')
    filename = '{}/node_vis/node_{}_vis.jpg'.format(destination_folder, node.index)
    img.save(filename)
    if isinstance(node, Leaf):
        s = '{}[imagepos="tc" imagescale=height image="{}" label="{}" labelloc=b fontsize=10 penwidth=0 fontname=Helvetica];\n'.format(node.index, filename, str_targets)
    else:
        s = '{}[image="{}" xlabel="{}" fontsize=6 labelfontcolor=gray50 fontname=Helvetica];\n'.format(node.index, filename, node.index)
    if isinstance(node, Branch):
        return s\
               + _gen_dot_nodes(node.l, destination_folder, upsample_dir, classes)\
               + _gen_dot_nodes(node.r, destination_folder, upsample_dir, classes)
    if isinstance(node, Leaf):
        return s


def _gen_dot_edges(node: Node, classes:tuple):
    if isinstance(node, Branch):
        edge_l, targets_l = _gen_dot_edges(node.l, classes)
        edge_r, targets_r = _gen_dot_edges(node.r, classes)
        str_targets_l = ','.join(str(t) for t in targets_l) if len(targets_l) > 0 else ""
        str_targets_r = ','.join(str(t) for t in targets_r) if len(targets_r) > 0 else ""
        s = '{} -> {} [label="Absent" fontsize=10 tailport="s" headport="n" fontname=Helvetica];\n {} -> {} [label="Present" fontsize=10 tailport="s" headport="n" fontname=Helvetica];\n'.format(node.index, node.l.index, 
                                                                       node.index, node.r.index)
        return s + edge_l + edge_r, sorted(list(set(targets_l + targets_r)))
    if isinstance(node, Leaf):
        if node._log_probabilities:
            ws = copy.deepcopy(torch.exp(node.distribution()).cpu().detach().numpy())
        else:
            ws = copy.deepcopy(node.distribution().cpu().detach().numpy())
        argmax = np.argmax(ws)
        targets = [argmax] if argmax.shape == () else argmax.tolist()
        class_targets = copy.deepcopy(targets)
        for i in range(len(targets)):
            t = targets[i]
            class_targets[i] = classes[t]
        return '', class_targets


import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import argparse
from subprocess import check_call
import math
from PIL import Image
from prototree.upsample import find_high_activation_crop, imsave_with_bbox
import torch

import torchvision
from torchvision.utils import save_image

from prototree.prototree import ProtoTree
from structure import *

def upsample_local(tree: ProtoTree,
                 sample: torch.Tensor,
                 sample_dir: str,
                 folder_name: str,
                 img_name: str,
                 decision_path: list,
                 args: argparse.Namespace):
    
    dir = os.path.join(os.path.join(os.path.join(args.log_dir, folder_name),img_name), args.dir_for_saving_images)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with torch.no_grad():
        _, distances_batch, _ = tree.forward_partial(sample)
        sim_map = torch.exp(-distances_batch[0,:,:,:]).cpu().numpy()
    for i, node in enumerate(decision_path[:-1]):
        decision_node_idx = node.index
        node_id = tree._out_map[node]
        img = Image.open(sample_dir)
        x_np = np.asarray(img)
        x_np = np.float32(x_np)/ 255
        if x_np.ndim == 2: #convert grayscale to RGB
            x_np = np.stack((x_np,)*3, axis=-1)
        
        img_size = x_np.shape[:2]
        similarity_map = sim_map[node_id]

        rescaled_sim_map = similarity_map - np.amin(similarity_map)
        rescaled_sim_map= rescaled_sim_map / np.amax(rescaled_sim_map)
        similarity_heatmap = cv2.applyColorMap(np.uint8(255*rescaled_sim_map), cv2.COLORMAP_JET)
        similarity_heatmap = np.float32(similarity_heatmap) / 255
        similarity_heatmap = similarity_heatmap[...,::-1]
        plt.imsave(fname=os.path.join(dir,'%s_heatmap_latent_similaritymap.png'%str(decision_node_idx)), arr=similarity_heatmap, vmin=0.0,vmax=1.0)

        upsampled_act_pattern = cv2.resize(similarity_map,
                                            dsize=(img_size[1], img_size[0]),
                                            interpolation=cv2.INTER_CUBIC)
        rescaled_act_pattern = upsampled_act_pattern - np.amin(upsampled_act_pattern)
        rescaled_act_pattern = rescaled_act_pattern / np.amax(rescaled_act_pattern)
        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1]
        overlayed_original_img = 0.5 * x_np + 0.2 * heatmap
        plt.imsave(fname=os.path.join(dir,'%s_heatmap_original_image.png'%str(decision_node_idx)), arr=overlayed_original_img, vmin=0.0,vmax=1.0)

        # save the highly activated patch
        masked_similarity_map = np.ones(similarity_map.shape)
        masked_similarity_map[similarity_map < np.max(similarity_map)] = 0 #mask similarity map such that only the nearest patch z* is visualized
        
        upsampled_prototype_pattern = cv2.resize(masked_similarity_map,
                                            dsize=(img_size[1], img_size[0]),
                                            interpolation=cv2.INTER_CUBIC)
        plt.imsave(fname=os.path.join(dir,'%s_masked_upsampled_heatmap.png'%str(decision_node_idx)), arr=upsampled_prototype_pattern, vmin=0.0,vmax=1.0) 
            
        high_act_patch_indices = find_high_activation_crop(upsampled_prototype_pattern, args.upsample_threshold)
        high_act_patch = x_np[high_act_patch_indices[0]:high_act_patch_indices[1],
                                            high_act_patch_indices[2]:high_act_patch_indices[3], :]
        plt.imsave(fname=os.path.join(dir,'%s_nearest_patch_of_image.png'%str(decision_node_idx)), arr=high_act_patch, vmin=0.0,vmax=1.0)

        # save the original image with bounding box showing high activation patch
        imsave_with_bbox(fname=os.path.join(dir,'%s_bounding_box_nearest_patch_of_image.png'%str(decision_node_idx)),
                            img_rgb=x_np,
                            bbox_height_start=high_act_patch_indices[0],
                            bbox_height_end=high_act_patch_indices[1],
                            bbox_width_start=high_act_patch_indices[2],
                            bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))

def gen_pred_vis(tree: ProtoTree,
                 sample: torch.Tensor,
                 sample_dir: str,
                 folder_name: str,
                 args: argparse.Namespace,
                 classes: tuple,
                 pred_kwargs: dict = None,
                 ):
    pred_kwargs = pred_kwargs or dict()  # TODO -- assert deterministic routing
    
    # Create dir to store visualization
    img_name = sample_dir.split('/')[-1].split(".")[-2]
    
    if not os.path.exists(os.path.join(args.log_dir, folder_name)):
        os.makedirs(os.path.join(args.log_dir, folder_name))
    destination_folder=os.path.join(os.path.join(args.log_dir, folder_name),img_name)
    
    if not os.path.isdir(destination_folder):
        os.mkdir(destination_folder)
    if not os.path.isdir(destination_folder + '/node_vis'):
        os.mkdir(destination_folder + '/node_vis')

    # Get references to where source files are stored
    upsample_path = os.path.join(os.path.join(args.log_dir,args.dir_for_saving_images),'pruned_and_projected')
    nodevis_path = os.path.join(args.log_dir,'pruned_and_projected/node_vis')
    local_upsample_path = os.path.join(destination_folder, args.dir_for_saving_images)

    # Get the model prediction
    with torch.no_grad():
        pred, pred_info = tree.forward(sample, sampling_strategy='greedy', **pred_kwargs)
        probs = pred_info['ps']
        label_ix = torch.argmax(pred, dim=1)[0].item()
        assert 'out_leaf_ix' in pred_info.keys()

    # Save input image
    sample_path = destination_folder + '/node_vis/sample.jpg'
    # save_image(sample, sample_path)
    Image.open(sample_dir).save(sample_path)

    # Save an image containing the model output
    output_path = destination_folder + '/node_vis/output.jpg'
    leaf_ix = pred_info['out_leaf_ix'][0]
    leaf = tree.nodes_by_index[leaf_ix]
    decision_path = tree.path_to(leaf)

    upsample_local(tree,sample,sample_dir,folder_name,img_name,decision_path,args)

    # Prediction graph is visualized using Graphviz
    # Build dot string
    s = 'digraph T {margin=0;rankdir=LR\n'
    # s += "subgraph {"
    s += 'node [shape=plaintext, label=""];\n'
    s += 'edge [penwidth="0.5"];\n'

    # Create a node for the sample image
    s += f'sample[image="{sample_path}"];\n'

    # Create nodes for all decisions/branches
    # Starting from the leaf
    for i, node in enumerate(decision_path[:-1]):
        node_ix = node.index
        prob = probs[node_ix].item()
        
        s += f'node_{i+1}[image="{upsample_path}/{node_ix}_nearest_patch_of_image.png" group="{"g"+str(i)}"];\n' 
        if prob > 0.5:
            s += f'node_{i+1}_original[image="{local_upsample_path}/{node_ix}_bounding_box_nearest_patch_of_image.png" imagescale=width group="{"g"+str(i)}"];\n'  
            label = "Present      \nSimilarity %.4f                   "%prob
            s += f'node_{i+1}->node_{i+1}_original [label="{label}" fontsize=10 fontname=Helvetica];\n'
        else:
            s += f'node_{i+1}_original[image="{sample_path}" group="{"g"+str(i)}"];\n'
            label = "Absent      \nSimilarity %.4f                   "%prob
            s += f'node_{i+1}->node_{i+1}_original [label="{label}" fontsize=10 fontname=Helvetica];\n'
        # s += f'node_{i+1}_original->node_{i+1} [label="{label}" fontsize=10 fontname=Helvetica];\n'
        
        s += f'node_{i+1}->node_{i+2};\n'
        s += "{rank = same; "f'node_{i+1}_original'+"; "+f'node_{i+1}'+"};"

    # Create a node for the model output
    s += f'node_{len(decision_path)}[imagepos="tc" imagescale=height image="{nodevis_path}/node_{leaf_ix}_vis.jpg" label="{classes[label_ix]}" labelloc=b fontsize=10 penwidth=0 fontname=Helvetica];\n'

    # Connect the input image to the first decision node
    s += 'sample->node_1;\n'


    s += '}\n'

    with open(os.path.join(destination_folder, 'predvis.dot'), 'w') as f:
        f.write(s)

    from_p = os.path.join(destination_folder, 'predvis.dot')
    to_pdf = os.path.join(destination_folder, 'predvis.pdf')
    check_call('dot -Tpdf -Gmargin=0 %s -o %s' % (from_p, to_pdf), shell=True)


