import torch as ch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
from standard_utils import *

# Take gradient step in the direction 'grad' with magnitude 'step_size'
def grad_step(adv_inputs, grad, step_size):
    l = len(adv_inputs.shape) - 1
    grad_norm = ch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, *([1]*l))
    scaled_grad = grad / (grad_norm + 1e-10)
    return adv_inputs + scaled_grad * step_size

# Compute feature attack for the specified images 'seed_images' by 
# increasing the feature values of features at index 'feature_indices'
def feature_attack(model, seed_images, feature_indices, eps=500, 
                   step_size=1, iterations=1000):
    seed_images = seed_images.cuda()
    batch_size = seed_images.shape[0]
    for i in range(iterations+1):
        seed_images.requires_grad_()

        (_, features), _ = model(seed_images, with_latent=True)
        features_select = features[ch.arange(batch_size), feature_indices]
        if i==iterations:
            seed_images = seed_images.detach()
            break
            
        adv_loss = features_select.sum()
        grads = ch.autograd.grad(adv_loss, [seed_images])[0]
        seed_images = grad_step(seed_images, grads, step_size)
        seed_images = ch.clamp(seed_images, min=0., max=1.)
    return seed_images, features_select

# Compute feature map for the specified images 'images' using 
# the feature map i.e the layer just before the logits layer
# (before applying the global average pooling layer)
def compute_feature_maps(images, model, layer_name='layer4'):
    images = images.cuda()
    normalizer_module = model._modules['normalizer']
    feature_module = model._modules['model']
    x = normalizer_module(images)
    for name, module in feature_module._modules.items():
        x = module(x)
        if name == layer_name:
            break
    return x

# Compute CAM map for the images using the feature at 'feature_index'
# layer_name='layer4' specifies the layer to be used for constructing 
# the heatmap
def compute_cam(model, images, feature_index, layer_name='layer4'):
    b_size = images.shape[0]
    feature_maps = compute_feature_maps(images, model, layer_name=layer_name)
    cam_maps = (feature_maps[:, feature_index, :, :]).detach()
    cam_maps_flat = cam_maps.view(b_size, -1) 
    cam_maps_max, _ = ch.max(cam_maps_flat, dim=1, keepdim=True)
    cam_maps_flat = cam_maps_flat/cam_maps_max
    cam_maps = cam_maps_flat.view_as(cam_maps)

    cam_maps_resized = []
    for cam_map in cam_maps:
        cam_map = cam_map.cpu().numpy()
        cam_map = cv2.resize(cam_map, images.shape[2:])
        cam_maps_resized.append(cam_map)
    cam_maps = np.stack(cam_maps_resized, axis=0)
    cam_maps = ch.from_numpy(1-cam_maps)
    return cam_maps

def print_class_mapping(trunc_class_names):
    if len(trunc_class_names) > 0:
        for class_index in trunc_class_names.keys():
            class_name = trunc_class_names[class_index]
            print('Class index: {:d} ==> Class name: {:s}'.format(class_index, class_name))
        print("")

# Compute heatmaps by multiplying images and masks together
def compute_heatmaps(imgs, masks):
    heatmaps = []
    for (img, mask) in zip(imgs, masks):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap + np.float32(img)
        heatmap = heatmap / np.max(heatmap)
        heatmaps.append(heatmap)
    heatmaps = np.stack(heatmaps, axis=0)
    heatmaps = ch.from_numpy(heatmaps).permute(0, 3, 1, 2)
    return heatmaps

# Visualize a single feature 'feature_id' using 'robust_model'
# and 'group_images'
def feature_visualization(robust_model, group_images, features, feature_id,
                          data_loader, grouping, num_images=6):
        
    dataset = data_loader.dataset
    sorted_indices = np.argsort(features[:, feature_id])
    indices_high = sorted_indices[-num_images:]
    image_indices_high = group_images[indices_high]
    images_highest, labels_highest, preds_highest = load_images(image_indices_high, data_loader)

    trunc_class_names = {}
    images_captions = []
    heatmaps_captions = []
    for i, index in enumerate(indices_high):
        if grouping == "label":
            caption = preds_highest[i]
        else:
            caption = labels_highest[i]#.split(',')[0]
            
        if len(caption) <= 12:
            images_captions.append(caption)
        else:
            index = dataset.class_indices_dict[caption]
            images_captions.append(str(index))                
            trunc_class_names[index] = caption

        images_captions.append(caption)

        
    cam_maps = compute_cam(robust_model, images_highest, 
                                   feature_id, layer_name='layer4')        
    images_heatmaps = compute_heatmaps(images_highest.permute(0, 2, 3, 1), 
                                       cam_maps)
    
    images = []
    
    if grouping == "label":
        footnote = "model prediction"
    else:
        footnote = "label"
    title = "Images that most strongly have this Feature ({:s} at bottom)".format(footnote)
    img = show_image_row([images_highest.cpu()], [], tlist=[images_captions], title=title, fontsize=18)
    images.append(img)
    #print_class_mapping(trunc_class_names)
    
    title = "Heatmaps for this Feature in those images"
    img = show_image_row([images_heatmaps.cpu()], [], tlist=[], 
                   title=title, y_offset=-0.4, fontsize=18)
    images.append(img)

    images_attack, features_attack = feature_attack(robust_model, 
                                                    images_highest, 
                                                    feature_id)

    title = "Amplifying this Feature in those images"
    img = show_image_row([images_attack.cpu()], [], tlist=[], 
                   title=title, y_offset=-0.4, fontsize=18)
    images.append(img)
    return images

# Display the most activating images, heatmaps and feature attack 
# images for a decision_path
def display_images(decision_path, data_loader, model, features, 
                   grouping, image_indices=None, num_images=6):
    if image_indices is None:
        image_indices = np.arange(len(data_loader.dataset))
    img_list = []
    feature_id_list = []
    for node in decision_path:
        node_id, feature_id, feature_threshold, direction = node
        feature_visualization(model, image_indices, features, feature_id,
                              data_loader, grouping, num_images=num_images)
        
# Display the failures at a leaf node
def display_failures(leaf_id, leaf_failure_indices, data_loader, grouping, 
                     num_images=6, num_rows=2):
    dataset = data_loader.dataset
    if grouping == "label":
        footnote = "model prediction"
    else:
        footnote = "label"
    title = "Errors from this Group ({:s} at bottom)".format(footnote)

    image_indices = np.arange(len(data_loader.dataset))
    if len(leaf_failure_indices) <= num_images*num_rows:
        # more images to show than leaves
        leaf_select_failures = leaf_failure_indices
    else:
        # sample without replacement
        leaf_select_failures = np.random.choice(leaf_failure_indices, num_images*num_rows, replace=False)
    
    image_indices_failures = image_indices[leaf_select_failures]

    images = []
    trunc_class_names = {}
    start = 0
    row = 0
    while (row < num_rows) and (start < len(leaf_failure_indices)):
        image_indices_select = image_indices_failures[start: start + num_images]
        images_failures, labels_failures, preds_failures = load_images(image_indices_select, data_loader)
        
        images_captions = []
        
        n_this_row = len(image_indices_select)
        
        for i in range(n_this_row):
            if grouping == "label":
                caption = preds_failures[i]
            else:
                caption = labels_failures[i]
            if len(caption) <= 12:
                images_captions.append(caption)
            else:
                index = dataset.class_indices_dict[caption]
                images_captions.append(str(index))                
                trunc_class_names[index] = caption
        if row != 0:
            title = ""
        
        # If needed, append blank images to [images_failures] in order to fill out the grid
        n_blank = num_images - n_this_row
        if n_blank > 0:
            images_failures = ch.cat((images_failures, ch.ones((n_blank, images_failures.shape[1], images_failures.shape[2], images_failures.shape[3]))), dim=0)
            images_captions.extend(['' for n in range(n_blank)])
        
        img = show_image_row([images_failures], [], tlist=[images_captions], title=title, fontsize=18)
        images.append(img)
        start = start + num_images
        row += 1
          
    #print_class_mapping(trunc_class_names)
    return images
    
def get_axis(axarr, H, W, i, j):
    H, W = H - 1, W - 1
    if not (H or W):
        ax = axarr
    elif not (H and W):
        ax = axarr[max(i, j)]
    else:
        ax = axarr[i][j]
    return ax
        
# Show image row using images specified in xlist
def show_image_row(xlist, ylist=None, fontsize=12, size=(2.5, 2.5), title="", y_offset=-0.2, tlist=None):
    H, W = len(xlist), len(xlist[0])
    fig, axarr = plt.subplots(H, W, figsize=(size[0] * W, size[1] * H))
    for w in range(W):
        for h in range(H):
            ax = get_axis(axarr, H, W, h, w)                
            ax.imshow(xlist[h][w].permute(1, 2, 0))
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            if ylist and w == 0: 
                ax.set_ylabel(ylist[h], fontsize=fontsize)
            if tlist:
                ax.set_title(tlist[h][w], y=y_offset, fontsize=fontsize)
    plt.suptitle(title, fontsize=16)
    canvas = FigureCanvas(fig)
    canvas.draw()
    plt.close()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image