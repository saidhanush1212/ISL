%% TODO: adapt these variables 
% add your matcaffe path here
addpath('');
% path to your images
data_path = '';
depth_prefix = 'depth/';
depth_suffix = '_depth.png';
colored_depth_prefix = 'color_on_depth/';
colored_depth_suffix = '_color_on_depth.png';
% CNN stuff
GPU_id = 0;
net_base_path = '';
HALNet_architecture = 'HALNet/HALNet_deploy.prototxt';
HALNet_weights = 'HALNet/HALNet_weights.caffemodel';
JORNet_architecture = 'JORNet/JORNet_deploy.prototxt';
JORNet_weights = 'JORNet/JORNet_weights.caffemodel';
% image preprocessing
invalid_depth = 32001;
maximum_depth = 1000; % in millimeters
depth_intrinsics = [ 475.62/2,       0, 311.125/2; ...
      0,  475.62/2, 245.965/2; ...
      0,       0,       1];
depth_intrinsics_inv = inv(depth_intrinsics);
%% -----------------------------

width = 320;
height = 240;
crop_size = 128;
num_joints = 21;
o1_parent = [1, 1:4, 1, 6:8, 1, 10:12, 1, 14:16, 1, 18:20];
root_id = 10;
crop_size_factor = 1.15 * 23500;

image_list_depth = dir([data_path, depth_prefix, '*', depth_suffix]);
image_list_color = dir([data_path, colored_depth_prefix, '*', colored_depth_suffix]);
num_images = length(image_list_depth);
if (num_images ~= length(image_list_color))
    error('Unequal amount of depth and colored depth images.')
end
all_pred3D = zeros(num_images, 3, num_joints);
all_pred2D = zeros(num_images, 3, num_joints);

caffe.set_mode_gpu()
caffe.set_device(GPU_id)
caffe.reset_all()

HALNet = caffe.Net([net_base_path, HALNet_architecture], [net_base_path, HALNet_weights], 'test');
JORNet = caffe.Net([net_base_path, JORNet_architecture], [net_base_path, JORNet_weights], 'test');

for i=1:num_images
    depth_image_full = single(imread([data_path,depth_prefix,image_list_depth(i).name]));
    depth_image = permute(imresize(depth_image_full, [height, width], 'nearest'), [2 1 3]);
    depth_image(depth_image > maximum_depth | depth_image == invalid_depth) = maximum_depth;
    depth_image = depth_image / maximum_depth;
    color_image_full = single(imread([data_path,colored_depth_prefix,image_list_color(i).name])) / 255;
    color_image = permute(imresize(color_image_full, [height, width], 'bilinear'), [2 1 3]);
    
    data = cat(3, depth_image, color_image);
    p = HALNet.forward({data});
    p_heatmap_2D = p{1,1};

    heatmap_root = p_heatmap_2D(:,:,root_id);
    heatmap_sized = imresize(heatmap_root, [width, height], 'bicubic');
    [~, maxLoc] = max(heatmap_sized(:));
    [max_u, max_v] = ind2sub(size(heatmap_sized), maxLoc);
    uv_root = [max_u; max_v];

    color_image_vis = permute(color_image, [2 1 3]);
    color_image_vis = insertShape(color_image_vis, 'circle', [max_u,max_v,2], 'Color', 'red');
    figure(1); imshow(color_image_vis);
    figure(2); imshow(transpose(heatmap_sized));

    % mean depth in 5 x 5 window
    start_uv = bsxfun(@max, [max_u, max_v]-2, [1,1]);
    end_uv = bsxfun(@min, [max_u, max_v]+2, [width, height]);
    mean_depth = 0;
    num_valid = 0;
    for w=start_uv(1):end_uv(1)
        for h=start_uv(2):end_uv(2)
            if (depth_image(w,h) ~= 1.0)
                mean_depth = mean_depth + maximum_depth * depth_image(w,h);
                num_valid = num_valid + 1;
            end
        end
    end
    if (num_valid > 0 && mean_depth > 0)
        mean_depth = mean_depth / num_valid;
        radCrop = round(crop_size_factor * (1/mean_depth));
    else
        continue;
    end

    % backproject using intrinsics
    normPoint = depth_intrinsics_inv * [uv_root-1; 1];
    normPoint = (normPoint / normPoint(3)) * mean_depth;
    crop_depth = ones(2*radCrop + 1, 2*radCrop + 1);
    crop_color = zeros(2*radCrop + 1, 2*radCrop + 1,3);

    norm_z = mean_depth / maximum_depth;
    uv_bb_start = uv_root - [radCrop; radCrop];
    uv_bb_end = uv_root + [radCrop; radCrop];
    % fill cropped depth and color
    target_uv_start = bsxfun(@max, [1; 1], -uv_bb_start+2);
    target_uv_end = bsxfun(@min, [2*radCrop+1; 2*radCrop+1], [2*radCrop+1; 2*radCrop+1] - (uv_bb_end - [width; height]));
    % index with u,v because of transpose ! make sure to read from
    % transposed variables !
    crop_depth(target_uv_start(1):target_uv_end(1), target_uv_start(2):target_uv_end(2)) = depth_image(max(1,uv_bb_start(1)):min(width,uv_bb_end(1)),max(1,uv_bb_start(2)):min(height,uv_bb_end(2)),1);
    crop_color(target_uv_start(1):target_uv_end(1), target_uv_start(2):target_uv_end(2),:) = color_image(max(1,uv_bb_start(1)):min(width,uv_bb_end(1)),max(1,uv_bb_start(2)):min(height,uv_bb_end(2)),:);
    crop_image_sized = imresize(crop_depth,[crop_size crop_size],'nearest');

    mask_valid = (crop_image_sized ~= 1.0);
    crop_image_sized(mask_valid) = crop_image_sized(mask_valid) - norm_z;
    crop_color_sized = imresize(crop_color, [crop_size, crop_size],'bilinear');

    crop_data = cat(3, crop_image_sized, crop_color_sized);
    p = JORNet.forward({crop_data});
    p_2D = p{1,1};
    p_rel3D = p{2,1};

    % visualize per joint
    pred_3D = single(zeros(3, num_joints));
    for j=1:num_joints
        % compute global 3D from relative
        p_j_3D = 1000 * p_rel3D(:,1,j) + normPoint;
        pred_3D(:, j) = p_j_3D;
        all_pred3D(i, :, j) = p_j_3D';

        heatmap_j = imresize(p_2D(:,:,j),[crop_size, crop_size],'bicubic');
        [conf, max_heat_pos] = max(heatmap_j(:));
        [heat_j_col,heat_j_row] = ind2sub(size(heatmap_j),max_heat_pos); % this gives row,col but the prediction is transposed
        % get corresponding uv position in original image
        orig_uv = uv_bb_start + [heat_j_col;heat_j_row] * ((2*radCrop+1)/crop_size);
        all_pred2D(i, :, j) = [orig_uv(1), orig_uv(2), conf];

        color_image_vis = insertShape(color_image_vis, 'circle', [orig_uv(1),orig_uv(2),2], 'Color', 'red');
        figure(3); imshow(color_image_vis);
        figure(4); imshow(transpose(heatmap_j));
    end
    figure(5); clf; hold on;
    plot3([pred_3D(1,:); pred_3D(1,o1_parent)], [pred_3D(2,:); pred_3D(2,o1_parent)], [pred_3D(3,:); pred_3D(3,o1_parent)],'r','LineWidth',3);
    hold off;
end