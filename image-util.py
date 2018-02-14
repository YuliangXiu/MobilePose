def expand_bbox(left, right, top, bottom, img_width, img_height):
    width = right-left
    height = bottom-top
    ratio = 0.15
    new_left = np.clip(left-ratio*width,0,img_width)
    new_right = np.clip(right+ratio*width,0,img_width)
    new_top = np.clip(top-ratio*height,0,img_height)
    new_bottom = np.clip(bottom+ratio*height,0,img_height)
    return [int(new_left), int(new_top), int(new_right), int(new_bottom)]

def display_pose(img, pose):
    pose  = pose.data.cpu().numpy().reshape([-1,2])
    img = img.cpu().numpy().transpose(1,2,0)
    img_width, img_height,_ = img.shape
    ax = plt.gca()
    plt.imshow(img)
    for idx in range(16):
        plt.plot(pose[idx,0], pose[idx,1], marker='o', color='yellow')
    xmin = np.min(pose[:,0])
    ymin = np.min(pose[:,1])
    xmax = np.max(pose[:,0])
    ymax = np.max(pose[:,1])
    bndbox = np.array(expand_bbox(xmin, xmax, ymin, ymax, img_width, img_height))
    coords = (bndbox[0], bndbox[1]), bndbox[2]-bndbox[0]+1, bndbox[3]-bndbox[1]+1
    ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='yellow', linewidth=2))