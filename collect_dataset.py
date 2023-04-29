from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import numpy as np
from tqdm import tqdm

from simulator import Simulator

def foo(num_grasps, num_rotations, img_size, render=False, show_progress=False):
    '''Runs env to collect samples, for parallelization'''
    sim = Simulator(img_size=img_size, render=render, obj_rotation_range=(-0.2, 0.2))

    imgs = []
    pxyrs = []

    if show_progress:
        pbar = tqdm(total=num_grasps)

    while len(imgs) < num_grasps:
        sim.reset()

        img = sim.render_image()

        # look for occupied pixels only, knowing background is white
        occ_mask = np.argwhere((img < 230).any(axis=2))
        pxy = occ_mask[np.random.randint(len(occ_mask))]

        x, y = sim._convert_from_pixel(pxy)
        r = np.random.randint(num_rotations)
        theta = r * np.pi / num_rotations

        if not sim.execute_grasp(x, y, theta):
            continue

        imgs.append(img)
        pxyrs.append((*pxy, r))

        if show_progress:
            pbar.update(1)

    return imgs, pxyrs

def collect_dataset(num_grasps: int,
                    img_size: int,
                    num_rotations: int,
                    render: bool,
                    num_processes: int,
                   ):
    with ProcessPoolExecutor() as executor:
        bg_futures = []
        for i in range(1, num_processes):
            new_future = executor.submit(foo, num_grasps//num_processes,
                                         num_rotations, img_size)
            bg_futures.append(new_future)

        grasps_left = num_grasps - (num_processes-1) * (num_grasps//num_processes)
        imgs, actions = foo(grasps_left, num_rotations, img_size,
                                    render, show_progress=True)

        for f in as_completed(bg_futures):
            _imgs, _actions = f.result()
            imgs.extend(_imgs)
            actions.extend(_actions)

    imgs = np.array(imgs)
    actions = np.array(actions)

    return dict(imgs=imgs, actions=actions)


def split_dataset(dataset, train_fraction=0.8):
    imgs = dataset['imgs']
    actions = dataset['actions']

    ids = np.arange(len(imgs))
    num_train = int(train_fraction * len(imgs))
    train_ids = ids[:num_train]
    val_ids = ids[num_train:]

    train_data = dict(
        imgs=imgs[train_ids],
        actions=actions[train_ids],
    )
    val_data = dict(
        imgs=imgs[val_ids],
        actions=actions[val_ids],
    )

    np.savez('train_dataset', **train_data)
    np.savez('val_dataset', **val_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, default="grasp_dataset",
                        help='File path where data will be saved')
    parser.add_argument('--num_grasps', type=int, default=1000,
                        help='Number of grasps in dataset')
    parser.add_argument('--train_fraction', type=float, default=0.8,
                        help='Fraction of samples in train set')
    parser.add_argument('--img_size', type=int, default=64,
                        help='Height/Width of top down image')
    parser.add_argument('--num_rotations', type=int, default=2,
                        help='Number of gripper rotations from 0 to PI')
    parser.add_argument('--num_processes', type=int, default=8,
                        help='Number of parallel processes')
    parser.add_argument('--render', action='store_true',
                        help='If true, render gui during dataset collection')
    args = parser.parse_args()

    dataset = collect_dataset(
        num_grasps=args.num_grasps,
        img_size=args.img_size,
        num_rotations=args.num_rotations,
        render= args.render,
        num_processes= args.num_processes,
       )

    split_dataset(dataset, args.train_fraction)
