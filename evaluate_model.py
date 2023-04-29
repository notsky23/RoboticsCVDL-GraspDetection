import argparse
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms.functional as TF

from simulator import Simulator
from trainer import MobileUNet


def main(args):
    np.random.seed(0)
    sim = Simulator(render=bool(args.render))
    num_rotations = 4

    model = MobileUNet(pretrained=False)
    model.load(args.model_path)
    model.eval()

    labels = []
    pbar = tqdm(range(args.num_grasps),
                desc=f'Success rate = ---%')
    for grasp_id in pbar:
        sim.reset()
        img = sim.render_image()
        img = TF.to_tensor(img)
        rot, px, py = model.predict(img)

        x, y = sim._convert_from_pixel(np.array((px, py)))
        theta = rot * num_rotations / np.pi
        label = sim.execute_grasp(x, y, theta)
        labels.append(label)

        pbar.set_description(f'Success rate = {np.mean(labels):.1%}')

    # print(f'Final success rate: {np.mean(labels):.1%}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', type=str, default="grasp_mobilenet.pt",
                        help='File path where torch model is saved')
    parser.add_argument('--num_grasps', '-n', type=int, default=50,
                        help='number of grasps used to evaluate model performance')
    parser.add_argument('--render', type=int, default=1)
    args = parser.parse_args()

    main(args)

