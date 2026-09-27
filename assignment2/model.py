from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            self.decoder = nn.Sequential(
                # Global image feature -> coarse 3D latent volume
                nn.Linear(512, 256 * 4 * 4 * 4),
                nn.ReLU(),

                # B x (256*4*4*4) -> B x 256 x 4 x 4 x 4
                nn.Unflatten(1, (256, 4, 4, 4)),

                # 4^3 -> 8^3
                nn.ConvTranspose3d(
                    in_channels=256,
                    out_channels=128,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                nn.ReLU(),

                # 8^3 -> 16^3
                nn.ConvTranspose3d(
                    in_channels=128,
                    out_channels=64,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                nn.ReLU(),

                # 16^3 -> 32^3
                nn.ConvTranspose3d(
                    in_channels=64,
                    out_channels=1,
                    kernel_size=4,
                    stride=2,
                    padding=1
                )
            )    
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points

            self.decoder = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),

                nn.Linear(1024, 2048),
                nn.ReLU(),

                nn.Linear(2048, args.n_points * 3)
            )          
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            n_verts = mesh_pred.verts_packed().shape[0]

            self.decoder = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),

                nn.Linear(1024, 2048),
                nn.ReLU(),

                nn.Linear(2048, n_verts * 3)
            )          

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # TODO:
            voxels_pred = self.decoder(encoded_feat)        
            return voxels_pred

        elif args.type == "point":
            pointclouds_pred = self.decoder(encoded_feat)
            pointclouds_pred = pointclouds_pred.reshape(
                B, self.n_point, 3
            )
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            deform_vertices_pred = self.decoder(encoded_feat)          
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          

