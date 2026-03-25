import numpy              as np
import torch
import tensorflow         as tf
import torch.nn           as nn

#######################################################
#************ BASELINE MODEL:              ***********#
#######################################################

# Encapsulate Danny Ko. model
class Danny_KerasModel(nn.Module):
    def __init__(self, uni_directional=None):
        super().__init__() 
        self.uni_directional = uni_directional
        if uni_directional == 0:
            path = "./Danny_Original/UnetRS_ModelvZ1-4/UnetRS_ModelvZ1-4.ckpt"
            self.model          = tf.keras.models.load_model(
                 path
            )
        elif uni_directional == 1:
            path = "./Danny_Original/UnetRS_ModelvY1-1/UnetRS_ModelvY1-1.ckpt"
            self.model          = tf.keras.models.load_model(
                 path
            )
        elif uni_directional == 2:
            path = "./Danny_Original/UnetRS_ModelvX1-3/UnetRS_ModelvX1-3.ckpt"
            self.model          = tf.keras.models.load_model(
                 path
            )
        else:
            path = "./Danny_Original/UnetRSXYZ_ModelvXYZ1-8/UnetRSXYZ_ModelvXYZ1-8.ckpt"
            self.model          = tf.keras.models.load_model(
                 path,
                 custom_objects = {'div_loss2': Danny_KerasModel.div_loss2}
            )
        
    # Inputs: Tensor (B,1,Z=120,Y=120,X=120),where 0's are solids, 1's are pores
    def predict(self, inputs):
        # Pytorch (B,C,Z,Y,X) -> Tensorflow (B,Z,Y,X,C)
        x          = inputs.permute(0,2,3,4,1)   
        # Use it as boolean numpy
        x          = x.detach().cpu().numpy()        
        # Danny works alligned to X
        x          = np.transpose(x, axes=[0, 3, 2, 1, 4])
        # Make sure it is binary
        x          = (x>0).astype(np.float32)
        # Predict
        y          = np.float32(self.model.predict(x=[x]))
        # Rotate alligning to Z
        y          = np.transpose(y, axes=[0, 3, 2, 1, 4])
        # Return prediction as Pytorch tensor
        y          = torch.from_numpy(y)
        # Tensorflow (B,Z,Y,X,C) -> Pytorch (B,C,Z,Y,X)
        y          = y.permute(0,4,1,2,3)
        # Reorder the channels: C=Vx,Vy,Vz to C=Vz,Vy,Vx
        if self.uni_directional is None:
            y          = y[:, [2, 1, 0], :, :, :]
        return y
 
    @staticmethod
    def div_loss2(y_true, y_pred):

        scale         = 3
        mse           = tf.math.reduce_mean( tf.math.square(y_true - y_pred) )
        dVxdx_pred    = (y_pred[:,2:,1:-1,1:-1,0] - y_pred[:,:-2,1:-1,1:-1,0])/2
        dVydy_pred    = (y_pred[:,1:-1,2:,1:-1,1] - y_pred[:,1:-1,:-2,1:-1,1])/2
        dVzdz_pred    = (y_pred[:,1:-1,1:-1,2:,2] - y_pred[:,1:-1,1:-1,:-2,2])/2
        div_pred      = dVxdx_pred + dVydy_pred + dVzdz_pred
        div_loss      = tf.math.reduce_mean( tf.math.abs(div_pred) )
        loss          = mse + div_loss*scale

        return loss