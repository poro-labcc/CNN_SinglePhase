import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .Functional import pad_same, crop_same, Channel_Concat
import matplotlib.pyplot as plt
from Unet import Base_Unet    
from Utilities import velocity_usage as vu


# Just playing around with some models


class MY_PIMODEL(nn.Module):
    def __init__(self, bin_input=True):
        super().__init__() 
                
        self.bin_input = bin_input
        
        self.x_model = Base_Unet(
            input_channels=1,
            output_channels=1,
            filter_num=10,
            filter_num_increase=2,
            filter_size=4,
            activation='selu',
            momentum=0.01,
            dropout=0.2,
            res_num=4,
            bin_input=bin_input)
     
        self.y_model = Base_Unet(
            input_channels=1,
            output_channels=1,
            filter_num=10,
            filter_num_increase=2,
            filter_size=4,
            activation='selu',
            momentum=0.01,
            dropout=0.2,
            res_num=4,
            bin_input=bin_input)
        
        self.z_model = Base_Unet(
            input_channels=1,
            output_channels=1,
            filter_num=10,
            filter_num_increase=2,
            filter_size=4,
            activation='selu',
            momentum=0.01,
            dropout=0.1,
            res_num=4,
            bin_input=bin_input)
        
        self.p_model = Base_Unet(
            input_channels=1,
            output_channels=1,
            filter_num=10,
            filter_num_increase=2,
            filter_size=4,
            activation='selu',
            momentum=0.01,
            dropout=0.1,
            res_num=4,
            bin_input=bin_input)
        
        self.concat = Channel_Concat()

        # main model
        # First Derivative combination
        self.C00 =  nn.Conv3d(in_channels=4,
                              out_channels=1, 
                              kernel_size=1,
                              stride=1, 
                              padding=0)

        self.C10 =  nn.Conv3d(in_channels=4,
                              out_channels=1, 
                              kernel_size=1,
                              stride=1, 
                              padding=0)

        self.C20 =  nn.Conv3d(in_channels=4,
                              out_channels=1, 
                              kernel_size=1,
                              stride=1, 
                              padding=0)

        self.C30 =  nn.Conv3d(in_channels=4,
                              out_channels=1, 
                              kernel_size=1,
                              stride=1, 
                              padding=0)

        self.C0  = nn.Conv3d(in_channels=16,
                              out_channels=1, 
                              kernel_size=1,
                              stride=1, 
                              padding=0)
        
        # Second derivative combination
        self.C01 =  nn.Conv3d(in_channels=7,
                              out_channels=1, 
                              kernel_size=1,
                              stride=1, 
                              padding=0)

        self.C11 =  nn.Conv3d(in_channels=7,
                              out_channels=1, 
                              kernel_size=1,
                              stride=1, 
                              padding=0)

        self.C21 =  nn.Conv3d(in_channels=7,
                              out_channels=1, 
                              kernel_size=1,
                              stride=1, 
                              padding=0)

        self.C31 =  nn.Conv3d(in_channels=7,
                              out_channels=1, 
                              kernel_size=1,
                              stride=1, 
                              padding=0)
        
        
    
    # Modified to freeze sub-models
    def train(self, mode=True):
        # 1. Call the standard train method for the main_model
        super().train(mode)
        
        # 2. Force the sub-models back to eval mode immediately
        self.x_model.eval()
        self.y_model.eval()
        self.z_model.eval()
        self.p_model.eval()
        return self
    
        
    def forward(self, x):
        if self.bin_input: 
            x_bin = (x > 0).to(torch.float32)
        else:
            x_bin = x # Assuming x is already binary/mask-like for the derivatives

        with torch.no_grad():
            u = self.x_model.predict(x)
            v = self.y_model.predict(x)
            w = self.z_model.predict(x)
            p = self.p_model.predict(x)
                                
        combined = self.concat(w, v, u, p)

        dw = self.concat(pad_same(vu.d_dz(w, x_bin, c=0).unsqueeze(1), 3, 1), 
                         pad_same(vu.d_dy(w, x_bin, c=0).unsqueeze(1), 3, 1), 
                         pad_same(vu.d_dx(w, x_bin, c=0).unsqueeze(1), 3, 1))
        
        dv = self.concat(pad_same(vu.d_dz(v, x_bin, c=0).unsqueeze(1), 3, 1), 
                         pad_same(vu.d_dy(v, x_bin, c=0).unsqueeze(1), 3, 1), 
                         pad_same(vu.d_dx(v, x_bin, c=0).unsqueeze(1), 3, 1))
        
        du = self.concat(pad_same(vu.d_dz(u, x_bin, c=0).unsqueeze(1), 3, 1), 
                         pad_same(vu.d_dy(u, x_bin, c=0).unsqueeze(1), 3, 1), 
                         pad_same(vu.d_dx(u, x_bin, c=0).unsqueeze(1), 3, 1))
        
        dp = self.concat(pad_same(vu.d_dz(p, x_bin, c=0).unsqueeze(1), 3, 1), 
                         pad_same(vu.d_dy(p, x_bin, c=0).unsqueeze(1), 3, 1), 
                         pad_same(vu.d_dx(p, x_bin, c=0).unsqueeze(1), 3, 1))

        d2w = self.concat(pad_same(vu.d2_dz2(w, x_bin, c=0).unsqueeze(1), 3, 1), 
                          pad_same(vu.d2_dy2(w, x_bin, c=0).unsqueeze(1), 3, 1), 
                          pad_same(vu.d2_dx2(w, x_bin, c=0).unsqueeze(1), 3, 1))
        
        d2v = self.concat(pad_same(vu.d2_dz2(v, x_bin, c=0).unsqueeze(1), 3, 1), 
                          pad_same(vu.d2_dy2(v, x_bin, c=0).unsqueeze(1), 3, 1), 
                          pad_same(vu.d2_dx2(v, x_bin, c=0).unsqueeze(1), 3, 1))
        
        d2u = self.concat(pad_same(vu.d2_dz2(u, x_bin, c=0).unsqueeze(1), 3, 1), 
                          pad_same(vu.d2_dy2(u, x_bin, c=0).unsqueeze(1), 3, 1), 
                          pad_same(vu.d2_dx2(u, x_bin, c=0).unsqueeze(1), 3, 1))
        
        d2p = self.concat(pad_same(vu.d2_dz2(p, x_bin, c=0).unsqueeze(1), 3, 1), 
                          pad_same(vu.d2_dy2(p, x_bin, c=0).unsqueeze(1), 3, 1), 
                          pad_same(vu.d2_dx2(p, x_bin, c=0).unsqueeze(1), 3, 1))
        
        cw = self.C00(self.concat(w, dw))
        cv = self.C10(self.concat(v, dv))
        cu = self.C20(self.concat(u, du))
        cp = self.C30(self.concat(p, dp))
        
        cwvup = self.C0(self.concat(w, v, u, p, dw, dv, du, dp)) 
        
        c2w = self.C01(self.concat(d2w, dw, w))
        c2v = self.C11(self.concat(d2v, dv, v))
        c2u = self.C21(self.concat(d2u, du, u))
        c2p = self.C31(self.concat(d2p, dp, p))
        
    
        
        addition = self.concat(w + cw + c2w + cwvup, 
                               v + cv + c2v + cwvup, 
                               u + cu + c2u + cwvup, 
                               p + cp + c2p + cwvup)

        final_out = combined + addition
        
        """
        # ==========================================
        # SEÇÃO DE PLOT / DEBUG (Agora com 7 colunas)
        # ==========================================
        num_channels = final_out.shape[1]
        x_slice_idx = final_out.shape[4] // 2
        
        # Concatenate intermediate terms to align structurally with 'addition' and 'combined'
        terms_c1 = self.concat(cw, cv, cu, cp)
        terms_c2 = self.concat(c2w, c2v, c2u, c2p)
        terms_cwvup = self.concat(cwvup, cwvup, cwvup, cwvup)

        # Aumentamos para 7 colunas e ajustamos a largura total (figsize de 24 para 42)
        fig, axes = plt.subplots(num_channels, 7, figsize=(42, 5 * num_channels), squeeze=False)
        
        print(f"UNET {num_channels} channels")
        
        for c in range(num_channels):
            img_z_out     = combined[0, c, :, :, x_slice_idx].detach().cpu().float().numpy()
            img_t1        = terms_c1[0, c, :, :, x_slice_idx].detach().cpu().float().numpy()
            img_t2        = terms_c2[0, c, :, :, x_slice_idx].detach().cpu().float().numpy()
            img_t3        = terms_cwvup[0, c, :, :, x_slice_idx].detach().cpu().float().numpy()
            add           = addition[0, c, :, :, x_slice_idx].detach().cpu().float().numpy()
            img_final_out = final_out[0, c, :, :, x_slice_idx].detach().cpu().float().numpy()
            
            # --- COLUNA 0: Sub-modelo (Combined) ---
            im0 = axes[c, 0].imshow(img_z_out, cmap='jet')
            axes[c, 0].set_title(f"Combined Sub-models - Ch {c}")
            fig.colorbar(im0, ax=axes[c, 0], fraction=0.046, pad=0.04)
            
            # --- COLUNA 1: Termo de 1ª Derivada (cw, cv, cu, cp) ---
            t1_min, t1_max = np.percentile(img_t1, [1, 99])
            im1 = axes[c, 1].imshow(img_t1, cmap='RdBu_r', vmin=t1_min, vmax=t1_max)
            axes[c, 1].set_title(f"Term 1 (c_X) - Ch {c}\nScale: [{t1_min:.2e}, {t1_max:.2e}]")
            fig.colorbar(im1, ax=axes[c, 1], fraction=0.046, pad=0.04)

            # --- COLUNA 2: Termo de 2ª Derivada (c2w, c2v, c2u, c2p) ---
            t2_min, t2_max = np.percentile(img_t2, [1, 99])
            im2 = axes[c, 2].imshow(img_t2, cmap='RdBu_r', vmin=t2_min, vmax=t2_max)
            axes[c, 2].set_title(f"Term 2 (c2_X) - Ch {c}\nScale: [{t2_min:.2e}, {t2_max:.2e}]")
            fig.colorbar(im2, ax=axes[c, 2], fraction=0.046, pad=0.04)

            # --- COLUNA 3: Termo Multivariável (cwvup) ---
            t3_min, t3_max = np.percentile(img_t3, [1, 99])
            im3 = axes[c, 3].imshow(img_t3, cmap='RdBu_r', vmin=t3_min, vmax=t3_max)
            axes[c, 3].set_title(f"Term 3 (cwvup) - Ch {c}\nScale: [{t3_min:.2e}, {t3_max:.2e}]")
            fig.colorbar(im3, ax=axes[c, 3], fraction=0.046, pad=0.04)

            # --- COLUNA 4: Adição Total (Main Model) ---
            add_min, add_max = np.percentile(add, [1, 99])
            im4 = axes[c, 4].imshow(add, cmap='RdBu_r', vmin=add_min, vmax=add_max)
            axes[c, 4].set_title(f"Main Addition (Sum) - Ch {c}\nScale: [{add_min:.2e}, {add_max:.2e}]")
            fig.colorbar(im4, ax=axes[c, 4], fraction=0.046, pad=0.04)
            
            # --- COLUNA 5: Saída Final ---
            im5 = axes[c, 5].imshow(img_final_out, cmap='jet')
            axes[c, 5].set_title(f"Final Output - Ch {c}")
            fig.colorbar(im5, ax=axes[c, 5], fraction=0.046, pad=0.04)

            # --- COLUNA 6: HISTOGRAMA DA ADIÇÃO ---
            axes[c, 6].hist(add.flatten(), bins=50, color='gray', edgecolor='black', alpha=0.7,range=(add_min, add_max))
            axes[c, 6].axvline(add_min, color='blue', linestyle='dashed', linewidth=1.5, label='1st %')
            axes[c, 6].axvline(add_max, color='red', linestyle='dashed', linewidth=1.5, label='99th %')
            axes[c, 6].set_title(f"Addition Histogram - Ch {c}")
            axes[c, 6].set_xlabel("Residual Value")
            axes[c, 6].set_ylabel("Frequency")
            axes[c, 6].legend()
            
        plt.tight_layout()
        plt.savefig('debug.png', dpi=300)
        
        plt.close(fig)
        """
        return final_out
    
    def predict(self, x):
        
        if self.bin_input: x = (x > 0).to(torch.float32)
        
        with torch.no_grad():
            out     = self.forward(x)

            # Mask Output, making solid always zero
            mask    = (x > 0).to(torch.float32) 
            mask    = mask.expand(-1, out.shape[1], -1, -1, -1)
            return out * mask


class MY_PIMODEL_2(nn.Module):
    def __init__(self, bin_input=True):
        super().__init__() 
                
        self.bin_input = bin_input
        
        self.x_model = Base_Unet(
            input_channels=1,
            output_channels=1,
            filter_num=10,
            filter_num_increase=2,
            filter_size=4,
            activation='selu',
            momentum=0.01,
            dropout=0.2,
            res_num=4,
            bin_input=bin_input)
     
        self.y_model = Base_Unet(
            input_channels=1,
            output_channels=1,
            filter_num=10,
            filter_num_increase=2,
            filter_size=4,
            activation='selu',
            momentum=0.01,
            dropout=0.2,
            res_num=4,
            bin_input=bin_input)
        
        self.z_model = Base_Unet(
            input_channels=1,
            output_channels=1,
            filter_num=10,
            filter_num_increase=2,
            filter_size=4,
            activation='selu',
            momentum=0.01,
            dropout=0.1,
            res_num=4,
            bin_input=bin_input)
        
        self.p_model = Base_Unet(
            input_channels=1,
            output_channels=1,
            filter_num=10,
            filter_num_increase=2,
            filter_size=4,
            activation='selu',
            momentum=0.01,
            dropout=0.1,
            res_num=4,
            bin_input=bin_input)
        
        self.concat = Channel_Concat()

        # Helper function to generate blocks cleanly
        def make_corr_block(in_c):
            return nn.Sequential( 
                nn.Conv3d(in_channels=in_c, out_channels=1, kernel_size=7, stride=1, padding=3),
                nn.Tanh(), 
                nn.Conv3d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2),
                nn.Tanh(), 
                nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
                nn.Tanh(), 
                nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)  
            )

        # Initialize blocks dynamically
        self.C0w = make_corr_block(7)
        self.C0v = make_corr_block(7)
        self.C0u = make_corr_block(7)

        self.C1w = make_corr_block(7)
        self.C1v = make_corr_block(7)
        self.C1u = make_corr_block(7)

        self.C2w = make_corr_block(7)
        self.C2v = make_corr_block(7)
        self.C2u = make_corr_block(7)

    def train(self, mode=True):
        super().train(mode)
        self.x_model.eval()
        self.y_model.eval()
        self.z_model.eval()
        self.p_model.eval()
        return self
        
    def forward(self, x):
        if self.bin_input: 
            x_bin = (x > 0).to(torch.float32)
        else:
            x_bin = x

        with torch.no_grad():
            u = self.x_model.predict(x)
            v = self.y_model.predict(x)
            w = self.z_model.predict(x)
            p = self.p_model.predict(x)
                                        
        dw_dz = pad_same(vu.d_dz(w, x_bin, c=0).unsqueeze(1), 3, 1)
        dv_dy = pad_same(vu.d_dy(v, x_bin, c=0).unsqueeze(1), 3, 1)
        du_dx = pad_same(vu.d_dx(u, x_bin, c=0).unsqueeze(1), 3, 1)
        div_sum = dw_dz + dv_dy + du_dx

        # Step 0
        input_0 = self.concat(w, v, u, dw_dz, dv_dy, du_dx, div_sum)
        corr_w0 = self.C0w(input_0)
        corr_v0 = self.C0v(input_0)
        corr_u0 = self.C0u(input_0)

        input_1 = self.concat(w+corr_w0, v+corr_v0, u+corr_u0, dw_dz, dv_dy, du_dx, div_sum)
        corr_w1 = corr_w0 + self.C1w(input_1)
        corr_v1 = corr_v0 + self.C1v(input_1)
        corr_u1 = corr_u0 + self.C1u(input_1)
        
        input_2 = self.concat(w+corr_w1, v+corr_v1, u+corr_u1, dw_dz, dv_dy, du_dx, div_sum)
        corr_w2 = corr_w1 + self.C2w(input_2)
        corr_v2 = corr_v1 + self.C2v(input_2)
        corr_u2 = corr_u1 + self.C2u(input_2)
        
        final_out = self.concat(w+corr_w2, v+corr_v2, u+corr_u2, p)
        
        return final_out
    
    def predict(self, x):
        if self.bin_input: x = (x > 0).to(torch.float32)
        
        with torch.no_grad():
            out = self.forward(x)
            mask = (x > 0).to(torch.float32) 
            mask = mask.expand(-1, out.shape[1], -1, -1, -1)
            return out * mask
        
        
class Residual_Conv(nn.Module):

    def __init__(self):
        super().__init__() 
        
        self.concat = Channel_Concat()
        self.channeMul = ChannelWiseMult()

        self.c1u = nn.Sequential(
            nn.Conv3d(in_channels=10, out_channels=3, kernel_size=1, stride=1, padding='same'),
            nn.Tanh()
        )
        self.c1v = nn.Sequential(
            nn.Conv3d(in_channels=10, out_channels=3, kernel_size=1, stride=1, padding='same'),
            nn.Tanh()
        )
        self.c1w = nn.Sequential(
            # Original out_channels = 3
            nn.Conv3d(in_channels=10, out_channels=3, kernel_size=1, stride=1, padding='same'),
            nn.Tanh()
        )

        self.track_mean = None

    def forward(self, x_bin, w,v,u,p):
        visc = 1/3
        # First order derivatives 
        dw_dz = pad_same(vu.d_dz(w, x_bin, c=0).unsqueeze(1), 3, 1)
        dv_dz = pad_same(vu.d_dz(v, x_bin, c=0).unsqueeze(1), 3, 1)
        du_dz = pad_same(vu.d_dz(u, x_bin, c=0).unsqueeze(1), 3, 1)
        dp_dz = pad_same(vu.d_dz(p, x_bin, c=0).unsqueeze(1), 3, 1)
        
        dw_dy = pad_same(vu.d_dy(w, x_bin, c=0).unsqueeze(1), 3, 1)
        dv_dy = pad_same(vu.d_dy(v, x_bin, c=0).unsqueeze(1), 3, 1)
        du_dy = pad_same(vu.d_dy(u, x_bin, c=0).unsqueeze(1), 3, 1)
        dp_dy = pad_same(vu.d_dy(p, x_bin, c=0).unsqueeze(1), 3, 1)
        
        dw_dx = pad_same(vu.d_dx(w, x_bin, c=0).unsqueeze(1), 3, 1)
        dv_dx = pad_same(vu.d_dx(v, x_bin, c=0).unsqueeze(1), 3, 1)
        du_dx = pad_same(vu.d_dx(u, x_bin, c=0).unsqueeze(1), 3, 1)
        dp_dx = pad_same(vu.d_dx(p, x_bin, c=0).unsqueeze(1), 3, 1)
    
        # Second order derivatives
        d2w_dz = pad_same(vu.d2_dz2(w, x_bin, c=0).unsqueeze(1), 3, 1)
        d2v_dz = pad_same(vu.d2_dz2(v, x_bin, c=0).unsqueeze(1), 3, 1)
        d2u_dz = pad_same(vu.d2_dz2(u, x_bin, c=0).unsqueeze(1), 3, 1)
        
        d2w_dy = pad_same(vu.d2_dy2(w, x_bin, c=0).unsqueeze(1), 3, 1)
        d2v_dy = pad_same(vu.d2_dy2(v, x_bin, c=0).unsqueeze(1), 3, 1)
        d2u_dy = pad_same(vu.d2_dy2(u, x_bin, c=0).unsqueeze(1), 3, 1)
        
        d2w_dx = pad_same(vu.d2_dx2(w, x_bin, c=0).unsqueeze(1), 3, 1)
        d2v_dx = pad_same(vu.d2_dx2(v, x_bin, c=0).unsqueeze(1), 3, 1)
        d2u_dx = pad_same(vu.d2_dx2(u, x_bin, c=0).unsqueeze(1), 3, 1)
    
        # PHYSICS RESIDUALS
        R_mass = du_dx + dv_dy + dw_dz
        R_u = (u * du_dx + v * du_dy + w * du_dz) + dp_dx - visc * (d2u_dx + d2u_dy + d2u_dz)
        R_v = (u * dv_dx + v * dv_dy + w * dv_dz) + dp_dy - visc * (d2v_dx + d2v_dy + d2v_dz)
        R_w = (u * dw_dx + v * dw_dy + w * dw_dz) + dp_dz - visc * (d2w_dx + d2w_dy + d2w_dz)

        self.track_mean = [R_mass.mean().item(), R_u.mean().item(), R_v.mean().item(), R_w.mean().item()]
        # ---------------------------------------------------------
        # COMBINED MACRO & MICRO PHYSICAL STATES (10 Channels each)
        # ---------------------------------------------------------
        # 1. Define the Global Base State (7 channels)
        global_base = self.concat(u, v, w, R_u, R_v, R_w, R_mass)
    
        # 2. Append the Isolated Local Gradients (3 channels)
        u_combined_state = self.concat(global_base, du_dx, du_dy, du_dz)
        v_combined_state = self.concat(global_base, dv_dx, dv_dy, dv_dz)
        w_combined_state = self.concat(global_base, dw_dx, dw_dy, dw_dz)
    
        # Convolve: Extracts exactly 3 feature maps (channels) per velocity component,
        # perfectly bounded between -1 and 1 by the Tanh activation.
        c1u_out = self.c1u(u_combined_state)
        c1v_out = self.c1v(v_combined_state) 
        c1w_out = self.c1w(w_combined_state) 
        
        # Pass the 3 channels directly to ChannelWiseMult so it can execute 
        # the C1 * C2 * C3 multiplication matrix. (Removed torch.sum here).
        u_corr = self.channeMul(c1u_out)
        v_corr = self.channeMul(c1v_out)
        w_corr = self.channeMul(c1w_out)
    
        return w + w_corr, v + v_corr, u + u_corr, p
    
class MY_PIMODEL_3(nn.Module):
    def __init__(self, bin_input=True):
        super().__init__() 
                
        self.bin_input = bin_input
        
        # Base models
        self.x_model = Base_Unet(input_channels=1, output_channels=1, filter_num=10, filter_num_increase=2, filter_size=4, activation='selu', momentum=0.01, dropout=0.2, res_num=4, bin_input=bin_input)
        self.y_model = Base_Unet(input_channels=1, output_channels=1, filter_num=10, filter_num_increase=2, filter_size=4, activation='selu', momentum=0.01, dropout=0.2, res_num=4, bin_input=bin_input)
        self.z_model = Base_Unet(input_channels=1, output_channels=1, filter_num=10, filter_num_increase=2, filter_size=4, activation='selu', momentum=0.01, dropout=0.1, res_num=4, bin_input=bin_input)
        self.p_model = Base_Unet(input_channels=1, output_channels=1, filter_num=10, filter_num_increase=2, filter_size=4, activation='selu', momentum=0.01, dropout=0.1, res_num=4, bin_input=bin_input)

        self.res_conv_1 = Residual_Conv()
        self.res_conv_2 = Residual_Conv()
        self.res_conv_3 = Residual_Conv()
        self.concat = Channel_Concat()

    def train(self, mode=True):
        super().train(mode)
        self.x_model.eval()
        self.y_model.eval()
        self.z_model.eval()
        self.p_model.eval()
        return self
        
    def forward(self, x):
        if self.bin_input: 
            x_bin = (x > 0).to(torch.float32)
        else:
            x_bin = x

        with torch.no_grad():
            u = self.x_model.predict(x)
            v = self.y_model.predict(x)
            w = self.z_model.predict(x)
            p = self.p_model.predict(x)

        visc = 1/3

        w_1, v_1, u_1, p_1 = self.res_conv_1(x_bin, w,v,u,p)
        w_2, v_2, u_2, p_2 = self.res_conv_2(x_bin, w_1,v_1,u_1,p_1)
        w_3, v_3, u_3, p_3 = self.res_conv_3(x_bin, w_2,v_2,u_2,p_2)

        print("--")
        print("  ResConv 1: ", self.res_conv_1.track_mean)
        print("  ResConv 2: ", self.res_conv_2.track_mean)
        print("  ResConv 3: ", self.res_conv_3.track_mean)
        print("--")
        
        return self.concat(w_3, v_3, u_3, p_3)
    
    def predict(self, x):
        if self.bin_input: x = (x > 0).to(torch.float32)
        
        with torch.no_grad():
            out = self.forward(x)
            mask = (x > 0).to(torch.float32) 
            mask = mask.expand(-1, out.shape[1], -1, -1, -1)
            return out * mask


class MY_PIMODEL_4(nn.Module):
    def __init__(self, bin_input=True):
        super().__init__() 
                
        self.bin_input = bin_input
        
        # ---------------------------------------------------------
        # 1. Base Models (Estes ficarão congelados / eval)
        # ---------------------------------------------------------
        self.x_model = Base_Unet(input_channels=1, output_channels=1, filter_num=10, filter_num_increase=2, filter_size=4, activation='selu', momentum=0.01, dropout=0.2, res_num=4, bin_input=bin_input)
        self.y_model = Base_Unet(input_channels=1, output_channels=1, filter_num=10, filter_num_increase=2, filter_size=4, activation='selu', momentum=0.01, dropout=0.2, res_num=4, bin_input=bin_input)
        self.z_model = Base_Unet(input_channels=1, output_channels=1, filter_num=10, filter_num_increase=2, filter_size=4, activation='selu', momentum=0.01, dropout=0.1, res_num=4, bin_input=bin_input)
        self.p_model = Base_Unet(input_channels=1, output_channels=1, filter_num=10, filter_num_increase=2, filter_size=4, activation='selu', momentum=0.01, dropout=0.1, res_num=4, bin_input=bin_input)

        self.concat = Channel_Concat()

        # ---------------------------------------------------------
        # 2. U-Nets de Correção Baseadas nos Resíduos
        # ---------------------------------------------------------
        # A entrada de cada U-Net terá 5 canais:
        # [Velocidade_Específica, R_mass, R_u, R_v, R_w]
        
        # IMPORTANTE: bin_input DEVE ser False aqui para não binarizar os gradientes/resíduos!
        unet_kwargs = dict(
            input_channels=5, 
            output_channels=1, 
            filter_num=1,             # Conforme solicitado
            filter_num_increase=1,    # Conforme solicitado
            filter_size=3,            # Conforme solicitado
            activation='selu', 
            momentum=0.01, 
            dropout=0.001, 
            res_num=3,                # Usei 3 resoluções para ser leve, ajuste se necessário
            bin_input=False           # CRÍTICO
        )

        self.unet_w = Base_Unet(**unet_kwargs)
        self.unet_v = Base_Unet(**unet_kwargs)
        self.unet_u = Base_Unet(**unet_kwargs)
        self.unet_p = Base_Unet(**unet_kwargs)

    # Garante que os modelos base fiquem congelados (eval) durante o treino
    def train(self, mode=True):
        super().train(mode)
        self.x_model.eval()
        self.y_model.eval()
        self.z_model.eval()
        self.p_model.eval()
        return self
        
    def forward(self, x):
        if self.bin_input: 
            x_bin = (x > 0).to(torch.float32)
        else:
            x_bin = x

        # 1. Previsões Base (Congeladas)
        with torch.no_grad():
            u = self.x_model.predict(x)
            v = self.y_model.predict(x)
            w = self.z_model.predict(x)
            p = self.p_model.predict(x)

        visc = 1/3
        
        # ---------------------------------------------------------
        # 2. Cálculo das Derivadas (1ª Ordem)
        # ---------------------------------------------------------
        dw_dz = pad_same(vu.d_dz(w, x_bin, c=0).unsqueeze(1), 3, 1)
        dv_dz = pad_same(vu.d_dz(v, x_bin, c=0).unsqueeze(1), 3, 1)
        du_dz = pad_same(vu.d_dz(u, x_bin, c=0).unsqueeze(1), 3, 1)
        dp_dz = pad_same(vu.d_dz(p, x_bin, c=0).unsqueeze(1), 3, 1)
        
        dw_dy = pad_same(vu.d_dy(w, x_bin, c=0).unsqueeze(1), 3, 1)
        dv_dy = pad_same(vu.d_dy(v, x_bin, c=0).unsqueeze(1), 3, 1)
        du_dy = pad_same(vu.d_dy(u, x_bin, c=0).unsqueeze(1), 3, 1)
        dp_dy = pad_same(vu.d_dy(p, x_bin, c=0).unsqueeze(1), 3, 1)
        
        dw_dx = pad_same(vu.d_dx(w, x_bin, c=0).unsqueeze(1), 3, 1)
        dv_dx = pad_same(vu.d_dx(v, x_bin, c=0).unsqueeze(1), 3, 1)
        du_dx = pad_same(vu.d_dx(u, x_bin, c=0).unsqueeze(1), 3, 1)
        dp_dx = pad_same(vu.d_dx(p, x_bin, c=0).unsqueeze(1), 3, 1)
    
        # ---------------------------------------------------------
        # 3. Cálculo das Derivadas (2ª Ordem)
        # ---------------------------------------------------------
        d2w_dz = pad_same(vu.d2_dz2(w, x_bin, c=0).unsqueeze(1), 3, 1)
        d2v_dz = pad_same(vu.d2_dz2(v, x_bin, c=0).unsqueeze(1), 3, 1)
        d2u_dz = pad_same(vu.d2_dz2(u, x_bin, c=0).unsqueeze(1), 3, 1)
        
        d2w_dy = pad_same(vu.d2_dy2(w, x_bin, c=0).unsqueeze(1), 3, 1)
        d2v_dy = pad_same(vu.d2_dy2(v, x_bin, c=0).unsqueeze(1), 3, 1)
        d2u_dy = pad_same(vu.d2_dy2(u, x_bin, c=0).unsqueeze(1), 3, 1)
        
        d2w_dx = pad_same(vu.d2_dx2(w, x_bin, c=0).unsqueeze(1), 3, 1)
        d2v_dx = pad_same(vu.d2_dx2(v, x_bin, c=0).unsqueeze(1), 3, 1)
        d2u_dx = pad_same(vu.d2_dx2(u, x_bin, c=0).unsqueeze(1), 3, 1)
    
        # ---------------------------------------------------------
        # 4. Cálculo dos Mapas de Resíduos da Física
        # ---------------------------------------------------------
        R_mass = du_dx + dv_dy + dw_dz
        R_u = (u * du_dx + v * du_dy + w * du_dz) + dp_dx - visc * (d2u_dx + d2u_dy + d2u_dz)
        R_v = (u * dv_dx + v * dv_dy + w * dv_dz) + dp_dy - visc * (d2v_dx + d2v_dy + d2v_dz)
        R_w = (u * dw_dx + v * dw_dy + w * dw_dz) + dp_dz - visc * (d2w_dx + d2w_dy + d2w_dz)

        # Empacota os resíduos (4 canais)
        physics_residuals = self.concat(R_mass, R_u, R_v, R_w)

        # ---------------------------------------------------------
        # 5. Passagem pelas Novas U-Nets Separadas
        # ---------------------------------------------------------
        # Concatena a variável alvo (1 canal) com os mapas de resíduos (4 canais) -> Total 5 canais
        input_w = self.concat(w, physics_residuals)
        input_v = self.concat(v, physics_residuals)
        input_u = self.concat(u, physics_residuals)
        input_p = self.concat(p, physics_residuals)

        # Calcula a correção passando pelas U-Nets
        w_corr = self.unet_w(input_w)
        v_corr = self.unet_v(input_v)
        u_corr = self.unet_u(input_u)
        p_corr = self.unet_p(input_p)

        # ---------------------------------------------------------
        # 6. Concatenação Final
        # ---------------------------------------------------------
        # Aplica a correção na previsão base e concatena na ordem padrão [w, v, u, p]
        final_out = self.concat(w + w_corr, 
                                v + v_corr, 
                                u + u_corr, 
                                p + p_corr)
        
        return final_out
    
    def predict(self, x):
        if self.bin_input: x = (x > 0).to(torch.float32)
        
        with torch.no_grad():
            out = self.forward(x)
            
            # Mask Output, making solid always zero
            mask = (x > 0).to(torch.float32) 
            mask = mask.expand(-1, out.shape[1], -1, -1, -1)
            return out * mask